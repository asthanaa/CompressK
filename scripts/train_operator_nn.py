from __future__ import annotations

"""Train OperatorNN on precomputed matvec-magnitude labels.

This training script is intentionally lightweight: it assumes you already have
PyG `Data` objects that contain:
  - `x`: node features [N, F]
  - `edge_index`: [2, E]
  - `edge_attr` (optional): [E, Fe]
  - `coeff`: per-node coefficient x_i [N] or [N,1]
  - `target`: per-node target t_i = log(eps + |(H x)_i|) [N]

It trains an OperatorNN to predict per-node scores for *support selection only*.

Example:
  python scripts/train_operator_nn.py \
    --train data/train.pt --val data/val.pt \
    --node-in-dim 16 --edge-in-dim 8 --epochs 50

No Hamiltonian routines are touched here; labels are expected to be precomputed.
"""

import argparse
from dataclasses import dataclass


def _require_torch_and_pyg():
    import torch  # type: ignore

    from torch_geometric.loader import DataLoader  # type: ignore

    return torch, DataLoader


@dataclass(frozen=True)
class TrainConfig:
    train_path: str
    val_path: str
    node_in_dim: int
    edge_in_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int
    lr: float
    huber_delta: float
    topk: int
    topk_weight: float
    device: str
    mode: str


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train", dest="train_path", required=True)
    p.add_argument("--val", dest="val_path", required=True)
    p.add_argument("--node-in-dim", type=int, required=True)
    p.add_argument("--edge-in-dim", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--topk-weight", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mode", type=str, default="implicit", choices=["implicit", "sparse_matmul"])
    a = p.parse_args()

    return TrainConfig(
        train_path=a.train_path,
        val_path=a.val_path,
        node_in_dim=a.node_in_dim,
        edge_in_dim=a.edge_in_dim,
        hidden_dim=a.hidden_dim,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        huber_delta=a.huber_delta,
        topk=a.topk,
        topk_weight=a.topk_weight,
        device=a.device,
        mode=a.mode,
    )


def _loss(pred, target, *, huber_delta: float, topk: int, topk_weight: float):
    torch, _ = _require_torch_and_pyg()

    # Huber/MSE-ish loss
    err = pred - target
    abs_err = torch.abs(err)
    quad = 0.5 * (err**2)
    lin = huber_delta * (abs_err - 0.5 * huber_delta)
    huber = torch.where(abs_err <= huber_delta, quad, lin)

    if topk_weight <= 0.0:
        return huber.mean()

    # Optional weighting: emphasize nodes that are truly important.
    k = min(int(topk), int(target.numel()))
    idx = torch.topk(target, k=k, largest=True, sorted=False).indices
    w = torch.ones_like(target)
    w[idx] = 1.0 + float(topk_weight)

    return (huber * w).mean()


def _eval_metrics(model, loader, *, device: str, topk: int):
    from ccik.operator_nn import topk_recall

    torch, _ = _require_torch_and_pyg()

    model.eval()
    losses = []
    recalls = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, batch.coeff)
            target = batch.target
            losses.append(((pred - target) ** 2).mean().detach().cpu())
            recalls.append(topk_recall(pred, target, k=topk).detach().cpu())

    mse = float(torch.stack(losses).mean()) if losses else 0.0
    r = float(torch.stack(recalls).mean()) if recalls else 0.0
    return mse, r


def main() -> None:
    cfg = _parse_args()
    torch, DataLoader = _require_torch_and_pyg()

    from ccik.operator_nn import OperatorNN, OperatorNNConfig

    train_data = torch.load(cfg.train_path)
    val_data = torch.load(cfg.val_path)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)

    model = OperatorNN(
        OperatorNNConfig(
            node_in_dim=cfg.node_in_dim,
            edge_in_dim=cfg.edge_in_dim,
            hidden_dim=cfg.hidden_dim,
            mode=cfg.mode,
        )
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        model.train(True)
        for batch in train_loader:
            batch = batch.to(cfg.device)

            pred = model(batch, batch.coeff)
            target = batch.target

            L = _loss(
                pred,
                target,
                huber_delta=cfg.huber_delta,
                topk=cfg.topk,
                topk_weight=cfg.topk_weight,
            )

            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()

        val_mse, val_recall = _eval_metrics(model, val_loader, device=cfg.device, topk=cfg.topk)
        print(f"epoch {ep:03d}  val_mse={val_mse:.6e}  topk_recall@{cfg.topk}={val_recall:.4f}")

    # Save the model weights next to train file.
    out_path = cfg.train_path + ".operator_nn.pt"
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
