"""Train a small torch MLP on engineered candidate features.

This trains the simple FeatureMLP model (no PyG) that plugs into the
GNNSelector legacy path: model(X)->scores.

Input: .npz created by scripts/generate_ai_selector_training_data.py
  - X: (M, F)
  - y: (M,)

Run from repo root:
  python scripts/train_feature_mlp.py --data data/n2_train.npz --out data/n2_feature_mlp.pt

You can optionally specify a validation split:
  python scripts/train_feature_mlp.py --val-frac 0.1

Requires: PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _require_torch():
    import torch  # type: ignore

    return torch


def _topk_recall(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    k = int(min(max(k, 1), pred.size))
    p = np.argpartition(pred, -k)[-k:]
    t = np.argpartition(target, -k)[-k:]
    inter = len(set(p.tolist()) & set(t.tolist()))
    return float(inter) / float(k)


def main() -> None:
    p = argparse.ArgumentParser(description="Train FeatureMLP on AI-selector candidate features")
    p.add_argument("--data", required=True, help="Path to .npz with X,y")
    p.add_argument("--out", required=True, help="Output checkpoint .pt")
    p.add_argument(
        "--init",
        default=None,
        help="Optional FeatureMLP checkpoint to initialize from (transfer learning / fine-tuning).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=256)
    args = p.parse_args()

    from ccik.feature_mlp import (
        FeatureMLP,
        FeatureMLPConfig,
        load_feature_mlp_checkpoint,
        save_feature_mlp_checkpoint,
    )

    torch = _require_torch()

    rng = np.random.default_rng(int(args.seed))

    data = np.load(args.data)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"Bad shapes: X={X.shape} y={y.shape}")

    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(float(args.val_frac) * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32) if n_val > 0 else None
    yva = torch.tensor(y[val_idx], dtype=torch.float32) if n_val > 0 else None

    if args.init:
        model = load_feature_mlp_checkpoint(args.init, map_location="cpu").to(args.device)
        # Sanity check on input feature dimension.
        if int(getattr(model.config, "in_dim", -1)) != int(X.shape[1]):
            raise ValueError(
                f"Init checkpoint expects in_dim={getattr(model.config, 'in_dim', None)} but data has in_dim={int(X.shape[1])}"
            )
        print(f"Initialized from: {args.init}")
    else:
        model = FeatureMLP(
            FeatureMLPConfig(in_dim=int(X.shape[1]), hidden_dim=int(args.hidden_dim), depth=int(args.depth))
        ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    def batches(Xt, yt, bs: int):
        m = Xt.shape[0]
        order = torch.randperm(m)
        for i in range(0, m, bs):
            j = order[i : i + bs]
            yield Xt[j], yt[j]

    for ep in range(int(args.epochs)):
        model.train(True)
        losses = []
        for xb, yb in batches(Xtr.to(args.device), ytr.to(args.device), int(args.batch_size)):
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())

        tr_mse = float(torch.stack(losses).mean()) if losses else 0.0

        if Xva is not None and yva is not None and Xva.numel() > 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xva.to(args.device)).detach().cpu().numpy()
            tv = yva.detach().cpu().numpy()
            val_mse = float(np.mean((pv - tv) ** 2))
            r = _topk_recall(pv, tv, int(args.topk))
            print(f"epoch {ep:03d}  train_mse={tr_mse:.6e}  val_mse={val_mse:.6e}  topk_recall@{int(args.topk)}={r:.4f}")
        else:
            print(f"epoch {ep:03d}  train_mse={tr_mse:.6e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using the model's current config (including when fine-tuning from --init).
    save_feature_mlp_checkpoint(out_path, model=model, config=model.config)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
