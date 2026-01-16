from __future__ import annotations

"""Operator-learning network for support selection.

This module defines :class:`OperatorNN`, a PyTorch/PyG model that predicts per-node
scores to approximate the magnitude of the Hamiltonian action:

  y = H x
  s_i \approx log(eps + |y_i|)

Crucially, the model is designed to be (approximately) *linear in x*.
We enforce this by making x only appear through a learned *linear operator*
A_θ(graph_features) applied to x:

  y = A_θ(G) x

The graph-dependent nonlinearity is allowed in the computation of the operator
weights, but x is only ever multiplied and added linearly.

This model is meant for *support selection only*. It must NOT be used to produce
final coefficients; those should still be computed by exact Hamiltonian
projection on the selected support.

No heavy dependencies beyond PyTorch + PyTorch Geometric are required.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OperatorNNConfig:
    """Configuration for :class:`OperatorNN`.

    Args:
        node_in_dim: Dimensionality of input node features (Data.x).
        edge_in_dim: Dimensionality of input edge features (Data.edge_attr). Use 0 if absent.
        hidden_dim: Hidden size used for graph-feature encoders.
        mode:
            - "implicit": compute y via scatter-add using edge weights (default, simplest).
            - "sparse_matmul": explicitly build a sparse operator A and compute y=A@x.
        use_self_term: Whether to include a learned diagonal/self contribution to y_i.
        eps: Small constant used when converting y -> log-magnitude score.
    """

    node_in_dim: int
    edge_in_dim: int = 0
    hidden_dim: int = 64
    mode: Literal["implicit", "sparse_matmul"] = "implicit"
    use_self_term: bool = True
    eps: float = 1e-12


def _require_torch_and_pyg() -> tuple["torch", "Data", "scatter"]:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OperatorNN requires PyTorch to be installed") from e

    try:
        from torch_geometric.data import Data  # type: ignore
        from torch_geometric.utils import scatter  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OperatorNN requires torch_geometric (PyG) to be installed") from e

    return torch, Data, scatter


class OperatorNN:  # intentionally not subclassing nn.Module until torch import succeeds
    """Graph neural operator used to score/select determinants.

    Forward signatures:
        - forward(data, x): returns per-node scores (log-magnitude) for selection.
        - forward_linear(data, x): returns raw linear output y = A_θ(G) x (linearity sanity).

    The model is linear in x for a fixed graph, i.e.
        forward_linear(G, a*x1 + b*x2) == a*forward_linear(G, x1) + b*forward_linear(G, x2)
    up to floating point error.
    """

    def __init__(self, config: OperatorNNConfig):
        torch, _, _ = _require_torch_and_pyg()
        import torch.nn as nn  # type: ignore

        self.torch = torch
        self.config = config

        node_in = int(config.node_in_dim)
        edge_in = int(config.edge_in_dim)
        h = int(config.hidden_dim)

        # Nonlinear graph-feature encoders (independent of x).
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        if edge_in > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_in, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
            )
        else:
            self.edge_mlp = None

        # Edge weight network produces a scalar weight w_e that is independent of x.
        # This defines an implicit sparse linear operator A_θ(G).
        self.edge_weight = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

        self.self_weight = None
        if bool(config.use_self_term):
            self.self_weight = nn.Sequential(
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        # Register as a real nn.Module.
        # We delay subclassing for import hygiene, but we still want module behavior.
        self._module = nn.Module()
        self._module.add_module("node_mlp", self.node_mlp)
        if self.edge_mlp is not None:
            self._module.add_module("edge_mlp", self.edge_mlp)
        self._module.add_module("edge_weight", self.edge_weight)
        if self.self_weight is not None:
            self._module.add_module("self_weight", self.self_weight)

    # nn.Module compatibility
    def parameters(self):
        return self._module.parameters()

    def to(self, *args, **kwargs):
        self._module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)

    def __call__(self, data, x):
        return self.forward(data, x)

    def forward(self, data, x):
        """Return per-node scores used for support selection.

        Args:
            data: PyG Data with at least:
                - x: node features, shape [N, F]
                - edge_index: shape [2, E]
                - edge_attr (optional): shape [E, Fe]
            x: per-node scalar coefficients, shape [N] or [N,1]

        Returns:
            scores: shape [N], where higher is more important.
                    Default is log(eps + |y|) with y=A_θ(G)x.
        """

        torch = self.torch
        y = self.forward_linear(data, x)
        eps = float(self.config.eps)
        scores = torch.log(torch.abs(y) + eps)
        return scores

    def forward_linear(self, data, x):
        """Return raw linear output y = A_θ(G) x.

        This is the quantity that is linear in x for a fixed graph.
        """

        torch, _, scatter = _require_torch_and_pyg()

        x_node = data.x
        edge_index = data.edge_index
        if x_node is None or edge_index is None:
            raise ValueError("data.x and data.edge_index are required")

        # Coefficients x: [N] float
        x_coeff = x
        if isinstance(x_coeff, (list, tuple)):
            x_coeff = torch.tensor(x_coeff, dtype=torch.float32, device=x_node.device)
        if x_coeff.dim() == 2 and x_coeff.size(-1) == 1:
            x_coeff = x_coeff[:, 0]
        if x_coeff.dim() != 1:
            raise ValueError("x must have shape [N] or [N,1]")

        # Geometry/connectivity embeddings independent of x
        h_node = self.node_mlp(x_node)

        src = edge_index[0]
        dst = edge_index[1]

        if getattr(data, "edge_attr", None) is not None and self.edge_mlp is not None:
            h_edge = self.edge_mlp(data.edge_attr)
        else:
            # No edge features: use zeros.
            h_edge = torch.zeros((src.numel(), h_node.size(-1)), dtype=h_node.dtype, device=h_node.device)

        h_src = h_node[src]
        h_dst = h_node[dst]
        e_in = torch.cat([h_src, h_dst, h_edge], dim=-1)

        w_e = self.edge_weight(e_in).squeeze(-1)  # [E]

        # Option 1 (default): implicit operator application via scatter-add.
        if self.config.mode == "implicit":
            msg = w_e * x_coeff[src]
            y = scatter(msg, dst, dim=0, dim_size=h_node.size(0), reduce="sum")

        # Option 2: explicit sparse operator A and sparse matmul.
        elif self.config.mode == "sparse_matmul":
            n = int(h_node.size(0))
            idx = torch.stack([dst, src], dim=0)  # A[dst,src]
            A = torch.sparse_coo_tensor(idx, w_e, size=(n, n), device=h_node.device, dtype=h_node.dtype)
            y = torch.sparse.mm(A, x_coeff[:, None])[:, 0]

        else:
            raise ValueError(f"Unknown mode={self.config.mode!r}")

        if self.self_weight is not None:
            diag = self.self_weight(h_node).squeeze(-1)
            y = y + diag * x_coeff

        return y


def select_topk_nodes(scores: "torch.Tensor", k: int) -> "torch.Tensor":
    """Select top-k node indices.

    Returns unique indices (torch.topk guarantees uniqueness by index).
    """

    torch, _, _ = _require_torch_and_pyg()
    if scores.dim() != 1:
        raise ValueError("scores must be 1D")
    k = int(k)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    k = min(k, int(scores.numel()))
    return torch.topk(scores, k=k, largest=True, sorted=False).indices


def topk_recall(pred_scores: "torch.Tensor", true_scores: "torch.Tensor", k: int) -> "torch.Tensor":
    """Compute top-k recall between predicted and true importance scores."""

    torch, _, _ = _require_torch_and_pyg()
    pred = select_topk_nodes(pred_scores, k)
    tru = select_topk_nodes(true_scores, k)
    if tru.numel() == 0:
        return torch.tensor(1.0, device=pred_scores.device)

    # Intersection size / k
    # Use CPU set for simplicity and determinism in tiny-k metric.
    inter = len(set(pred.detach().cpu().tolist()) & set(tru.detach().cpu().tolist()))
    return torch.tensor(float(inter) / float(tru.numel()), device=pred_scores.device)
