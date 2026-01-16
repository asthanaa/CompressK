from __future__ import annotations

"""Lightweight MLP model for the GNNSelector legacy feature path.

The AI-selector Krylov code supports an optional "GNN" selector backend.
Despite the name, the selector can use any torch model that maps:

  (N, F) float features -> (N,) scores

This module provides a small, dependency-minimal model for that path.
It intentionally depends only on PyTorch (no PyG).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureMLPConfig:
    in_dim: int
    hidden_dim: int = 64
    depth: int = 2
    activation: str = "silu"  # "relu" or "silu"


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("FeatureMLP requires PyTorch to be installed") from e
    return torch


def _act(name: str):
    torch = _require_torch()
    import torch.nn as nn  # type: ignore

    n = str(name).strip().lower()
    if n in ("relu",):
        return nn.ReLU()
    if n in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation={name!r}. Expected 'relu' or 'silu'.")


class FeatureMLP:  # not subclassing nn.Module until torch import succeeds
    def __init__(self, config: FeatureMLPConfig):
        torch = _require_torch()
        import torch.nn as nn  # type: ignore

        self.torch = torch
        self.config = config

        layers: list[Any] = []
        in_dim = int(config.in_dim)
        h = int(config.hidden_dim)
        depth = int(config.depth)

        if depth <= 0:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_act(config.activation))
            for _ in range(depth - 1):
                layers.append(nn.Linear(h, h))
                layers.append(_act(config.activation))
            layers.append(nn.Linear(h, 1))

        self.net = nn.Sequential(*layers)

        self._module = nn.Module()
        self._module.add_module("net", self.net)

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

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        torch = self.torch
        xt = x
        if not isinstance(xt, torch.Tensor):
            xt = torch.tensor(xt, dtype=torch.float32)
        y = self.net(xt).squeeze(-1)
        return y


def save_feature_mlp_checkpoint(path: str | Path, *, model: FeatureMLP, config: FeatureMLPConfig) -> None:
    torch = _require_torch()
    p = Path(path)
    payload = {"kind": "feature_mlp", "config": config.__dict__, "state_dict": model.state_dict()}
    torch.save(payload, str(p))


def load_feature_mlp_checkpoint(path: str | Path, *, map_location: str | None = "cpu") -> FeatureMLP:
    torch = _require_torch()
    payload: dict[str, Any] = torch.load(str(path), map_location=map_location)
    kind = str(payload.get("kind", "feature_mlp"))
    if kind != "feature_mlp":
        raise ValueError(f"Unsupported checkpoint kind={kind!r}. Expected 'feature_mlp'.")
    cfg = payload.get("config", {})
    config = FeatureMLPConfig(**cfg)
    model = FeatureMLP(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
