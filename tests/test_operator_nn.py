from __future__ import annotations

import pytest


def _torch_and_pyg():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    return torch, Data


def test_operator_nn_linearity_forward_linear() -> None:
    torch, Data = _torch_and_pyg()

    from ccik.operator_nn import OperatorNN, OperatorNNConfig

    torch.manual_seed(0)

    # Tiny directed graph with 4 nodes
    x_feat = torch.randn(4, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 2, 3, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.size(1), 2)

    data = Data(x=x_feat, edge_index=edge_index, edge_attr=edge_attr)

    model = OperatorNN(OperatorNNConfig(node_in_dim=3, edge_in_dim=2, hidden_dim=16, mode="implicit")).eval()

    x1 = torch.randn(4)
    x2 = torch.randn(4)
    a = 1.7
    b = -0.3

    y12 = model.forward_linear(data, a * x1 + b * x2)
    ylin = a * model.forward_linear(data, x1) + b * model.forward_linear(data, x2)

    assert torch.allclose(y12, ylin, rtol=1e-5, atol=1e-6)


def test_select_topk_nodes_unique_and_shape() -> None:
    torch, _ = _torch_and_pyg()

    from ccik.operator_nn import select_topk_nodes

    scores = torch.tensor([0.1, 0.2, -5.0, 3.0, 3.0, 0.0])
    idx = select_topk_nodes(scores, k=4)

    assert idx.shape == (4,)
    as_list = idx.detach().cpu().tolist()
    assert len(as_list) == len(set(as_list))
    assert all(0 <= int(i) < scores.numel() for i in as_list)
