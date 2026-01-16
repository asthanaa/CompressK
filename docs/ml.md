# ML components

This repo contains optional ML components intended to improve *support selection* (which determinants to include), not to replace the underlying physics.

Important rule of thumb:

- ML proposes candidates / scores.
- The algorithm still computes energies and coefficients via exact projection/contraction on the selected support.

## AI-selector Krylov integration

The AI-selector workflow is implemented in `src/ccik/ai_selector_krylov.py`.

Key pieces:

- `Selector` protocol: `propose(psi_sparse, iteration, context) -> set[DetId]`
- `GNNSelector`: uses a PyTorch model to score candidate determinants based on engineered features.
- `CIPSISelector`: deterministic baseline selector using the same PT2-like score used by dense CCIK.

The driver `scripts/n2_cas_scan.py` supports an optional model checkpoint:

- CLI: `--gnn-model <path>`
- or config key: `ai_selector_krylov.gnn_model_path`

## FeatureMLP (PyTorch-only)

`src/ccik/feature_mlp.py` provides:

- `FeatureMLPConfig`
- `FeatureMLP`
- `load_feature_mlp_checkpoint(path)`

This model maps `(N, F)` features to `(N,)` scores.

## OperatorNN (PyTorch + PyG)

`src/ccik/operator_nn.py` defines an operator-learning model intended to approximate `y = H x` at the level of per-node magnitudes.

- It is designed to be (approximately) linear in `x` for a fixed graph.
- It produces per-node scores `log(eps + |y_i|)`.

This is intended for research/prototyping and is not required for the default workflows.

## Training scripts

Training/experiment scripts live in `scripts/`:

- `scripts/train_feature_mlp.py`
- `scripts/train_operator_nn.py`
- `scripts/generate_ai_selector_training_data.py`

These scripts may require additional dependencies (PyTorch, torch_geometric).

## Dependency philosophy

The core package depends only on NumPy by default.

- Quantum chemistry workflows: `pip install -e ".[quantum]"` (adds PySCF)
- Tests: `pip install -e ".[test]"`
- ML workflows: install PyTorch (+ PyG for OperatorNN) separately as needed.
