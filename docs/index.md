# CompressK / `ccik` Documentation

This `docs/` folder is the “source of truth” documentation set for the repository.

Note: the importable package name is `ccik` for historical reasons, but the repo implements multiple methods:

- `ccik` (dense CCIK)
- `ccik_thick` (CCIK with thick restart)
- `fciqmckrylov` (FCIQMC-inspired Krylov selection)
- `ai_selector_krylov` (AI-guided selection)
- `cipsi_var` (CIPSI variational-only)

## Start here

- **Quickstart (run the sample N2 input):** `docs/quickstart.md`
- **Configuration reference (TOML schema):** `docs/configuration.md`

## Concepts

- **Algorithms (CCIK, thick restart, CIPSI-var, FCIQMC-Krylov, AI-selector):** `docs/algorithms.md`
- **Repository architecture + module map:** `docs/architecture.md`
- **Public Python API surface:** `docs/api.md`

## Advanced / contributor docs

- **ML components (OperatorNN, FeatureMLP, training scripts):** `docs/ml.md`
- **Development & testing:** `docs/development.md`
- **Legacy scripts and compatibility shims:** `docs/legacy.md`

## Pointers to existing docs

- `scripts/README.md` contains a step-by-step run walkthrough for the N2 example.
- `src/ccik/README.md` maps the implementation to equations in `krylov_cipsi.pdf`.
- `README.md` and `README_run.md` are short “landing page” docs.
