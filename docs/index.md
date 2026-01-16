# CompressK / `ccik` Documentation

This folder is the “source of truth” documentation set for the repository.

Note: the importable package name is `ccik` for historical reasons, but the repo implements multiple methods:

- `ccik` (dense CCIK)
- `ccik_thick` (CCIK with thick restart)
- `fciqmckrylov` (FCIQMC-inspired Krylov selection)
- `ai_selector_krylov` (AI-guided selection)
- `cipsi_var` (CIPSI variational-only)

## Start here

- Quickstart (run the sample N2 input): [quickstart.md](quickstart.md)
- Configuration reference (TOML schema): [configuration.md](configuration.md)

## Concepts

- Algorithms (CCIK, thick restart, CIPSI-var, FCIQMC-Krylov, AI-selector): [algorithms.md](algorithms.md)
- Repository architecture + module map: [architecture.md](architecture.md)
- Public Python API surface: [api.md](api.md)

## Advanced / contributor docs

- ML components (OperatorNN, FeatureMLP, training scripts): [ml.md](ml.md)
- Development & testing: [development.md](development.md)
- Legacy scripts and compatibility shims: [legacy.md](legacy.md)

## Pointers to existing docs

- `scripts/README.md` contains a step-by-step run walkthrough for the N2 example.
- `src/ccik/README.md` maps the implementation to equations in `krylov_cipsi.pdf`.
- `README.md` and `README_run.md` are short “landing page” docs.
