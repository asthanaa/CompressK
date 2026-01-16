# CompressK / `ccik`

This repo contains an importable Python package (`src/ccik/`) plus runnable driver scripts (`scripts/`) for running **compressed Krylov + selected-configuration matvecs** algorithms (CCIK family) and related variants (thick restart, FCIQMC-inspired selection, AI-guided selection, CIPSI-var).

## Documentation

- Hosted docs (GitHub Pages): https://asthanaa.github.io/CompressK/
- Full docs index (in-repo): `docs/index.md`
- Quickstart: `docs/quickstart.md`
- Config reference: `docs/configuration.md`

## Recommended entry points (current)

- Package code: `src/ccik/`
	- Algorithm/PDF mapping overview: `src/ccik/README.md`

- Driver scripts: `scripts/`
	- N2 CAS scan (multiple methods): `python scripts/n2_cas_scan.py --config <toml>`
	- Step-by-step run instructions: `scripts/README.md`
	- Sample N2 input file: `tests/n2_sample.toml`

## Legacy scripts

Older, pre-package one-off scripts are kept under `legacy/` for historical reference.

For convenience, compatibility shims remain at the repo root:

- `krylov_cipsi.py`
- `krylov cipsi primitive.py`

New work should use `src/ccik/` and `scripts/`.
