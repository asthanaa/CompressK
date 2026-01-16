# legacy/

This folder contains older one-off scripts that predate the current `src/ccik/` package layout.

They are kept for historical reference and for comparing against earlier experiments.

## Recommended entry points (current)

- Package algorithms: `src/ccik/`
- Driver scripts: `scripts/`
  - N2 scan: `python scripts/n2_cas_scan.py --config ...`

## Compatibility shims

For convenience, small wrapper scripts remain at the repo root:
- `krylov_cipsi.py`
- `krylov cipsi primitive.py`

These wrappers simply delegate to the implementations in this folder or to the newer drivers.
