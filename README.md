# ccik

This repo contains a reusable Python package and driver scripts for running a dense (full-CAS matvec) implementation of a CCIK/CCIK-style “compressed Krylov + selected configurations” algorithm.

## Layout

- `src/ccik/`: importable package (algorithm + PySCF helpers)
- `scripts/`: runnable drivers (example scans)
- `configs/`: editable TOML configs (no code edits required)
- `experiments/`: legacy one-off notebooks/scripts (if present)

## Quick start (no install)

From repo root:

```bash
python scripts/n2_cas_scan.py
```

Edit parameters in:

- packaged default: `src/ccik/defaults/config.toml`

To override defaults, pass your own file:

```bash
python scripts/n2_cas_scan.py --config configs/my_config.toml
```

## Install (editable)

```bash
pip install -e .
```

Optional dependencies:

- `pip install -e .[quantum]` (PySCF)
- `pip install -e .[plot]` (matplotlib)
- `pip install -e .[all]`

## Documentation

See `krylov_cipsi.pdf` for the theory, and `src/ccik/core.py` for equation-to-code mapping comments.

Additional algorithms (optional):
- Thick restart (reuse Ritz vectors): `src/ccik/thick_restart.py`
- CIPSI variational-only (no PT2): `src/ccik/cipsi.py`
