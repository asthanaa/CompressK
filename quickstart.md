# Quickstart

## Install

Python 3.11 or newer is required.

Minimal install:

```bash
pip install -e .
```

Recommended for chemistry workflows:

```bash
pip install -e ".[quantum]"
```

Optional test dependencies:

```bash
pip install -e ".[quantum,test]"
```

## Run the sample N2 input

Use the sample config in `tests/n2_sample.toml`.

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

This writes `out_n2_cas_scan.csv`.

## Valid solver names

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Example:

```toml
[run]
methods = ["ccik", "ccik_stochastic"]
```

## Common gotchas

- For a singlet, the drivers check `mol.nelectron == 2*ncore + nelecas`.
- CAS-FCI is used as a reference in the example scripts and becomes expensive quickly as the active space grows.
- `ccik_stochastic` is reproducible only when `seed` is fixed.
