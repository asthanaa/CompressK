# scripts/

This folder contains runnable drivers built on top of the importable package in `src/ccik`.

## Install prerequisites

- Python >= 3.11
- PySCF

From the repo root:

```bash
pip install -e ".[quantum]"
```

Optional test dependencies:

```bash
pip install -e ".[quantum,test]"
```

## Main example: N2 scan

Use the sample config at `tests/n2_sample.toml`.

Run:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

The script computes CAS-FCI plus the requested solver and writes `out_n2_cas_scan.csv`.

## Supported solver names

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Example:

```toml
[run]
methods = ["ccik_stochastic", "ccik_thick"]
```

## Plot driver

For a PES plot comparing thick restart against stochastic CCIK:

```bash
python scripts/n2_pes_scan_and_plot.py --config configs/config.toml --npoints 20
```
