# CompressK Documentation

This page is a fallback documentation hub for GitHub Pages branch deployments.

If GitHub Pages is serving the repository directly, use the links below.
If GitHub Pages is configured for GitHub Actions, the full MkDocs site is available at the same base URL with sidebar navigation.

## Start here

- [Quickstart](quickstart.html)
- [Algorithms](algorithms.html)
- [Configuration](configuration.html)
- [API](api.html)
- [Architecture](architecture.html)

## Solvers

- [`ccik`](algorithms.html#ccik)
- [`ccik_thick`](algorithms.html#ccik_thick)
- [`ccik_stochastic`](algorithms.html#ccik_stochastic)

## What this repository contains

- `src/ccik`: importable Python package
- `scripts`: runnable drivers and scans
- `configs`: example TOML configs
- `tests`: smoke tests and sample input

## Main driver

Run the N2 example with:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```
