# CompressK / `ccik`

[![Documentation](https://img.shields.io/badge/Documentation-Open%20Docs-0A7EA4?style=for-the-badge)](https://asthanaa.github.io/CompressK/)

This repository contains a Python package in `src/ccik` and runnable drivers in `scripts` for three compressed-Krylov solvers:

- `ccik`: baseline dense CCIK
- `ccik_thick`: CCIK with thick restart
- `ccik_stochastic`: CCIK with stochastic candidate discovery

## Documentation

- Hosted docs: https://asthanaa.github.io/CompressK/
- In-repo docs index: `docs/index.md`
- Quickstart: `docs/quickstart.md`
- Configuration reference: `docs/configuration.md`
- Algorithms overview: `docs/algorithms.md`

## Algorithm docs

- `ccik`: https://asthanaa.github.io/CompressK/algorithms.html#ccik
- `ccik_thick`: https://asthanaa.github.io/CompressK/algorithms.html#ccik_thick
- `ccik_stochastic`: https://asthanaa.github.io/CompressK/algorithms.html#ccik_stochastic
- Full API reference: https://asthanaa.github.io/CompressK/api.html

## Recommended entry points

- Main N2 scan driver: `python scripts/n2_cas_scan.py --config <toml>`
- N2 PES plot driver: `python scripts/n2_pes_scan_and_plot.py --config <toml>`
- Sample config: `tests/n2_sample.toml`

## Current scope

The repository has been simplified to the three solver variants above. Neural-network support-selection code and its training utilities have been removed from the active codebase.
