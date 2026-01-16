# Quickstart

This repo is a Python package (`src/ccik/`) plus runnable driver scripts (`scripts/`).

The simplest end-to-end run is the provided N2 CAS example using a TOML config.

## 1) Install

Python >= 3.11 is required.

Install the package in editable mode.

- Minimal install (no PySCF, good for importing utilities only):

```bash
pip install -e .
```

- Recommended for actually running the quantum chemistry workflows (requires PySCF):

```bash
pip install -e ".[quantum]"
```

- Optional (tests):

```bash
pip install -e ".[quantum,test]"
```

## 2) Run the sample N2 input

A small example input lives at:

- `tests/n2_sample.toml`

The main driver is:

- `scripts/n2_cas_scan.py`

Run:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

Outputs:

- Prints per-geometry energy lines.
- Writes a CSV to repo root (currently `out_n2_cas_scan.csv`).

## 3) Choose which method(s) to run

Even though the importable package name is `ccik`, the driver supports multiple backends (CCIK, CCIK thick restart, FCIQMC-Krylov, AI-selector, and CIPSI-var).

The driver supports running one method (`[run].method`) or multiple methods (`[run].methods`).

The canonical method names the driver understands are:

- `ccik` — dense CCIK
- `ccik_thick` — dense CCIK with thick restart
- `cipsi_var` — CIPSI variational-only (no PT2)
- `fciqmckrylov` — dense Krylov build with FCIQMC-inspired stochastic selection
- `ai_selector_krylov` — Krylov build with an AI selector (optional torch model)

Example: run a single method

```toml
[run]
method = "ccik_thick"
```

Example: run multiple methods

```toml
[run]
methods = ["fciqmckrylov", "ccik_thick"]
```

## 4) Common gotchas

- **CAS electron count consistency:** for a closed-shell singlet (`spin=0`), the driver checks
  `mol.nelectron == 2*ncore + nelecas`.
- **Performance:** CAS-FCI scales very fast with active space size. The sample uses STO-3G and a small CAS.

## Next

- Configuration reference: [configuration.md](configuration.md)
- Algorithm details: [algorithms.md](algorithms.md)
- Module map: [architecture.md](architecture.md)
