# Architecture & module map

## Repo layout

- `src/ccik/` — importable Python package (the “library”)
- `scripts/` — runnable drivers (examples / sweeps). These are intended to be copied and modified.
- `tests/` — unit/smoke tests and small sample configs (e.g. `tests/n2_sample.toml`).
- `configs/` — optional user override configs (not required).
- `legacy/` — older scripts preserved for reference (plus compatibility shims at repo root).

## Design goals

- Keep heavy dependencies optional. The package can be imported with only NumPy installed.
- Use PySCF only inside “runtime” paths (local imports) so import remains lightweight.
- Keep “how to build a Hamiltonian” separate from “how to run compressed Krylov / selection”.

## Core flow (dense algorithms)

Most algorithms follow this high-level shape:

1. Build a CAS Hamiltonian using PySCF
2. Obtain an exact CAS-FCI reference energy for comparison (optional but used in scripts)
3. Run an approximate solver that repeatedly applies `H|q>` and uses selection/compression
4. Solve a (generalized) Ritz problem in the Krylov subspace

## Key modules

### `ccik.pyscf_cas`

- `make_mol_pyscf(...)` — builds a PySCF `Mole`
- `build_cas_hamiltonian_pyscf(...)` — produces `(h1eff, eri8, ecore, ncas, nelec)`
- `exact_cas_fci_energy_pyscf(...)` — robust exact CAS-FCI reference

### `ccik.core`

Dense CCIK implementation (the baseline algorithm).

Important helpers are colocated here:

- `generalized_eigh(H, S)` — generalized Rayleigh–Ritz solve
- `compress_keep_top_mask(ci, nkeep)` — support compression by top-|coeff|
- `topk_positive_mask(score, k)` — selection helper used across methods

### `ccik.thick_restart`

Dense CCIK with thick restart:

- `ccik_ground_energy_dense_thick_restart(...)`

### `ccik.cipsi`

CIPSI variational-only solver:

- `cipsi_dense_variational(...)`

### `ccik.fciqmc_krylov`

FCIQMC-inspired candidate discovery and selection, integrated into a dense Krylov build:

- `ccik_ground_energy_fciqmc_krylov(...)`

### `ccik.ai_selector_krylov`

AI-driven selection interface and an AI-selector Krylov workflow:

- `Selector` protocol
- `CIPSISelector` — score-based deterministic selector
- `GNNSelector` — ML-driven selector (optional torch model)
- `ccik_ground_energy_ai_selector_krylov(...)`

### `ccik.operator_nn`

Optional PyTorch/PyG operator-learning model for support selection.

### `ccik.feature_mlp`

Optional PyTorch-only MLP for scoring candidates in the AI-selector legacy path.

### `ccik.config` + `ccik.params`

- `ccik.params` — dataclass defaults for each method
- `ccik.config` — parse TOML dicts into dataclass instances

## Drivers

### `scripts/n2_cas_scan.py`

A complete “runner” that:

- reads config
- builds a bond-length scan
- for each bond length:
  - builds a PySCF molecule
  - builds a CAS Hamiltonian
  - computes exact CAS-FCI reference
  - runs one or more approximate methods
  - writes a CSV

See also `scripts/README.md` and [quickstart.md](quickstart.md).
