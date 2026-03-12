# Architecture

## Repository layout

- `src/ccik/`: importable package
- `scripts/`: runnable drivers
- `tests/`: smoke tests and sample config
- `configs/`: example user configs
- `legacy/`: preserved historical scripts

## Core modules

### `ccik.pyscf_cas`

Hamiltonian construction helpers:

- `make_mol_pyscf(...)`
- `build_cas_hamiltonian_pyscf(...)`
- `exact_cas_fci_energy_pyscf(...)`

### `ccik.core`

Baseline dense CCIK:

- `ccik_ground_energy_dense(...)`
- `compress_keep_top_mask(...)`
- `topk_positive_mask(...)`
- `generalized_eigh(...)`

### `ccik.thick_restart`

Thick-restart variant:

- `ccik_ground_energy_dense_thick_restart(...)`

### `ccik.stochastic`

Stochastic candidate-discovery variant:

- `ccik_ground_energy_stochastic(...)`

### `ccik.params`

- `CCIKParams`
- `CCIKThickRestartParams`
- `CCIKStochasticParams`

### `ccik.config`

TOML loading and conversion into parameter dataclasses.

## Drivers

### `scripts/n2_cas_scan.py`

Main N2 scan runner:

1. Load config
2. Generate the bond scan
3. Build the CAS Hamiltonian
4. Compute CAS-FCI reference
5. Run one or more CCIK solvers
6. Write `out_n2_cas_scan.csv`

### `scripts/n2_pes_scan_and_plot.py`

PES plotting helper for `ccik_thick` and `ccik_stochastic`.

### `scripts/cr2_scan.py`

Independent Cr2 CAS-FCI scan using the same CAS-building utilities.
