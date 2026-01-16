# Public API

This project uses a `src/` layout; the importable package name is `ccik`.

The package exports a convenient surface from `src/ccik/__init__.py`. For stable imports in external scripts, prefer importing from `ccik` (top-level) rather than internal modules.

## Core algorithms

- `ccik_ground_energy_dense(...)` — dense CCIK baseline
- `ccik_ground_energy_dense_thick_restart(...)` — thick restart variant
- `cipsi_dense_variational(...)` — CIPSI variational-only
- `ccik_ground_energy_fciqmc_krylov(...)` — FCIQMC-inspired selection
- `ccik_ground_energy_ai_selector_krylov(...)` — AI-selector Krylov

## CAS/PySCF helpers

- `make_mol_pyscf(...)`
- `build_cas_hamiltonian_pyscf(...)`
- `exact_cas_fci_energy_pyscf(...)`

## Parameter dataclasses

- `CCIKParams`
- `CCIKThickRestartParams`
- `CIPSIParams`
- `FCIQMCKrylovParams`
- `AISelectorKrylovParams`

## Config helpers

- `load_toml(path)`
- `load_default_toml()`
- `load_config(path_or_none)`

- `run_method_from_dict(run_dict)`
- `run_methods_from_dict(run_dict)`

- `ccik_params_from_dict(d)`
- `ccik_thick_restart_params_from_dict(d)`
- `cipsi_params_from_dict(d)`
- `fciqmc_krylov_params_from_dict(d)`
- `ai_selector_krylov_params_from_dict(d)`

- `cas_spec_from_dict(d)`

## AI/ML support-selection utilities

- `Selector` (protocol)
- `CIPSISelector`, `GNNSelector`

- `OperatorNN`, `OperatorNNConfig`
- `FeatureMLP`, `FeatureMLPConfig`, `load_feature_mlp_checkpoint(...)`

Note: ML dependencies are optional and imported lazily.
