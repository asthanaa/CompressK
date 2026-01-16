# Configuration reference (TOML)

Most runnable workflows use a TOML config consumed by `scripts/n2_cas_scan.py`.

Config loading behavior:

- If you pass `--config <path>`, that file is loaded.
- If you omit `--config`, the driver prefers `configs/config.toml` if it exists.
- Otherwise it falls back to the packaged defaults in `src/ccik/defaults/config.toml`.

Implementation:

- `ccik.config.load_config()` and `ccik.config.load_default_toml()`.

## Top-level sections

A typical config contains:

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`

and then per-method parameter blocks:

- `[ccik]`
- `[ccik_thick]`
- `[cipsi]`
- `[fciqmckrylov]`
- `[ai_selector_krylov]` (optional)

If a section/key is omitted, defaults come from the dataclasses in `src/ccik/params.py`.

## `[run]`

Controls which algorithm(s) to run.

Supported shapes:

- `method = "ccik"`  
  Runs a single method.

- `methods = ["ccik", "ccik_thick"]`  
  Runs multiple methods in a sweep.

The driver additionally canonicalizes some aliases (see `_canonical_method()` in `scripts/n2_cas_scan.py`).

Canonical method names:

- `ccik`
- `ccik_thick`
- `cipsi_var`
- `fciqmckrylov`
- `ai_selector_krylov`

## `[molecule]`

Used when building a PySCF molecule.

Keys:

- `basis` (string)
- `unit` (string; typically `"Angstrom"`)
- `charge` (int)
- `spin` (int; PySCF convention `2S`)
- `verbose` (int)

The geometry itself for N2 is generated inside the driver from the scan bond length.

## `[scan]`

Controls the bond scan for N2.

Keys:

- `R_min` (float; inclusive)
- `R_max` (float; inclusive)
- `n_points` (int)

The driver uses `numpy.linspace(R_min, R_max, n_points)`.

## `[cas]`

Defines the active space.

Keys (required unless the driver uses defaults):

- `ncas` (int)
- `nelecas` (int)
- `ncore` (int; default 0)

The CAS Hamiltonian is constructed via `ccik.pyscf_cas.build_cas_hamiltonian_pyscf`.

## `[ccik]` (dense CCIK)

Overrides `CCIKParams`.

Keys:

- `m` (int): Krylov dimension
- `nadd` (int): number of new determinants to add by score each step
- `nkeep` (int): compression support size for each Krylov vector
- `Kv` (int): stabilizer support size based on top-|v| values
- `orth_tol` (float): breakdown / near-dependence tolerance
- `verbose` (bool)

## `[ccik_thick]` (dense CCIK + thick restart)

Overrides `CCIKThickRestartParams`.

Cycle/restart keys:

- `m_cycle` (int)
- `ncycles` (int)
- `nroot` (int)
- `tol` (float)

Selection/compression keys (same meaning as `[ccik]`):

- `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`

Note: the implementation has a “checkpointed Krylov” mode when `m_cycle` is a clean multiple of `ncycles` (see `src/ccik/thick_restart.py`).

## `[cipsi]` (CIPSI variational-only)

Overrides `CIPSIParams`.

Keys:

- `niter` (int)
- `nadd` (int)
- `ndet_max` (int)
- `Kv` (int)
- `davidson_tol` (float)
- `verbose` (bool)

This method solves a projected variational problem in the selected space using PySCF’s Davidson solver.

## `[fciqmckrylov]` (FCIQMC-inspired selection)

Overrides `FCIQMCKrylovParams`.

Keys:

- Krylov/selection: `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`
- Walkers/spawning: `n_walkers`, `seed`, `parent_power`, `p_double`, `mixed_double_weight`, `eps_denom`

Important implementation detail: this backend still uses the dense Hamiltonian contraction for `H|q>`; the “FCIQMC-inspired” part is how candidates are discovered and ranked.

## `[ai_selector_krylov]` (AI selector)

This section configures the AI-selector Krylov workflow.

It uses the parameter dataclass `AISelectorKrylovParams` defined in `src/ccik/ai_selector_krylov.py`.

Keys:

- Krylov/selection: `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`
- Candidate discovery (walker spawning): `n_walkers`, `seed`, `parent_power`, `p_double`, `mixed_double_weight`
- Scoring denominator floor: `eps_denom`

Model checkpoint path:

- CLI: `python scripts/n2_cas_scan.py --gnn-model <path>`
- or config key: `ai_selector_krylov.gnn_model_path = "..."`

The selector can run with no model (deterministic fallback) or use a PyTorch model.

See [ml.md](ml.md) for details.
