# Algorithms

This repo implements dense (full-CAS) versions of compressed-Krylov / selected-configuration matvec algorithms.

The short theoretical note is in `krylov_cipsi.pdf`. The baseline dense implementation includes equation-level mapping comments in `src/ccik/core.py`.

## Methods implemented (the “4 things”, plus CIPSI)

The main driver `scripts/n2_cas_scan.py` can run these methods via `[run].method` or `[run].methods`:

- `ccik` → TOML: `[ccik]` → module: `ccik.core`
- `ccik_thick` → TOML: `[ccik_thick]` → module: `ccik.thick_restart`
- `fciqmckrylov` → TOML: `[fciqmckrylov]` → module: `ccik.fciqmc_krylov`
- `ai_selector_krylov` → TOML: `[ai_selector_krylov]` → module: `ccik.ai_selector_krylov`

Additionally:

- `cipsi_var` → TOML: `[cipsi]` → module: `ccik.cipsi`

## Common ingredients

All the dense solvers operate in a determinant basis for a CAS Hamiltonian.

- A Krylov vector is a CI coefficient tensor `q_k`.
- The Hamiltonian action is computed by a dense contraction (PySCF FCI contraction) to form `v = H|q_k>`.
- A score is computed to decide which determinants to keep (“support selection”).
- The resulting `v` is projected onto the selected support and orthogonalized to form the next basis vector.
- At the end, a Rayleigh–Ritz solve in the span of Krylov vectors provides an energy estimate.

Because vectors are explicitly compressed/truncated, orthogonality is not exact in finite precision, so the code solves a generalized eigenvalue problem:

- `H c = E S c`

where `S` is the overlap matrix of the compressed Krylov vectors.

## Dense CCIK (`ccik.core`)

Method name in config:

- `ccik`

Entry point:

- `ccik.core.ccik_ground_energy_dense(...)`

High-level behavior:

1. Start from a Hartree–Fock determinant `q0`.
2. For each iteration:
   - compute `v_full = H|q_k>`
   - compute the Rayleigh quotient `E_k = <q_k|H|q_k>`
   - compute a PT2-like importance score per determinant:

     `score(D) = |v_full[D]|^2 / |E_k - H_DD|`

   - select `nadd` determinants not already in the support
   - optionally force-include top-`Kv` determinants by `|v_full|` (stabilizer)
   - project and orthogonalize to get the next Krylov vector
3. Solve the generalized Ritz problem in the Krylov space.

Tuning knobs:

- `m`: number of Krylov vectors
- `nadd`: new determinants per iteration
- `nkeep`: compression size
- `Kv`: stabilizer size

## Thick restart CCIK (`ccik.thick_restart`)

Method name in config:

- `ccik_thick`

Entry point:

- `ccik.thick_restart.ccik_ground_energy_dense_thick_restart(...)`

Idea:

- Build a Krylov subspace for a cycle, solve Ritz, keep the best `nroot` Ritz vectors, and use them as starting vectors for the next cycle.

This is useful when you want to cap the Krylov dimension per cycle but still “grow” information over multiple restarts.

## CIPSI variational-only (`ccik.cipsi`)

Method name in config:

- `cipsi_var`

Entry point:

- `ccik.cipsi.cipsi_dense_variational(...)`

Workflow:

- Maintain a selected determinant set `S`.
- Solve the variational problem in `span(S)` using PySCF’s Davidson solver.
- Expand `S` using the same score as in CCIK and an optional top-|v| stabilizer.

This is variational (no PT2 correction in this implementation).

## FCIQMC-inspired Krylov selection (`ccik.fciqmc_krylov`)

Method name in config:

- `fciqmckrylov`

Entry point:

- `ccik.fciqmc_krylov.ccik_ground_energy_fciqmc_krylov(...)`

This method uses walker spawning to *discover* a candidate pool of connected determinants, then ranks/selects from those candidates.

Important detail:

- The code still uses the dense PySCF Hamiltonian contraction for `H|q>`.
- The stochastic component is candidate discovery + ranking, not a fully stochastic energy estimator.

## AI-selector Krylov (`ccik.ai_selector_krylov`)

Method name in config:

- `ai_selector_krylov`

Entry point:

- `ccik.ai_selector_krylov.ccik_ground_energy_ai_selector_krylov(...)`

This generalizes selection through a `Selector` interface.

Provided selector implementations:

- `CIPSISelector`: deterministic score-based selection (dense full-space or candidate-pool based)
- `GNNSelector`: ranks candidates via a PyTorch model if provided; otherwise falls back to a deterministic heuristic

See `docs/ml.md` for ML-specific details.
