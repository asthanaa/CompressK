# Configuration reference

Most runs use a TOML file consumed by `scripts/n2_cas_scan.py`.

Loading order:

- `--config <path>` if provided
- `configs/config.toml` if present
- packaged defaults in `src/ccik/defaults/config.toml`

## Top-level sections

A complete config can contain:

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`
- `[ccik]`
- `[ccik_thick]`
- `[ccik_stochastic]`

## `[run]`

Controls which solver or solvers are executed.

Supported shapes:

```toml
[run]
method = "ccik"
```

```toml
[run]
methods = ["ccik", "ccik_thick", "ccik_stochastic"]
```

Valid method names:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Aliases accepted by the driver:

- `ccik_thick_restart`, `reuse_ritz`, `reuseRitz` -> `ccik_thick`
- `stochastic`, `ccikstochastic`, `fciqmckrylov`, `fciqmc_krylov` -> `ccik_stochastic`

## `[molecule]`

Controls the PySCF molecule used to build the CAS Hamiltonian.

- `basis`: AO basis label
- `unit`: geometry unit, usually `Angstrom`
- `charge`: total molecular charge
- `spin`: PySCF spin value `2S`
- `verbose`: PySCF verbosity level

The N2 geometry itself is generated in the driver from the scan bond length.

## `[scan]`

Controls the bond scan grid.

- `R_min`: minimum bond length in Angstrom
- `R_max`: maximum bond length in Angstrom
- `n_points`: number of scan points, including both endpoints

The driver uses `numpy.linspace(R_min, R_max, n_points)`.

## `[cas]`

Defines the active space used to build the CAS Hamiltonian.

- `ncas`: number of active orbitals
- `nelecas`: number of active electrons
- `ncore`: number of frozen doubly occupied core orbitals

For singlets, the example drivers enforce `mol.nelectron == 2*ncore + nelecas`.

## `[ccik]`

Parameters for baseline dense CCIK.

- `m`: maximum Krylov dimension
- `nadd`: number of external determinants added from the score ranking at each step
- `nkeep`: maximum support retained in each compressed Krylov vector
- `Kv`: stabilizer size; always retain the top-`Kv` entries of `|H q_k|`
- `orth_tol`: residual norm threshold used to detect Krylov breakdown
- `verbose`: print per-iteration diagnostics

Interpretation:

- Larger `m` gives a richer Krylov subspace but increases runtime and memory.
- Larger `nadd` explores more external determinants each step.
- Larger `nkeep` stores denser Krylov vectors and usually improves accuracy.
- `Kv` keeps the matvec numerically stable when the score ranking misses large raw amplitudes.

## `[ccik_thick]`

Parameters for CCIK with thick restart.

- `m_cycle`: Krylov dimension built inside one restart cycle
- `ncycles`: maximum number of restart cycles
- `nroot`: number of lowest Ritz vectors reused as restart seeds
- `tol`: convergence threshold on successive cycle energies
- `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`: same meanings as in `[ccik]`

Interpretation:

- `m_cycle` limits the size of each local Krylov solve.
- `nroot` controls how much subspace information is carried into the next cycle.
- `tol` determines when restart energies are considered converged.

## `[ccik_stochastic]`

Parameters for CCIK-stochastic.

- `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`: same meanings as in `[ccik]`
- `n_walkers`: number of stochastic proposals used to discover candidates
- `seed`: random seed for reproducibility; use `null` for nondeterministic runs
- `parent_power`: parent sampling exponent, where parents are weighted by `|q_J|**parent_power`
- `p_double`: probability of proposing a double excitation instead of a single excitation
- `mixed_double_weight`: relative weight of mixed-spin doubles among all doubles
- `eps_denom`: lower bound used in the score denominator `|E_k - H_II|`

Interpretation:

- `n_walkers` is the main accuracy/runtime knob for stochastic discovery.
- `parent_power > 1` biases proposals more strongly toward large-amplitude parents.
- `p_double` and `mixed_double_weight` shape how the proposal budget is split across excitation types.
- `eps_denom` prevents singular or unstable score denominators near diagonal crossings.
