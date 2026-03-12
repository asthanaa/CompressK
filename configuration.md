# Configuration

Most runs use a TOML file consumed by `scripts/n2_cas_scan.py`.

Loading order:

- `--config <path>` if provided
- `configs/config.toml` if present
- packaged defaults in `src/ccik/defaults/config.toml`

## Top-level sections

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`
- `[ccik]`
- `[ccik_thick]`
- `[ccik_stochastic]`

## `[run]`

Use either:

```toml
[run]
method = "ccik"
```

or:

```toml
[run]
methods = ["ccik", "ccik_thick", "ccik_stochastic"]
```

Valid method names:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Accepted aliases:

- `ccik_thick_restart`, `reuse_ritz`, `reuseRitz` -> `ccik_thick`
- `stochastic`, `ccikstochastic`, `fciqmckrylov`, `fciqmc_krylov` -> `ccik_stochastic`

## `[molecule]`

- `basis`: AO basis
- `unit`: geometry unit
- `charge`: total molecular charge
- `spin`: PySCF spin value `2S`
- `verbose`: PySCF verbosity

## `[scan]`

- `R_min`: minimum bond length
- `R_max`: maximum bond length
- `n_points`: number of scan points

## `[cas]`

- `ncas`: number of active orbitals
- `nelecas`: number of active electrons
- `ncore`: number of frozen doubly occupied core orbitals

## `[ccik]`

- `m`: maximum Krylov dimension
- `nadd`: number of external determinants added from the score ranking
- `nkeep`: support retained in each compressed Krylov vector
- `Kv`: stabilizer size from the largest `|H q_k|`
- `orth_tol`: residual norm threshold for breakdown
- `verbose`: print iteration diagnostics

## `[ccik_thick]`

- `m_cycle`: Krylov dimension inside one restart cycle
- `ncycles`: maximum number of restart cycles
- `nroot`: number of lowest Ritz vectors reused as restart seeds
- `tol`: convergence threshold between restart energies
- `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`: same meanings as in `[ccik]`

## `[ccik_stochastic]`

- `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`: same meanings as in `[ccik]`
- `n_walkers`: number of stochastic proposals
- `seed`: random seed, or `null` for nondeterministic runs
- `parent_power`: parent sampling exponent
- `p_double`: probability of proposing doubles
- `mixed_double_weight`: weighting of mixed-spin doubles
- `eps_denom`: lower bound used in `|E_k - H_II|`
