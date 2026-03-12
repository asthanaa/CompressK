# Algorithms

The active codebase implements three dense compressed-Krylov solvers over a CAS Hamiltonian:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

All three use these core objects:

- `q_k`: current Krylov basis vector
- `v_full = H q_k`: exact dense Hamiltonian action
- `E_k = <q_k | H | q_k>`: Rayleigh quotient
- `H_II`: diagonal Hamiltonian element
- `score_I = |(v_full)_I|^2 / |E_k - H_II|`: determinant ranking score

Because the code compresses Krylov vectors explicitly, the Ritz step solves the generalized problem `H c = E S c`.

## `ccik`

Implementation: `src/ccik/core.py`

Workflow:

1. Start from the Hartree-Fock determinant.
2. Compute `v_full = H q_k`.
3. Rank external determinants by the score above.
4. Add the selected determinants and a `Kv`-sized stabilizer from the largest `|v_full|`.
5. Orthogonalize and compress the next Krylov vector.
6. Solve the generalized Ritz problem.

Important parameters:

- `m`: total Krylov dimension
- `nadd`: new determinants added per step
- `nkeep`: support retained in each basis vector
- `Kv`: stabilizer size
- `orth_tol`: breakdown threshold

## `ccik_thick`

Implementation: `src/ccik/thick_restart.py`

Difference from baseline:

- Build a local Krylov cycle of length `m_cycle`
- Solve Ritz
- Keep the best `nroot` Ritz vectors
- Restart from those vectors for up to `ncycles`

Important parameters:

- `m_cycle`: local Krylov dimension per restart cycle
- `ncycles`: number of restart cycles
- `nroot`: number of retained Ritz vectors
- `tol`: convergence threshold between restart energies
- `nadd`, `nkeep`, `Kv`, `orth_tol`: same meaning as in `ccik`

## `ccik_stochastic`

Implementation: `src/ccik/stochastic.py`

Difference from baseline:

- The score, compression, orthogonalization, and Ritz steps are unchanged.
- Only candidate discovery changes: stochastic excitation proposals identify a candidate pool, and the score is evaluated on that pool.

Important parameters:

- `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`: same meaning as in `ccik`
- `n_walkers`: number of stochastic proposals per Krylov step
- `seed`: random seed for reproducibility
- `parent_power`: bias toward large-amplitude parent determinants
- `p_double`: probability of a double excitation proposal
- `mixed_double_weight`: relative weight of mixed-spin doubles
- `eps_denom`: lower bound for `|E_k - H_II|`
