# Algorithms

The active codebase implements three dense compressed-Krylov solvers over a CAS Hamiltonian:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

All three operate in a determinant basis built from the CAS Hamiltonian returned by `ccik.pyscf_cas`.

## Shared building blocks

Each solver repeatedly works with these objects:

- `q_k`: the current Krylov basis vector as a CI coefficient tensor
- `v_full = H q_k`: the exact dense Hamiltonian action
- `E_k = <q_k | H | q_k>`: Rayleigh quotient for the current vector
- `H_II`: diagonal Hamiltonian elements
- `score_I = |(v_full)_I|^2 / |E_k - H_II|`: determinant ranking score
- `S_k`: selected support used to compress the matvec and next basis vector

Because the code compresses vectors explicitly, the Ritz solve uses a generalized eigenproblem `H c = E S c` instead of assuming an exactly orthonormal Krylov basis.

## `ccik`

Implementation: `src/ccik/core.py`

Entry point:

- `ccik_ground_energy_dense(...)`

Workflow:

1. Start from the Hartree-Fock determinant.
2. Compute the exact dense matvec `v_full = H q_k`.
3. Rank external determinants with the score above.
4. Keep the selected determinants plus a `Kv`-sized stabilizer from the largest `|v_full|` entries.
5. Orthogonalize and compress the next Krylov vector.
6. Solve the generalized Ritz problem over the collected Krylov basis.

Important knobs:

- `m`: total Krylov dimension
- `nadd`: new determinants added per step
- `nkeep`: support retained in each basis vector
- `Kv`: raw-amplitude stabilizer size

## `ccik_thick`

Implementation: `src/ccik/thick_restart.py`

Entry point:

- `ccik_ground_energy_dense_thick_restart(...)`

Difference from baseline:

- Instead of building one long Krylov chain, the solver builds a cycle of length `m_cycle`, solves Ritz, keeps the best `nroot` Ritz vectors, and starts the next cycle from those vectors.

Use this variant when:

- a single long Krylov chain is too large or too unstable
- you want a bounded working subspace per cycle

Important knobs:

- `m_cycle`: local Krylov dimension per cycle
- `ncycles`: restart budget
- `nroot`: number of retained Ritz vectors
- `tol`: stopping threshold between successive cycle energies

## `ccik_stochastic`

Implementation: `src/ccik/stochastic.py`

Entry point:

- `ccik_ground_energy_stochastic(...)`

Difference from baseline:

- The score, compression, orthogonalization, and Ritz steps stay the same.
- Only candidate discovery changes: stochastic excitation proposals identify a smaller candidate pool, and the usual score is evaluated only on that pool.

The stochastic discovery stage uses:

- `n_walkers`: number of proposals per Krylov step
- `parent_power`: bias toward large-amplitude parent determinants
- `p_double`: single vs double excitation split
- `mixed_double_weight`: weighting of mixed-spin doubles
- `seed`: reproducibility control

This is still a dense-matvec method. The stochastic part affects candidate discovery, not the Hamiltonian contraction itself.
