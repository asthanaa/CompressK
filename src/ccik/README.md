# `ccik`

This package implements three dense compressed-Krylov solvers over a CAS Hamiltonian:

- baseline CCIK
- CCIK with thick restart
- CCIK-stochastic

## Mapping to the theory note

The short derivation is in `krylov_cipsi.pdf`. The baseline implementation in `core.py` follows the same objects:

- Krylov basis: `q_k`
- Exact Hamiltonian action: `v_full = H q_k`
- Rayleigh quotient: `E_k = <q_k | H | q_k>`
- Determinant score: `|(v_full)_I|^2 / |E_k - H_II|`
- Compressed support: selected determinant set used to project `v_full`
- Ritz solve: generalized eigenproblem over the compressed Krylov basis

## Where each solver lives

- `core.py`: baseline dense CCIK
- `thick_restart.py`: thick restart
- `stochastic.py`: stochastic candidate discovery

## Key parameter meanings

- `m` or `m_cycle`: Krylov basis size budget
- `nadd`: number of score-ranked external determinants added per step
- `nkeep`: support size retained in each compressed basis vector
- `Kv`: stabilizer count for the largest raw `|H q_k|` amplitudes
- `orth_tol`: breakdown threshold during orthogonalization
- `n_walkers`: stochastic proposal count used only in `ccik_stochastic`
