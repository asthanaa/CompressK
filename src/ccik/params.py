from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CCIKParams:
    """Parameters controlling the baseline dense CCIK iteration."""

    # Maximum number of Krylov basis vectors to build.
    m: int = 100
    # Number of external determinants added from the score ranking at each step.
    nadd: int = 2000
    # Maximum support size retained for each compressed Krylov vector q_k.
    nkeep: int = 10000
    # Stabilizer support size retained from the largest |(H q_k)_I| entries.
    Kv: int = 3000
    # Emit per-iteration diagnostics.
    verbose: bool = False
    # Treat smaller orthogonalized residual norms as Krylov breakdown.
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CCIKThickRestartParams:
    """Parameters for dense CCIK with thick restart."""

    # Krylov basis size built within one restart cycle.
    m_cycle: int = 25
    # Maximum number of restart cycles.
    ncycles: int = 10
    # Number of lowest Ritz vectors reused as restart seeds.
    nroot: int = 3
    # Stop when successive restart energies differ by less than tol.
    tol: float = 1e-6

    # Same support-selection knobs used by baseline CCIK.
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000
    verbose: bool = False
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CCIKStochasticParams:
    """Parameters for CCIK-stochastic candidate discovery."""

    # Maximum number of Krylov basis vectors.
    m: int = 100

    # Same compression/selection knobs as baseline CCIK.
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000

    # Number of stochastic proposals used to discover candidates for each iteration.
    n_walkers: int = 20000
    # Random seed for reproducible candidate discovery. Use None for nondeterministic runs.
    seed: int | None = 0

    # Parents are sampled with probability proportional to |q_J|**parent_power.
    parent_power: float = 1.0
    # Probability of proposing a double excitation instead of a single excitation.
    p_double: float = 0.6
    # Relative weight of mixed-spin doubles among all double excitations.
    mixed_double_weight: float = 1.0
    # Lower bound for |E_k - H_II| when evaluating the selection score.
    eps_denom: float = 1e-12

    verbose: bool = False
    orth_tol: float = 1e-12
