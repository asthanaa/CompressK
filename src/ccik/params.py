from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CCIKParams:
    """Parameters controlling the dense CCIK iteration.

    This is the single source of truth for defaults; TOML config overrides are
    applied on top of these defaults (see `ccik.config`).
    """

    m: int = 100
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000
    verbose: bool = False

    # If the orthogonalized residual norm falls below this, treat as Krylov breakdown
    # and stop adding new basis vectors.
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CCIKThickRestartParams:
    """Parameters for dense CCIK with thick restart (reuse best Ritz vectors)."""

    # Subspace construction per cycle
    m_cycle: int = 25
    # Restart cycles
    ncycles: int = 10
    # Number of Ritz vectors to keep as restart starts
    nroot: int = 3
    # Convergence threshold on successive cycle energies
    tol: float = 1e-6

    # Shared selection/compression knobs
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000
    verbose: bool = False

    # Numerical tolerance used for Gram-Schmidt pruning of near-dependent vectors
    # and Krylov breakdown detection inside a cycle.
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CIPSIParams:
    """Parameters for CIPSI variational-only (no PT2) in the full CAS basis."""

    niter: int = 10
    nadd: int = 2000
    ndet_max: int = 10000
    Kv: int = 3000
    davidson_tol: float = 1e-10
    verbose: bool = False


@dataclass(frozen=True)
class FCIQMCKrylovParams:
    """Parameters for an FCIQMC-inspired stochastic selection backend.

    This backend is designed to plug into the existing dense Krylov build:
    it still uses the dense `H|q>` contraction, but it selects new determinants
    via walker sampling rather than deterministic top-n ranking.
    """

    # Krylov dimension
    m: int = 100

    # Selection / compression knobs (aligned with CCIKParams naming)
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000

    # Walker spawning parameters
    n_walkers: int = 20000
    seed: int | None = 0

    # Parent selection distribution: p(parent=J) ∝ |q_J|^parent_power
    parent_power: float = 1.0

    # Excitation proposal mixture
    p_double: float = 0.6
    mixed_double_weight: float = 1.0

    # Score denominator floor for |E - H_II|
    eps_denom: float = 1e-12

    verbose: bool = False
    orth_tol: float = 1e-12
