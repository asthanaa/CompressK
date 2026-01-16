from __future__ import annotations

import numpy as np

from .core import (
    apply_mask,
    compress_keep_top_mask,
    generalized_eigh,
    inner,
    normalize,
    occ_list_to_bitstring,
    topk_positive_mask,
)
from .params import CCIKThickRestartParams


def gram_schmidt_list(vecs: list[np.ndarray], tol: float = 1e-12) -> list[np.ndarray]:
    """Orthonormalize list of CI tensors; drop near-dependent ones."""
    Q: list[np.ndarray] = []
    for v in vecs:
        r = v.copy()
        for q in Q:
            r -= inner(q, r) * q
        nr = np.linalg.norm(r.ravel())
        if nr > tol:
            Q.append(r / nr)
    return Q


def _ccik_cycle_dense_multi_start(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    start_vecs: list[np.ndarray],
    params: CCIKThickRestartParams,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """One thick-restart cycle: build a compressed Krylov basis from multiple starts."""

    from pyscf.fci import cistring, direct_spin1

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag).reshape(na, nb)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    def H_contract(ci: np.ndarray) -> np.ndarray:
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    Q = gram_schmidt_list([normalize(v) for v in start_vecs], tol=float(getattr(params, "orth_tol", 1e-12)))
    if len(Q) == 0:
        raise RuntimeError("All start vectors vanished after orthogonalization.")

    supports = [compress_keep_top_mask(q, nkeep=params.nkeep) for q in Q]
    Q = [normalize(apply_mask(q, s)) for q, s in zip(Q, supports)]
    supports = [compress_keep_top_mask(q, nkeep=params.nkeep) for q in Q]

    while len(Q) < params.m_cycle:
        qk = Q[-1]
        supp_k = supports[-1]

        v_full = H_contract(qk)
        Ek = inner(qk, v_full)

        denom = np.abs(Ek - hdiag)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = score.copy()
        score_ext[supp_k] = 0.0

        select_mask = topk_positive_mask(score_ext, int(params.nadd or 0))

        topv_mask = (
            compress_keep_top_mask(v_full, nkeep=params.Kv)
            if (params.Kv is not None and params.Kv > 0)
            else None
        )

        supp_v = supp_k | select_mask
        if topv_mask is not None:
            supp_v |= topv_mask

        if params.verbose:
            v2_total = float(np.sum(np.abs(v_full) ** 2))
            v2_kept = float(np.sum(np.abs(v_full[supp_v]) ** 2))
            tail_frac = (max(v2_total - v2_kept, 0.0)) / max(v2_total, 1e-30)
            print(
                f"      |Q|={len(Q):02d} E(q)={Ek:+.10f} tail≈{tail_frac:.2e} supp_v={int(supp_v.sum())}"
            )

        v = apply_mask(v_full, supp_v)

        r = v.copy()
        for qj in Q:
            r -= inner(qj, r) * qj

        nr = np.linalg.norm(r.ravel())
        if nr < float(getattr(params, "orth_tol", 1e-12)):
            break

        q_next = r / nr
        supp_next = compress_keep_top_mask(q_next, nkeep=params.nkeep)
        q_next = normalize(apply_mask(q_next, supp_next))

        Q.append(q_next)
        supports.append(supp_next)

    m_eff = len(Q)
    Hproj = np.zeros((m_eff, m_eff))
    Sproj = np.zeros((m_eff, m_eff))

    Hq = [H_contract(Q[j]) for j in range(m_eff)]
    for i in range(m_eff):
        for j in range(m_eff):
            Sproj[i, j] = inner(Q[i], Q[j])
            Hproj[i, j] = inner(Q[i], Hq[j])

    evals, X = generalized_eigh(Hproj, Sproj)
    return evals, X, Q


def ccik_ground_energy_dense_thick_restart(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKThickRestartParams | None = None,
    stats: dict[str, float] | dict[str, int] | None = None,
) -> float:
    """Dense CCIK with thick restart (reuse best Ritz vectors between cycles).

    This is the “reuseRitz” feature from `experiments/krylov cipsi dense reuseRitz.py`.

    Returns: lowest electronic energy (NOT including any core/nuclear offset).
    """

    if params is None:
        params = CCIKThickRestartParams()

    # Heuristic “checkpointed Krylov” mode (requested behavior):
    # - Treat params.m_cycle as the *total* max Krylov vectors to build.
    # - Treat params.ncycles as the checkpoint interval (how many new Krylov vectors per chunk).
    # - After two chunks are built, compare successive checkpoint energies; if |ΔE| < tol, stop.
    # This is enabled when m_cycle is a clean multiple of ncycles and larger than ncycles.
    checkpoint_every: int | None = None
    total_max: int | None = None
    if (
        int(params.ncycles) > 0
        and int(params.m_cycle) > int(params.ncycles)
        and int(params.m_cycle) % int(params.ncycles) == 0
    ):
        checkpoint_every = int(params.ncycles)
        total_max = int(params.m_cycle)

    from pyscf.fci import cistring

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    occ_a = list(range(neleca))
    occ_b = list(range(nelecb))
    addr_a = cistring.str2addr(norb, neleca, occ_list_to_bitstring(occ_a))
    addr_b = cistring.str2addr(norb, nelecb, occ_list_to_bitstring(occ_b))

    q0 = np.zeros((na, nb))
    q0[addr_a, addr_b] = 1.0
    q0 = normalize(q0)

    if checkpoint_every is not None and total_max is not None:
        # Build one growing Krylov basis up to total_max, checking convergence every checkpoint_every.
        from pyscf.fci import direct_spin1

        hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
        hdiag = np.asarray(hdiag).reshape(na, nb)
        h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

        def H_contract(ci: np.ndarray) -> np.ndarray:
            return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

        Q: list[np.ndarray] = [q0]
        supports: list[np.ndarray] = [compress_keep_top_mask(q0, nkeep=params.nkeep)]
        prev_E: float | None = None
        ncheck_done = 0

        # Initial checkpoint is after the first chunk completes.
        while len(Q) < total_max:
            qk = Q[-1]
            supp_k = supports[-1]

            v_full = H_contract(qk)
            Ek = inner(qk, v_full)

            denom = np.abs(Ek - hdiag)
            denom = np.where(denom < 1e-12, 1e-12, denom)
            score = (np.abs(v_full) ** 2) / denom

            score_ext = score.copy()
            score_ext[supp_k] = 0.0

            select_mask = topk_positive_mask(score_ext, int(params.nadd or 0))

            topv_mask = (
                compress_keep_top_mask(v_full, nkeep=params.Kv)
                if (params.Kv is not None and params.Kv > 0)
                else None
            )

            supp_v = supp_k | select_mask
            if topv_mask is not None:
                supp_v |= topv_mask

            v = apply_mask(v_full, supp_v)
            r = v.copy()
            for qj in Q:
                r -= inner(qj, r) * qj

            nr = np.linalg.norm(r.ravel())
            if nr < float(getattr(params, "orth_tol", 1e-12)):
                break

            q_next = r / nr
            supp_next = compress_keep_top_mask(q_next, nkeep=params.nkeep)
            q_next = normalize(apply_mask(q_next, supp_next))
            Q.append(q_next)
            supports.append(supp_next)

            # Checkpoint after each chunk.
            if (len(Q) % checkpoint_every) == 0:
                m_eff = len(Q)
                Hproj = np.zeros((m_eff, m_eff))
                Sproj = np.zeros((m_eff, m_eff))
                Hq = [H_contract(Q[j]) for j in range(m_eff)]
                for i in range(m_eff):
                    for j in range(m_eff):
                        Sproj[i, j] = inner(Q[i], Q[j])
                        Hproj[i, j] = inner(Q[i], Hq[j])
                evals, _ = generalized_eigh(Hproj, Sproj)
                E0 = float(evals[0])
                ncheck_done += 1

                if params.verbose:
                    if prev_E is None:
                        print(f"    [CCIK checkpoint 1] m={m_eff} E0={E0:+.10f}")
                    else:
                        print(f"    [CCIK checkpoint {ncheck_done}] m={m_eff} E0={E0:+.10f} Δ={E0-prev_E:+.3e}")

                if prev_E is not None and abs(E0 - prev_E) < float(params.tol):
                    if stats is not None:
                        union = supports[0].copy()
                        ndet_sum = int(np.sum(union))
                        for s in supports[1:]:
                            union |= s
                            ndet_sum += int(np.sum(s))
                        stats["cycle"] = int(ncheck_done)
                        stats["m_eff"] = int(m_eff)
                        stats["ndet_union"] = int(np.sum(union))
                        stats["ndet_sum"] = int(ndet_sum)
                        stats["checkpoint_every"] = int(checkpoint_every)
                    return E0

                prev_E = E0

        # If we didn't converge early, return the best Ritz energy from the final basis.
        m_eff = len(Q)
        Hproj = np.zeros((m_eff, m_eff))
        Sproj = np.zeros((m_eff, m_eff))
        Hq = [H_contract(Q[j]) for j in range(m_eff)]
        for i in range(m_eff):
            for j in range(m_eff):
                Sproj[i, j] = inner(Q[i], Q[j])
                Hproj[i, j] = inner(Q[i], Hq[j])
        evals, _ = generalized_eigh(Hproj, Sproj)

        if stats is not None:
            union = supports[0].copy()
            ndet_sum = int(np.sum(union))
            for s in supports[1:]:
                union |= s
                ndet_sum += int(np.sum(s))
            stats["cycle"] = int(ncheck_done)
            stats["m_eff"] = int(m_eff)
            stats["ndet_union"] = int(np.sum(union))
            stats["ndet_sum"] = int(ndet_sum)
            stats["checkpoint_every"] = int(checkpoint_every)

        return float(evals[0])

    starts: list[np.ndarray] = [q0]
    E_prev = None

    for cyc in range(params.ncycles):
        evals, X, Q = _ccik_cycle_dense_multi_start(
            h1,
            eri8,
            norb,
            nelec,
            start_vecs=starts,
            params=params,
        )

        nroot_eff = min(params.nroot, X.shape[1])
        new_starts: list[np.ndarray] = []
        for r in range(nroot_eff):
            psi = np.zeros_like(Q[0])
            coeffs = X[:, r]
            for i, Qi in enumerate(Q):
                psi += coeffs[i] * Qi
            psi = normalize(psi)

            supp = compress_keep_top_mask(psi, nkeep=params.nkeep)
            psi = normalize(apply_mask(psi, supp))
            new_starts.append(psi)

        starts = gram_schmidt_list(new_starts, tol=float(getattr(params, "orth_tol", 1e-12)))
        if len(starts) == 0:
            starts = [new_starts[0]]

        E0 = float(evals[0])

        if stats is not None:
            # Count unique determinants across the stored Krylov basis for this cycle.
            # Q vectors are explicitly masked -> exact zeros are safe to count.
            if len(Q) > 0:
                union = (Q[0] != 0)
                ndet_sum = int(np.sum(union))
                for Qi in Q[1:]:
                    mask = (Qi != 0)
                    union |= mask
                    ndet_sum += int(np.sum(mask))
                stats["cycle"] = int(cyc + 1)
                stats["m_eff"] = int(len(Q))
                stats["ndet_union"] = int(np.sum(union))
                stats["ndet_sum"] = int(ndet_sum)
                stats["nstart"] = int(len(starts))
            else:
                stats["cycle"] = int(cyc + 1)
                stats["m_eff"] = 0
                stats["ndet_union"] = 0
                stats["ndet_sum"] = 0
                stats["nstart"] = int(len(starts))

        if params.verbose:
            if E_prev is None:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f}")
            else:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f} Δ={E0-E_prev:+.3e}")

        if E_prev is not None and abs(E0 - E_prev) < params.tol:
            return E0
        E_prev = E0

    if E_prev is None:
        raise RuntimeError("CCIK thick restart did not produce an energy")

    return float(E_prev)
