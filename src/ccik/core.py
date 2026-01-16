from __future__ import annotations

import numpy as np

from .params import CCIKParams


# PDF mapping note
# ---------------
# This module implements the “Compressed Krylov Subspace Method with Selected-Configuration Matvecs”
# described in `krylov_cipsi.pdf`, Section II (A–F).
#
# Key correspondences used below:
# - Krylov space: PDF Sec. II.A, Eq. (1)
# - Exact matvec coefficients v_I^(k): PDF Sec. II.B, Eq. (4)
# - Rayleigh quotient E_k: PDF Sec. II.C, Eq. (5)
# - Selection score eta_I^(k): PDF Sec. II.C, Eq. (6)
# - Compressed matvec |v_tilde_k> = P_{S_k} H|q_k>: PDF Sec. II.C, Eq. (7)–(8)
# - Orthogonalization / next basis vector: PDF Sec. II.E, Eq. (11)–(12)
# - Ritz in Krylov subspace: PDF Sec. II.F, Eq. (14)–(17)
#
# Important implementation detail:
# The PDF assumes exact orthonormality so S=I (PDF Eq. (13)). Here we also compress vectors, so we
# solve a generalized Ritz problem using the computed overlap Sproj.


def normalize(ci: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(ci.ravel())
    return ci if nrm == 0 else (ci / nrm)


def inner(ci1: np.ndarray, ci2: np.ndarray) -> float:
    return float(np.vdot(ci1.ravel(), ci2.ravel()))


def compress_keep_top_mask(ci: np.ndarray, nkeep: int | None = None, min_abs: float = 0.0) -> np.ndarray:
    """Boolean mask keeping up to `nkeep` largest |coeff| entries (and >= `min_abs`)."""
    flat = ci.ravel()
    absflat = np.abs(flat)

    if nkeep is None or nkeep >= flat.size:
        return (absflat >= min_abs).reshape(ci.shape)

    # Use strict '>' so that with the default min_abs=0.0 we don't treat exact zeros
    # as eligible and end up selecting arbitrary zero entries.
    eligible = np.where(absflat > min_abs)[0]
    if eligible.size == 0:
        idx = int(np.argmax(absflat))
        mask = np.zeros_like(flat, dtype=bool)
        mask[idx] = True
        return mask.reshape(ci.shape)

    if eligible.size > nkeep:
        sub = absflat[eligible]
        top_idx = np.argpartition(sub, -nkeep)[-nkeep:]
        keep = eligible[top_idx]
    else:
        keep = eligible

    mask = np.zeros_like(flat, dtype=bool)
    mask[keep] = True
    return mask.reshape(ci.shape)


def apply_mask(ci: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ci)
    out[mask] = ci[mask]
    return out


def occ_list_to_bitstring(occ: list[int]) -> int:
    s = 0
    for p in occ:
        s |= (1 << p)
    return s


def generalized_eigh(H: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Solve H x = e S x for symmetric H and SPD S via Cholesky."""
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(S + eps * np.eye(S.shape[0]))

    Linv = np.linalg.inv(L)
    A = Linv @ H @ Linv.T
    evals, Y = np.linalg.eigh(A)
    X = Linv.T @ Y
    return evals, X


def ccik_ground_energy_dense(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKParams | None = None,
    stats: dict[str, float] | dict[str, int] | None = None,
) -> float:
    """Dense CIPSI-scored inexact Krylov in a determinant basis.

    PDF mapping:
    - Build H|q_k> exactly via a dense matvec (PDF Sec. II.B, Eq. (4)).
    - Score determinants using eta_I^(k) = |v_I^(k)|^2 / |E_k - H_II| (PDF Sec. II.C, Eq. (6)).
    - Define the compressed image |v_tilde_k> by projecting onto selected set S_k (PDF Sec. II.C, Eq. (7)–(8)).
    - Build the next Krylov vector by orthogonalization + normalization (PDF Sec. II.E, Eq. (11)–(12)).
    - Finish with a (generalized) Ritz solve in the span of the Krylov basis (PDF Sec. II.F, Eq. (14)–(17)).

    This implementation expects PySCF-style integrals (h1, eri8) and uses
    `pyscf.fci.direct_spin1.contract_2e` for the matvec.

    Returns: lowest electronic energy (NOT including any core/nuclear offset).
    """

    if params is None:
        params = CCIKParams()

    # Local import so the package can be imported even if pyscf isn't available
    # (only required when actually running the algorithm).
    from pyscf.fci import cistring, direct_spin1

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    # H_II diagonal elements used in the selection denominator |E_k - H_II| (PDF Eq. (6))
    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag).reshape(na, nb)

    # Build an effective 2e object so that contract_2e computes H|q_k> (PDF Eq. (4)).
    # Note: PySCF's type hints may declare fac as int; runtime expects float values like 0.5.
    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    def H_contract(ci: np.ndarray) -> np.ndarray:
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    # HF determinant in this CI tensor
    occ_a = list(range(neleca))
    occ_b = list(range(nelecb))
    addr_a = cistring.str2addr(norb, neleca, occ_list_to_bitstring(occ_a))
    addr_b = cistring.str2addr(norb, nelecb, occ_list_to_bitstring(occ_b))

    q0 = np.zeros((na, nb))
    q0[addr_a, addr_b] = 1.0
    q0 = normalize(q0)

    Q: list[np.ndarray] = [q0]
    supports: list[np.ndarray] = [compress_keep_top_mask(q0, nkeep=params.nkeep)]

    for k in range(params.m - 1):
        qk = Q[k]
        supp_k = supports[k]

        # Exact Hamiltonian action: v_full is the coefficient tensor v_I^(k) for H|q_k> (PDF Eq. (4)).
        v_full = H_contract(qk)

        # Rayleigh quotient E_k = <q_k|H|q_k> (PDF Eq. (5))
        Ek = inner(qk, v_full)

        # Selection score eta_I^(k) (PDF Eq. (6))
        denom = np.abs(Ek - hdiag)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = score.copy()
        score_ext[supp_k] = 0.0

        select_mask = np.zeros_like(score_ext, dtype=bool)
        if params.nadd is not None and params.nadd > 0:
            flat = score_ext.ravel()
            pos = np.where(flat > 0)[0]
            if pos.size > 0:
                nsel = min(params.nadd, pos.size)
                vals = flat[pos]
                top_local = np.argpartition(vals, -nsel)[-nsel:]
                idx = pos[top_local]
                sflat = np.zeros_like(flat, dtype=bool)
                sflat[idx] = True
                select_mask = sflat.reshape(score_ext.shape)

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
                f"    k={k:02d} E(q)={Ek:+.10f} tail≈{tail_frac:.2e} supp_v={int(supp_v.sum())}"
            )

        # Compressed matvec: |v_tilde_k> = P_{S_k} H|q_k> (PDF Eq. (7)–(8))
        v = apply_mask(v_full, supp_v)

        # Orthogonalization: r_k = |v_tilde_k> - sum_j |q_j><q_j|v_tilde_k> (PDF Eq. (11))
        r = v.copy()
        for j in range(k + 1):
            r -= inner(Q[j], r) * Q[j]

        nr = np.linalg.norm(r.ravel())
        if nr < float(getattr(params, "orth_tol", 1e-12)):
            if params.verbose:
                print(f"[CCIK] breakdown at k={k}")
            break

        # Normalize to get next Krylov basis vector q_{k+1} (PDF Eq. (12))
        q_next = r / nr

        # compress next basis vector
        supp_next = compress_keep_top_mask(q_next, nkeep=params.nkeep)
        q_next = normalize(apply_mask(q_next, supp_next))

        Q.append(q_next)
        supports.append(supp_next)

    # Ritz step in the Krylov subspace (PDF Sec. II.F, Eq. (14)–(17)).
    # Because we compress vectors, we solve a generalized Ritz problem using the overlap Sproj.
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
        # Report how many *unique* determinants appear across all stored Krylov vectors.
        # Because we explicitly mask vectors, exact zeros are stable and can be used for counting.
        if len(supports) > 0:
            union = supports[0].copy()
            for s in supports[1:]:
                union |= s
            stats["m_eff"] = int(m_eff)
            stats["ndet_union"] = int(np.sum(union))
            stats["ndet_sum"] = int(sum(int(np.sum(s)) for s in supports))
        else:
            stats["m_eff"] = int(m_eff)
            stats["ndet_union"] = 0
            stats["ndet_sum"] = 0

    return float(evals[0])
