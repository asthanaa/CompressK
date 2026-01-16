from __future__ import annotations

from dataclasses import asdict
import numpy as np

from .core import (
    apply_mask,
    compress_keep_top_mask,
    inner,
    normalize,
    occ_list_to_bitstring,
    topk_positive_mask,
)
from .params import CIPSIParams


def _as_1d(x):
    """Convert x to a 1D numpy array if possible; else return None."""
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype == object:
        return None
    return arr.ravel()


def _parse_davidson1_output(out, dim: int, nroots: int = 1) -> tuple[float, np.ndarray]:
    """Robustly extract (E0, v0) from PySCF davidson1 outputs across versions."""
    if not isinstance(out, (tuple, list)):
        raise TypeError("Unexpected davidson1 output type")

    energies_candidate = None
    vecs_candidate = None

    # Find vector-like item of size dim
    for item in out:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            v = item[0]
            while isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            v1 = _as_1d(v)
            if v1 is not None and v1.size == dim:
                vecs_candidate = item
                break

        v1 = _as_1d(item)
        if v1 is not None:
            arr = np.asarray(item)
            if arr.ndim == 1 and arr.size == dim:
                vecs_candidate = arr
                break
            if arr.ndim == 2 and arr.shape[0] == nroots and arr.shape[1] == dim:
                vecs_candidate = arr
                break

    # Find energies-like item (small numeric array)
    for item in out:
        e1 = _as_1d(item)
        if e1 is None or e1.size == 0:
            continue
        if e1.size == dim:
            continue
        if e1.size <= max(20, nroots) and np.issubdtype(e1.dtype, np.number):
            energies_candidate = e1
            break

    if vecs_candidate is None:
        biggest = None
        biggest_size = -1
        for item in out:
            a = _as_1d(item)
            if a is not None and a.size > biggest_size:
                biggest = item
                biggest_size = a.size
        if biggest is not None and biggest_size == dim:
            vecs_candidate = biggest

    if vecs_candidate is None or energies_candidate is None:
        raise RuntimeError(
            "Could not parse davidson1 output. "
            f"Found energies={energies_candidate is not None}, vecs={vecs_candidate is not None}. "
            f"Output types={[type(x) for x in out]}"
        )

    E0 = float(energies_candidate[0])

    if isinstance(vecs_candidate, (list, tuple)):
        v = vecs_candidate[0]
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        v0 = _as_1d(v)
    else:
        arr = np.asarray(vecs_candidate)
        v0 = (arr[0].ravel() if arr.ndim == 2 else arr.ravel())

    if v0 is None or v0.size != dim:
        raise RuntimeError(
            f"Parsed davidson vector has wrong size: {None if v0 is None else v0.size}, expected {dim}"
        )

    return E0, np.asarray(v0, dtype=float)


def cipsi_dense_variational(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CIPSIParams | None = None,
    stats: dict[str, float] | dict[str, int] | None = None,
) -> float:
    """CIPSI variational-only (no PT2), in the full CAS determinant basis.

    This follows the workflow in `experiments/krylov cipsi dense reuseRitz.py`:
    - Maintain a selected determinant set S.
    - Solve the variational problem in span(S) using PySCF's davidson1.
    - Expand S by the same score used in CCIK (|v|^2 / |E - H_II|) and a top-|v| stabilizer.

    Returns: lowest electronic energy (NOT including any core/nuclear offset).
    """

    if params is None:
        params = CIPSIParams()

    # Local imports: keep package importable without pyscf.
    from pyscf.fci import cistring, direct_spin1
    from pyscf.lib import linalg_helper

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    dim = na * nb

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag).reshape(na, nb)
    diag_full = hdiag.ravel()

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    def H_contract(ci: np.ndarray) -> np.ndarray:
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    # HF determinant
    occ_a = list(range(neleca))
    occ_b = list(range(nelecb))
    addr_a = cistring.str2addr(norb, neleca, occ_list_to_bitstring(occ_a))
    addr_b = cistring.str2addr(norb, nelecb, occ_list_to_bitstring(occ_b))

    psi = np.zeros((na, nb))
    psi[addr_a, addr_b] = 1.0
    psi = normalize(psi)

    S = np.zeros((na, nb), dtype=bool)
    S[addr_a, addr_b] = True

    def proj_matvec(x: np.ndarray) -> np.ndarray:
        xS = apply_mask(x.reshape(na, nb), S)
        y = H_contract(xS)
        yS = apply_mask(y, S)
        return yS.ravel()

    def aop(xs):
        return [proj_matvec(x) for x in xs]

    def precond(r: np.ndarray, e0: float, x0=None):
        denom = e0 - diag_full
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return r / denom

    E_prev = None
    E_var = None

    for it in range(params.niter):
        x0 = apply_mask(psi, S).ravel()
        x0n = np.linalg.norm(x0)
        if x0n < 1e-16:
            x0 = np.zeros(dim)
            x0[addr_a * nb + addr_b] = 1.0
        else:
            x0 = x0 / x0n

        out = linalg_helper.davidson1(
            aop,
            [x0],
            precond,
            tol=params.davidson_tol,
            max_cycle=60,
            max_space=20,
            nroots=1,
            verbose=0,
        )

        E0, v0 = _parse_davidson1_output(out, dim=dim, nroots=1)
        E_var = float(E0)

        psiS = v0.reshape(na, nb)
        psi = normalize(apply_mask(psiS, S))

        # Select additions from v_full = H|psi>
        v_full = H_contract(psi)
        denom = np.abs(E_var - hdiag)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = score.copy()
        score_ext[S] = 0.0

        add_mask = np.zeros_like(S, dtype=bool)
        add_mask = topk_positive_mask(score_ext, int(params.nadd or 0))

        topv_mask = (
            compress_keep_top_mask(v_full, nkeep=params.Kv)
            if (params.Kv is not None and params.Kv > 0)
            else None
        )

        S_new = S | add_mask
        if topv_mask is not None:
            S_new |= topv_mask

        # Cap S size by ndet_max, prioritizing current coefficients and stabilizer.
        if params.ndet_max is not None and int(S_new.sum()) > int(params.ndet_max):
            tmp = np.zeros_like(psi)
            tmp[S_new] = psi[S_new]
            keep_mask = compress_keep_top_mask(tmp, nkeep=int(params.ndet_max))

            keep_mask |= S
            if topv_mask is not None:
                keep_mask |= topv_mask

            if int(keep_mask.sum()) > int(params.ndet_max):
                tmp2 = np.zeros_like(psi)
                tmp2[keep_mask] = psi[keep_mask]
                keep_mask = compress_keep_top_mask(tmp2, nkeep=int(params.ndet_max))
                if topv_mask is not None:
                    keep_mask |= topv_mask

            S = keep_mask
        else:
            S = S_new

        if params.verbose:
            print(f"    [CIPSI it={it:02d}] E_var={E_var:+.10f} |S|={int(S.sum())}")

        if E_prev is not None and abs(E_var - E_prev) < 1e-10:
            break
        E_prev = E_var

    if E_var is None:
        raise RuntimeError("CIPSI did not produce an energy")

    if stats is not None:
        # Determinants stored in the CIPSI variational wavefunction correspond to the selected set S.
        stats["ndet"] = int(S.sum())

    return float(E_var)
