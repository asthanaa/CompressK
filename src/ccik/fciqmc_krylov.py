from __future__ import annotations

import numpy as np

from .core import apply_mask, compress_keep_top_mask, generalized_eigh, inner, normalize, occ_list_to_bitstring
from .params import FCIQMCKrylovParams


def _bit_to_occ(bitstr: int, norb: int) -> list[int]:
    out: list[int] = []
    x = int(bitstr)
    while x:
        lsb = x & -x
        p = lsb.bit_length() - 1
        if p < norb:
            out.append(int(p))
        x ^= lsb
    return out


def _bit_to_vir(bitstr: int, norb: int) -> list[int]:
    full = (1 << norb) - 1
    vir = full ^ int(bitstr)
    return _bit_to_occ(vir, norb)


def _choose_two_distinct(rng: np.random.Generator, items: list[int]) -> tuple[int, int]:
    i = int(rng.integers(0, len(items)))
    j = int(rng.integers(0, len(items) - 1))
    if j >= i:
        j += 1
    a = items[i]
    b = items[j]
    return (a, b) if a < b else (b, a)


def _spawn_once(
    rng: np.random.Generator,
    *,
    occ_a: list[int],
    vir_a: list[int],
    occ_b: list[int],
    vir_b: list[int],
    p_double: float,
    mixed_double_weight: float,
) -> tuple[list[int], list[int], list[int], list[int], float]:
    """Propose a single connected move and return (holes_a, parts_a, holes_b, parts_b, p_gen)."""

    nocc_a = len(occ_a)
    nvir_a = len(vir_a)
    nocc_b = len(occ_b)
    nvir_b = len(vir_b)

    n_single_a = nocc_a * nvir_a
    n_single_b = nocc_b * nvir_b
    n_double_aa = (nocc_a * (nocc_a - 1) // 2) * (nvir_a * (nvir_a - 1) // 2)
    n_double_bb = (nocc_b * (nocc_b - 1) // 2) * (nvir_b * (nvir_b - 1) // 2)
    n_double_ab = nocc_a * nvir_a * nocc_b * nvir_b

    do_double = (rng.random() < p_double)
    if do_double and (n_double_aa + n_double_bb + n_double_ab) == 0:
        do_double = False
    if (not do_double) and (n_single_a + n_single_b) == 0:
        do_double = True

    if not do_double:
        tot = n_single_a + n_single_b
        pa = (n_single_a / tot) if tot > 0 else 0.0
        if rng.random() < pa:
            hole = occ_a[int(rng.integers(0, nocc_a))]
            part = vir_a[int(rng.integers(0, nvir_a))]
            p_gen = (1.0 - p_double) * pa * (1.0 / nocc_a) * (1.0 / nvir_a)
            return [hole], [part], [], [], float(p_gen)

        hole = occ_b[int(rng.integers(0, nocc_b))]
        part = vir_b[int(rng.integers(0, nvir_b))]
        p_gen = (1.0 - p_double) * (1.0 - pa) * (1.0 / nocc_b) * (1.0 / nvir_b)
        return [], [], [hole], [part], float(p_gen)

    # doubles
    w_aa = float(n_double_aa)
    w_bb = float(n_double_bb)
    w_ab = float(n_double_ab) * float(mixed_double_weight)
    wtot = w_aa + w_bb + w_ab
    if wtot <= 0.0:
        return [], [], [], [], 0.0

    r = rng.random() * wtot
    if r < w_aa:
        i, j = _choose_two_distinct(rng, occ_a)
        a, b = _choose_two_distinct(rng, vir_a)
        p_gen = p_double * (w_aa / wtot) * (1.0 / (nocc_a * (nocc_a - 1) / 2.0)) * (1.0 / (nvir_a * (nvir_a - 1) / 2.0))
        return [i, j], [a, b], [], [], float(p_gen)

    if r < (w_aa + w_bb):
        i, j = _choose_two_distinct(rng, occ_b)
        a, b = _choose_two_distinct(rng, vir_b)
        p_gen = p_double * (w_bb / wtot) * (1.0 / (nocc_b * (nocc_b - 1) / 2.0)) * (1.0 / (nvir_b * (nvir_b - 1) / 2.0))
        return [], [], [i, j], [a, b], float(p_gen)

    # mixed
    hole_a = occ_a[int(rng.integers(0, nocc_a))]
    part_a = vir_a[int(rng.integers(0, nvir_a))]
    hole_b = occ_b[int(rng.integers(0, nocc_b))]
    part_b = vir_b[int(rng.integers(0, nvir_b))]
    p_gen = p_double * (w_ab / wtot) * (1.0 / nocc_a) * (1.0 / nvir_a) * (1.0 / nocc_b) * (1.0 / nvir_b)
    return [hole_a], [part_a], [hole_b], [part_b], float(p_gen)


def _fciqmc_discover_candidates_mask(
    *,
    norb: int,
    nelec: tuple[int, int],
    qk: np.ndarray,
    params: FCIQMCKrylovParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a boolean mask of determinants discovered via walker spawning.

    Walkers select parent determinants from the current Krylov vector support and propose
    random connected excitations (single/double). Only the *visited* determinants are marked.

    This replaces the full-basis scan used by deterministic CIPSI scoring.
    """

    from pyscf.fci import cistring

    neleca, nelecb = nelec

    nz = np.argwhere(qk != 0)
    if nz.size == 0 or int(params.n_walkers) <= 0:
        return np.zeros_like(qk, dtype=bool)

    idx_a = nz[:, 0].astype(int)
    idx_b = nz[:, 1].astype(int)
    qvals = np.abs(qk[idx_a, idx_b]).astype(float)
    if float(params.parent_power) != 1.0:
        qvals = qvals ** float(params.parent_power)
    tot = float(np.sum(qvals))
    if tot <= 0.0:
        return np.zeros_like(qk, dtype=bool)
    p_parent = qvals / tot

    cand = np.zeros_like(qk, dtype=bool)

    # Cache parent decoding because walkers repeatedly pick parents from a small support.
    # This avoids calling addr2str and rebuilding occ/vir lists millions of times.
    parent_cache: dict[tuple[int, int], tuple[int, int, list[int], list[int], list[int], list[int]]] = {}
    addr2str = cistring.addr2str
    str2addr = cistring.str2addr

    nwalk = int(params.n_walkers)
    for w in range(nwalk):
        if params.verbose and nwalk >= 20000 and (w % 5000) == 0 and w > 0:
            print(f"      [FCIQMC-KRYLOV] walkers: {w}/{nwalk}")

        t = int(rng.choice(len(p_parent), p=p_parent))
        ia = int(idx_a[t])
        ib = int(idx_b[t])

        key = (ia, ib)
        cached = parent_cache.get(key)
        if cached is None:
            stra_raw = addr2str(norb, neleca, ia)
            strb_raw = addr2str(norb, nelecb, ib)
            if stra_raw is None or strb_raw is None:
                continue
            stra = int(stra_raw)
            strb = int(strb_raw)

            occ_a = _bit_to_occ(stra, norb)
            occ_b = _bit_to_occ(strb, norb)
            vir_a = _bit_to_vir(stra, norb)
            vir_b = _bit_to_vir(strb, norb)
            parent_cache[key] = (stra, strb, occ_a, vir_a, occ_b, vir_b)
        else:
            stra, strb, occ_a, vir_a, occ_b, vir_b = cached

        holes_a, parts_a, holes_b, parts_b, _pgen = _spawn_once(
            rng,
            occ_a=occ_a,
            vir_a=vir_a,
            occ_b=occ_b,
            vir_b=vir_b,
            p_double=float(params.p_double),
            mixed_double_weight=float(params.mixed_double_weight),
        )
        if len(holes_a) + len(holes_b) == 0:
            continue

        # apply excitation (no need for phase for discovery)
        stra2 = stra
        for h in holes_a:
            stra2 ^= (1 << int(h))
        for p in parts_a:
            stra2 ^= (1 << int(p))

        strb2 = strb
        for h in holes_b:
            strb2 ^= (1 << int(h))
        for p in parts_b:
            strb2 ^= (1 << int(p))

        ia2 = int(str2addr(norb, neleca, stra2))
        ib2 = int(str2addr(norb, nelecb, strb2))
        cand[ia2, ib2] = True

    return cand


def ccik_ground_energy_fciqmc_krylov(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: FCIQMCKrylovParams | None = None,
    stats: dict[str, float] | dict[str, int] | None = None,
) -> float:
    """FCIQMC-Krylov: walker-discovered candidate selection for compressed Krylov.

    Walkers *discover* connected determinants stochastically. For the visited set, we then
    compute the usual perturbative/CIPSI score using the exact matvec coefficients v=H|q>.
    """

    if params is None:
        params = FCIQMCKrylovParams()

    from pyscf.fci import cistring, direct_spin1

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag, dtype=float).reshape(na, nb)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    def H_contract(ci: np.ndarray) -> np.ndarray:
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    # HF determinant as start
    occ_a = list(range(neleca))
    occ_b = list(range(nelecb))
    addr_a = int(cistring.str2addr(norb, neleca, occ_list_to_bitstring(occ_a)))
    addr_b = int(cistring.str2addr(norb, nelecb, occ_list_to_bitstring(occ_b)))
    q0 = np.zeros((na, nb))
    q0[addr_a, addr_b] = 1.0
    q0 = normalize(q0)

    rng = np.random.default_rng(None if params.seed is None else int(params.seed))

    Q: list[np.ndarray] = [q0]
    supports: list[np.ndarray] = [compress_keep_top_mask(q0, nkeep=params.nkeep)]

    for k in range(int(params.m) - 1):
        qk = Q[k]
        supp_k = supports[k]

        v_full = H_contract(qk)
        Ek = inner(qk, v_full)

        # Discover candidate determinants via FCIQMC-style spawning from support(qk)
        cand_mask = _fciqmc_discover_candidates_mask(
            norb=norb,
            nelec=nelec,
            qk=apply_mask(qk, supp_k),
            params=params,
            rng=rng,
        )

        # Score ONLY the visited candidates (perturbative/CIPSI score)
        denom = np.abs(Ek - hdiag)
        denom = np.where(denom < float(params.eps_denom), float(params.eps_denom), denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = np.zeros_like(score)
        score_ext[cand_mask] = score[cand_mask]
        score_ext[supp_k] = 0.0

        select_mask = np.zeros_like(score_ext, dtype=bool)
        if int(params.nadd) > 0:
            flat = score_ext.ravel()
            pos = np.where(flat > 0)[0]
            if pos.size > 0:
                nsel = min(int(params.nadd), int(pos.size))
                vals = flat[pos]
                top_local = np.argpartition(vals, -nsel)[-nsel:]
                idx = pos[top_local]
                sflat = np.zeros_like(flat, dtype=bool)
                sflat[idx] = True
                select_mask = sflat.reshape(score_ext.shape)

        topv_mask = (
            compress_keep_top_mask(v_full, nkeep=params.Kv)
            if (params.Kv is not None and int(params.Kv) > 0)
            else None
        )

        supp_v = supp_k | select_mask
        if topv_mask is not None:
            supp_v |= topv_mask

        if params.verbose:
            print(
                f"    k={k:02d} E(q)={Ek:+.10f} visited={int(np.sum(cand_mask))} sel={int(np.sum(select_mask))} supp_v={int(np.sum(supp_v))}"
            )

        v = apply_mask(v_full, supp_v)

        r = v.copy()
        for j in range(k + 1):
            r -= inner(Q[j], r) * Q[j]

        nr = np.linalg.norm(r.ravel())
        if nr < float(getattr(params, "orth_tol", 1e-12)):
            if params.verbose:
                print(f"[FCIQMC-KRYLOV] breakdown at k={k}")
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

    evals, _ = generalized_eigh(Hproj, Sproj)

    if stats is not None:
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
