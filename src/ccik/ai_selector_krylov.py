from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .core import (
    apply_mask,
    compress_keep_top_mask,
    generalized_eigh,
    inner,
    normalize,
    occ_list_to_bitstring,
    topk_positive_flat_indices,
)

# Determinant identifier in this codebase: (alpha_addr, beta_addr)
DetId = tuple[int, int]


class Selector(Protocol):
    """Selector interface for proposing determinants for the compressed matvec.

    propose(psi_sparse, iteration, context) -> set[DetId]

    - psi_sparse: dict mapping det_id -> coefficient for current Krylov vector.
    - iteration: Krylov iteration index.
    - context: dict of light-weight metadata (e.g., hdiag, E_k, candidate pool).

    NOTE: Selectors must NOT compute Hamiltonian matrix elements or energies beyond
    what is already supplied in `context` by the main Krylov driver.
    """

    def propose(self, psi_sparse: dict[DetId, float], iteration: int, context: dict[str, Any]) -> set[DetId]:
        ...


class CIPSISelector:
    """Existing PT2-like / CIPSI-style importance selection.

    Uses the same score as the dense CCIK implementation:
      score(D) = |(H psi)[D]|^2 / |E_k - H_DD|

    This selector only *ranks* candidates; it relies on the driver to compute:
    - v_full = H|psi>
    - E_k
    - hdiag

    It can rank either:
    - the full space (if context['candidate_ids'] is None), or
    - only a provided candidate pool.
    """

    def __init__(self, *, nadd: int | None = None, eps_denom: float = 1e-12):
        self.nadd = nadd
        self.eps_denom = float(eps_denom)

    def propose(self, psi_sparse: dict[DetId, float], iteration: int, context: dict[str, Any]) -> set[DetId]:
        v_full: np.ndarray = context["v_full"]
        hdiag: np.ndarray = context["hdiag"]
        E_k: float = float(context["E_k"])
        in_support: np.ndarray = context["in_support_mask"]

        nadd = int(self.nadd if self.nadd is not None else int(context.get("nadd", 0)))
        if nadd <= 0:
            return set()

        denom = np.abs(E_k - hdiag)
        denom = np.where(denom < self.eps_denom, self.eps_denom, denom)
        score = (np.abs(v_full) ** 2) / denom

        cand_ids: list[DetId] | None = context.get("candidate_ids")
        if cand_ids is None:
            # Full-space ranking (only for small systems).
            score_ext = score.copy()
            score_ext[in_support] = 0.0
            idx = topk_positive_flat_indices(score_ext, nadd)
            if idx.size == 0:
                return set()
            ia, ib = np.unravel_index(idx, score_ext.shape)
            return {(int(a), int(b)) for a, b in zip(ia, ib)}

        # Candidate-pool ranking
        scored: list[tuple[float, DetId]] = []
        for det in cand_ids:
            ia, ib = det
            if in_support[ia, ib]:
                continue
            s = float(score[ia, ib])
            if s > 0.0:
                scored.append((s, det))

        if not scored:
            return set()

        scored.sort(key=lambda x: x[0], reverse=True)
        return {det for _, det in scored[:nadd]}


class GNNSelector:
    """AI-guided determinant proposer.

    The selector *only proposes which determinants to include*.
    It does NOT compute coefficients, energies, or Hamiltonian matrix elements.

    If a PyTorch model is provided, it is used to rank candidates based on
    engineered features. If not, a deterministic heuristic ranks candidates.

    Expected model behavior (if provided):
    - Legacy: Input float tensor of shape (N, F) -> output (N,) scores.
    - Operator-learning: Input (graph, x) -> output (N,) scores.

    No ML dependency is required: if torch is not importable, it falls back.
    """

    def __init__(
        self,
        *,
        nadd: int | None = None,
        eps_denom: float = 1e-12,
        model: Any | None = None,
    ):
        self.nadd = nadd
        self.eps_denom = float(eps_denom)
        self.model = model

    def _features(self, det: DetId, context: dict[str, Any]) -> np.ndarray:
        ia, ib = det
        hdiag: np.ndarray = context["hdiag"]
        E_k: float = float(context["E_k"])
        parent_w: dict[DetId, float] = context.get("candidate_parent_weight", {})

        w = float(parent_w.get(det, 0.0))
        denom = float(abs(E_k - float(hdiag[ia, ib])))
        return np.array(
            [
                w,
                np.log(w + 1e-30),
                denom,
                1.0 / (denom + self.eps_denom),
                float(context.get("iteration", 0)),
            ],
            dtype=float,
        )

    def propose(self, psi_sparse: dict[DetId, float], iteration: int, context: dict[str, Any]) -> set[DetId]:
        nadd = int(self.nadd if self.nadd is not None else int(context.get("nadd", 0)))
        if nadd <= 0:
            return set()

        in_support: np.ndarray = context["in_support_mask"]
        cand_ids: list[DetId] | None = context.get("candidate_ids")
        if not cand_ids:
            return set()

        # Stable order for deterministic behavior.
        cand_ids = sorted(cand_ids)

        # Build features (legacy path)
        context = dict(context)
        context["iteration"] = int(iteration)
        X = np.vstack([self._features(det, context) for det in cand_ids])

        scores: np.ndarray | None = None

        # Optional torch model
        if self.model is not None:
            try:
                import torch  # type: ignore[import-not-found]

                self.model.eval()
                with torch.no_grad():
                    # If the caller provided an operator-learning model, they should pass
                    # graph + x via context. We keep this optional to avoid forcing PyG.
                    graph = context.get("graph")
                    x_coeff = context.get("x_coeff")
                    if graph is not None and x_coeff is not None:
                        yt = self.model(graph, x_coeff)
                    else:
                        xt = torch.tensor(X, dtype=torch.float32)
                        yt = self.model(xt)
                    y = yt.detach().cpu().numpy()
                scores = y.reshape(-1)
            except Exception:
                scores = None

        if scores is None:
            # Deterministic fallback heuristic (no ML dependency):
            # prefer candidates generated from large parent amplitudes and with small |E_k - H_ii|.
            scores = X[:, 0] * X[:, 3]

        scored: list[tuple[float, DetId]] = []
        for s, det in zip(scores, cand_ids):
            ia, ib = det
            if in_support[ia, ib]:
                continue
            scored.append((float(s), det))

        if not scored:
            return set()

        scored.sort(key=lambda x: x[0], reverse=True)
        return {det for _, det in scored[:nadd]}


@dataclass(frozen=True)
class AISelectorKrylovParams:
    """Parameters for the AI-selector Krylov driver.

    The underlying Krylov algebra (orthogonalization, compression, Ritz) is kept
    identical to the dense CCIK implementation.

    Candidate discovery uses FCIQMC-style spawning to build a small pool to rank.
    """

    # Krylov dimension
    m: int = 100

    # Selection/compression knobs
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000

    # Candidate discovery (walker spawning)
    n_walkers: int = 20000
    seed: int | None = 0
    parent_power: float = 1.0
    p_double: float = 0.6
    mixed_double_weight: float = 1.0

    # Score denominator floor for CIPSI scoring
    eps_denom: float = 1e-12

    orth_tol: float = 1e-12
    verbose: bool = False


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
) -> tuple[list[int], list[int], list[int], list[int]]:
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
            return [hole], [part], [], []

        hole = occ_b[int(rng.integers(0, nocc_b))]
        part = vir_b[int(rng.integers(0, nvir_b))]
        return [], [], [hole], [part]

    w_aa = float(n_double_aa)
    w_bb = float(n_double_bb)
    w_ab = float(n_double_ab) * float(mixed_double_weight)
    wtot = w_aa + w_bb + w_ab
    if wtot <= 0.0:
        return [], [], [], []

    r = rng.random() * wtot
    if r < w_aa:
        i, j = _choose_two_distinct(rng, occ_a)
        a, b = _choose_two_distinct(rng, vir_a)
        return [i, j], [a, b], [], []

    if r < (w_aa + w_bb):
        i, j = _choose_two_distinct(rng, occ_b)
        a, b = _choose_two_distinct(rng, vir_b)
        return [], [], [i, j], [a, b]

    hole_a = occ_a[int(rng.integers(0, nocc_a))]
    part_a = vir_a[int(rng.integers(0, nvir_a))]
    hole_b = occ_b[int(rng.integers(0, nocc_b))]
    part_b = vir_b[int(rng.integers(0, nvir_b))]
    return [hole_a], [part_a], [hole_b], [part_b]


def _discover_candidate_pool(
    *,
    norb: int,
    nelec: tuple[int, int],
    qk: np.ndarray,
    params: AISelectorKrylovParams,
    rng: np.random.Generator,
) -> tuple[list[DetId], dict[DetId, float]]:
    """Discover candidate determinants via walker spawning.

    Returns:
      - candidate_ids: list of determinant ids (alpha_addr, beta_addr)
      - candidate_parent_weight: det_id -> parent selection weight proxy

    This is purely a *proposal* mechanism and does not evaluate H elements.
    """

    from pyscf.fci import cistring

    neleca, nelecb = nelec

    nz = np.argwhere(qk != 0)
    if nz.size == 0 or int(params.n_walkers) <= 0:
        return [], {}

    idx_a = nz[:, 0].astype(int)
    idx_b = nz[:, 1].astype(int)
    qabs = np.abs(qk[idx_a, idx_b]).astype(float)

    if float(params.parent_power) != 1.0:
        qabs = qabs ** float(params.parent_power)

    tot = float(np.sum(qabs))
    if tot <= 0.0:
        return [], {}

    p_parent = qabs / tot

    # Cache decoded parents
    parent_cache: dict[DetId, tuple[int, int, list[int], list[int], list[int], list[int]]] = {}
    addr2str = cistring.addr2str
    str2addr = cistring.str2addr

    cand_set: set[DetId] = set()
    cand_parent_w: dict[DetId, float] = {}

    nwalk = int(params.n_walkers)
    for w in range(nwalk):
        if params.verbose and nwalk >= 20000 and (w % 5000) == 0 and w > 0:
            print(f"      [AI-SELECTOR] walkers: {w}/{nwalk}")

        t = int(rng.choice(len(p_parent), p=p_parent))
        ia = int(idx_a[t])
        ib = int(idx_b[t])

        parent_id: DetId = (ia, ib)
        parent_w = float(qabs[t])

        cached = parent_cache.get(parent_id)
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
            parent_cache[parent_id] = (stra, strb, occ_a, vir_a, occ_b, vir_b)
        else:
            stra, strb, occ_a, vir_a, occ_b, vir_b = cached

        holes_a, parts_a, holes_b, parts_b = _spawn_once(
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
        det: DetId = (ia2, ib2)
        cand_set.add(det)

        prev = cand_parent_w.get(det)
        if prev is None or parent_w > prev:
            cand_parent_w[det] = parent_w

    return sorted(cand_set), cand_parent_w


def ccik_ground_energy_ai_selector_krylov(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    selector_backend: str = "cipsi",
    gnn_model: Any | None = None,
    selector: Selector | None = None,
    iteration_collector: Any | None = None,
    params: AISelectorKrylovParams | None = None,
    stats: dict[str, float] | dict[str, int] | None = None,
) -> float:
    """AI-selector Krylov driver (separate optional subroutine).

    This function keeps the Krylov algebra identical to `ccik_ground_energy_dense`.
    The only change is how we choose the determinant set for the compressed matvec.

    selector_backend:
      - "cipsi": uses CIPSISelector scoring (PT2-like heuristic)
      - "gnn": uses GNNSelector (model if provided; else deterministic fallback)

    Important: The selector must NOT compute Hamiltonian elements. The driver still computes
    v_full = H|q_k> using the existing exact routine (PySCF contract_2e) and then masks
    to the proposed determinant set.

    iteration_collector:
        Optional callback hook for data collection / debugging.
        If provided, it will be called once per Krylov iteration as:
            iteration_collector(k=k, qk=qk, v_full=v_full, E_k=E_k, hdiag=hdiag,
                                supp_k=supp_k, cand_ids=cand_ids, cand_parent_w=cand_parent_w)
        This is kept intentionally untyped to avoid importing ML/data deps.
    """

    if params is None:
        params = AISelectorKrylovParams()

    if selector is None:
        backend = selector_backend.strip().lower()
        if backend == "cipsi":
            selector = CIPSISelector(nadd=params.nadd, eps_denom=params.eps_denom)
        elif backend == "gnn":
            selector = GNNSelector(nadd=params.nadd, eps_denom=params.eps_denom, model=gnn_model)
        else:
            raise ValueError(f"Unknown selector_backend={selector_backend!r}. Expected 'cipsi' or 'gnn'.")

    from pyscf.fci import cistring, direct_spin1

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag, dtype=float).reshape(na, nb)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    def H_contract(ci: np.ndarray) -> np.ndarray:
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    # HF start vector
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

        # Exact Hamiltonian action
        v_full = H_contract(qk)
        E_k = inner(qk, v_full)

        # Provide sparse psi for selector (dict of nonzero entries)
        nz = np.argwhere(qk != 0)
        psi_sparse: dict[DetId, float] = {(int(i), int(j)): float(qk[int(i), int(j)]) for i, j in nz}

        # Candidate pool (discovery): selector ranks this pool.
        cand_ids, cand_parent_w = _discover_candidate_pool(
            norb=norb,
            nelec=nelec,
            qk=apply_mask(qk, supp_k),
            params=params,
            rng=rng,
        )

        context: dict[str, Any] = {
            "nadd": int(params.nadd),
            "E_k": float(E_k),
            "hdiag": hdiag,
            "in_support_mask": supp_k,
            "candidate_ids": cand_ids,
            "candidate_parent_weight": cand_parent_w,
            # Only CIPSISelector uses v_full; GNNSelector ignores it.
            "v_full": v_full,
        }

        if iteration_collector is not None:
            try:
                iteration_collector(
                    k=int(k),
                    qk=qk,
                    v_full=v_full,
                    E_k=float(E_k),
                    hdiag=hdiag,
                    supp_k=supp_k,
                    cand_ids=cand_ids,
                    cand_parent_w=cand_parent_w,
                )
            except Exception:
                # Collector hooks should not affect the main algorithm.
                pass

        proposed = selector.propose(psi_sparse, k, context)

        select_mask = np.zeros((na, nb), dtype=bool)
        for ia, ib in proposed:
            if 0 <= ia < na and 0 <= ib < nb and (not supp_k[ia, ib]):
                select_mask[ia, ib] = True

        topv_mask = compress_keep_top_mask(v_full, nkeep=params.Kv) if int(params.Kv) > 0 else None

        supp_v = supp_k | select_mask
        if topv_mask is not None:
            supp_v |= topv_mask

        if params.verbose:
            print(
                f"    k={k:02d} E(q)={E_k:+.10f} cand={len(cand_ids)} sel={int(select_mask.sum())} supp_v={int(supp_v.sum())}"
            )

        v = apply_mask(v_full, supp_v)

        # Orthogonalization (identical to core.ccik_ground_energy_dense)
        r = v.copy()
        for j in range(k + 1):
            r -= inner(Q[j], r) * Q[j]

        nr = np.linalg.norm(r.ravel())
        if nr < float(params.orth_tol):
            if params.verbose:
                print(f"[AI-SELECTOR] breakdown at k={k}")
            break

        q_next = r / nr
        supp_next = compress_keep_top_mask(q_next, nkeep=params.nkeep)
        q_next = normalize(apply_mask(q_next, supp_next))

        Q.append(q_next)
        supports.append(supp_next)

    # Ritz solve (identical to core.ccik_ground_energy_dense)
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
