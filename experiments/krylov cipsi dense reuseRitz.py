"""
N2 / 6-31G bond breaking in CAS(10e,10o):
Exact CAS-FCI vs
  (1) Dense CCIK + thick restart (reusing best Ritz vectors)
  (2) CIPSI variational only (NO PT2)

This version is robust to PySCF davidson1 API differences:
- precond required in some versions
- list-of-vectors interface required in some versions
- return signature/order differs across versions

Outputs:
- Prints per-geometry energies and errors
- Saves plot: n2_631g_cas10_10_fci_vs_ccik_vs_cipsi_var.png

Requires: pyscf, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo, fci
from pyscf.mcscf import casci
from pyscf.fci import cistring, direct_spin1
from pyscf.lib import linalg_helper


# -----------------------------
# Utilities
# -----------------------------
def normalize(ci):
    nrm = np.linalg.norm(ci.ravel())
    return ci if nrm == 0 else (ci / nrm)

def inner(ci1, ci2):
    return float(np.vdot(ci1.ravel(), ci2.ravel()))

def compress_keep_top_mask(ci, nkeep=None, min_abs=0.0):
    """Keep up to nkeep largest |coeff| entries (and those >= min_abs)."""
    flat = ci.ravel()
    absflat = np.abs(flat)

    if nkeep is None or nkeep >= flat.size:
        return (absflat >= min_abs).reshape(ci.shape)

    eligible = np.where(absflat >= min_abs)[0]
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

def apply_mask(ci, mask):
    out = np.zeros_like(ci)
    out[mask] = ci[mask]
    return out

def occ_list_to_bitstring(occ):
    s = 0
    for p in occ:
        s |= (1 << p)
    return s

def generalized_eigh(H, S, eps=1e-12):
    """Solve H x = e S x via Cholesky (H,S symmetric; S SPD-ish)."""
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

def gram_schmidt_list(vecs, tol=1e-12):
    """Orthonormalize list of CI tensors; drop near-dependent ones."""
    Q = []
    for v in vecs:
        r = v.copy()
        for q in Q:
            r -= inner(q, r) * q
        nr = np.linalg.norm(r.ravel())
        if nr > tol:
            Q.append(r / nr)
    return Q


def _as_1d(x):
    """Convert x to a 1D numpy array if possible."""
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.dtype == object:
        # object arrays often mean "nested"; refuse here
        return None
    return arr.ravel()

def parse_davidson_output(out, dim, nroots=1):
    """
    Robustly extract (energies, vecs) from davidson1 output with unknown ordering.

    We look through returned items and find:
      - energies: scalar or small 1D array (<= ~20 elements)
      - vecs: either
          * list/tuple of length nroots containing 1D arrays length dim
          * ndarray shape (nroots, dim) or (dim,)

    Returns:
      (E0_float, v0_1d)
    """
    if not isinstance(out, (tuple, list)):
        raise TypeError("Unexpected davidson1 output type")

    energies_candidate = None
    vecs_candidate = None

    # 1) find vector-like object of size dim
    for item in out:
        # case: list/tuple of vectors
        if isinstance(item, (list, tuple)) and len(item) > 0:
            # unwrap nested singletons like [[v]]
            v = item[0]
            while isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            v1 = _as_1d(v)
            if v1 is not None and v1.size == dim:
                vecs_candidate = item
                break

        # case: ndarray directly
        v1 = _as_1d(item)
        if v1 is not None:
            arr = np.asarray(item)
            if arr.ndim == 1 and arr.size == dim:
                vecs_candidate = arr
                break
            if arr.ndim == 2 and arr.shape[0] == nroots and arr.shape[1] == dim:
                vecs_candidate = arr
                break

    # 2) find energies-like object (small numeric)
    for item in out:
        e1 = _as_1d(item)
        if e1 is None:
            continue
        if e1.size == 0:
            continue
        # exclude things that look like vectors
        if e1.size == dim:
            continue
        # energies are typically small (nroots elements) or scalar
        if e1.size <= max(20, nroots):
            # prefer float dtype-ish
            if np.issubdtype(e1.dtype, np.number):
                energies_candidate = e1
                break

    if vecs_candidate is None:
        # As a last resort, try a different heuristic: anything big is probably vectors
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
            f"Could not parse davidson1 output. "
            f"Found energies={energies_candidate is not None}, vecs={vecs_candidate is not None}. "
            f"Output types={[type(x) for x in out]}"
        )

    # extract E0
    if energies_candidate.size == 1:
        E0 = float(energies_candidate[0])
    else:
        E0 = float(energies_candidate[0])

    # extract first vector v0
    if isinstance(vecs_candidate, (list, tuple)):
        v = vecs_candidate[0]
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        v0 = _as_1d(v)
    else:
        arr = np.asarray(vecs_candidate)
        if arr.ndim == 2:
            v0 = arr[0].ravel()
        else:
            v0 = arr.ravel()

    if v0 is None or v0.size != dim:
        raise RuntimeError(f"Parsed vector has wrong size: {None if v0 is None else v0.size}, expected {dim}")

    return E0, v0


# -----------------------------
# CAS Hamiltonian for N2 / 6-31G in CAS(10,10)
# -----------------------------
def make_n2_mol(R_ang, basis="6-31g"):
    z = R_ang / 2.0
    atom = f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0
    )
    return mol

def build_cas_hamiltonian(mol, ncas=10, nelecas=10, ncore=2):
    mf = scf.RHF(mol).run()

    mc = casci.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.mo_coeff = mf.mo_coeff

    h1eff, ecore = mc.get_h1eff()
    eri_packed = mc.get_h2eff()
    eri8 = ao2mo.restore(8, eri_packed, ncas)

    neleca = nelecas // 2
    nelecb = nelecas - neleca
    nelec_cas = (neleca, nelecb)

    return np.asarray(h1eff), np.asarray(eri8), float(ecore), ncas, nelec_cas

def exact_cas_fci_energy(h1eff, eri8, ncas, nelec_cas):
    cisolver = fci.direct_spin1.FCI()
    e_elec, _ = cisolver.kernel(h1eff, eri8, ncas, nelec_cas)
    return float(e_elec)


# -----------------------------
# Dense CCIK: thick restart
# -----------------------------
def ccik_cycle_dense_multi_start(
    h1, eri8, norb, nelec,
    start_vecs,
    m_cycle=25,
    nadd=2000, nkeep=10000, Kv=3000,
    verbose=False
):
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag).reshape(na, nb)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)

    def H_contract(ci):
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    Q = gram_schmidt_list([normalize(v) for v in start_vecs], tol=1e-12)
    if len(Q) == 0:
        raise RuntimeError("All start vectors vanished after orthogonalization.")

    supports = [compress_keep_top_mask(q, nkeep=nkeep) for q in Q]
    Q = [normalize(apply_mask(q, s)) for q, s in zip(Q, supports)]
    supports = [compress_keep_top_mask(q, nkeep=nkeep) for q in Q]

    while len(Q) < m_cycle:
        qk = Q[-1]
        supp_k = supports[-1]

        v_full = H_contract(qk)
        Ek = inner(qk, v_full)

        denom = np.abs(Ek - hdiag)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = score.copy()
        score_ext[supp_k] = 0.0

        select_mask = np.zeros_like(score_ext, dtype=bool)
        if nadd is not None and nadd > 0:
            flat = score_ext.ravel()
            pos = np.where(flat > 0)[0]
            if pos.size > 0:
                nsel = min(nadd, pos.size)
                vals = flat[pos]
                top_local = np.argpartition(vals, -nsel)[-nsel:]
                idx = pos[top_local]
                sflat = np.zeros_like(flat, dtype=bool)
                sflat[idx] = True
                select_mask = sflat.reshape(score_ext.shape)

        topv_mask = compress_keep_top_mask(v_full, nkeep=Kv) if (Kv is not None and Kv > 0) else None

        supp_v = supp_k | select_mask
        if topv_mask is not None:
            supp_v |= topv_mask

        if verbose:
            v2_total = float(np.sum(np.abs(v_full) ** 2))
            v2_kept = float(np.sum(np.abs(v_full[supp_v]) ** 2))
            tail_frac = (max(v2_total - v2_kept, 0.0)) / max(v2_total, 1e-30)
            print(f"      |Q|={len(Q):02d}  E(q)={Ek:+.10f}  tail≈{tail_frac:.2e}  supp_v={int(supp_v.sum())}")

        v = apply_mask(v_full, supp_v)

        r = v.copy()
        for qj in Q:
            r -= inner(qj, r) * qj

        nr = np.linalg.norm(r.ravel())
        if nr < 1e-12:
            break

        q_next = r / nr
        supp_next = compress_keep_top_mask(q_next, nkeep=nkeep)
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
    h1, eri8, norb, nelec,
    m_cycle=25, ncycles=10, nroot=3, tol=1e-6,
    nadd=2000, nkeep=10000, Kv=3000,
    verbose=False
):
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

    starts = [q0]
    E_prev = None

    for cyc in range(ncycles):
        evals, X, Q = ccik_cycle_dense_multi_start(
            h1, eri8, norb, nelec,
            start_vecs=starts,
            m_cycle=m_cycle,
            nadd=nadd, nkeep=nkeep, Kv=Kv,
            verbose=verbose
        )

        nroot_eff = min(nroot, X.shape[1])
        new_starts = []
        for r in range(nroot_eff):
            psi = np.zeros_like(Q[0])
            coeffs = X[:, r]
            for i, Qi in enumerate(Q):
                psi += coeffs[i] * Qi
            psi = normalize(psi)

            supp = compress_keep_top_mask(psi, nkeep=nkeep)
            psi = normalize(apply_mask(psi, supp))
            new_starts.append(psi)

        starts = gram_schmidt_list(new_starts, tol=1e-12)
        if len(starts) == 0:
            starts = [new_starts[0]]

        E0 = float(evals[0])

        if verbose:
            if E_prev is None:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f}")
            else:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f}  Δ={E0-E_prev:+.3e}")

        if E_prev is not None and abs(E0 - E_prev) < tol:
            return E0
        E_prev = E0

    return float(E_prev)


# -----------------------------
# CIPSI variational only (NO PT2)
# -----------------------------
def cipsi_dense_variational(
    h1, eri8, norb, nelec,
    niter=10,
    nadd=2000, ndet_max=10000, Kv=3000,
    davidson_tol=1e-10,
    verbose=False
):
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    dim = na * nb

    hdiag = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag = np.asarray(hdiag).reshape(na, nb)
    diag_full = hdiag.ravel()

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)

    def H_contract(ci):
        return np.asarray(direct_spin1.contract_2e(h2eff, ci, norb, nelec))

    occ_a = list(range(neleca))
    occ_b = list(range(nelecb))
    addr_a = cistring.str2addr(norb, neleca, occ_list_to_bitstring(occ_a))
    addr_b = cistring.str2addr(norb, nelecb, occ_list_to_bitstring(occ_b))

    psi = np.zeros((na, nb))
    psi[addr_a, addr_b] = 1.0
    psi = normalize(psi)

    S = np.zeros((na, nb), dtype=bool)
    S[addr_a, addr_b] = True

    def proj_matvec(x):
        xS = apply_mask(x.reshape(na, nb), S)
        y = H_contract(xS)
        yS = apply_mask(y, S)
        return yS.ravel()

    def aop(xs):
        return [proj_matvec(x) for x in xs]

    def precond(r, e0, x0=None):
        denom = e0 - diag_full
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return r / denom

    E_prev = None
    E_var = None

    for it in range(niter):
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
            tol=davidson_tol,
            max_cycle=60,
            max_space=20,
            nroots=1,
            verbose=0
        )

        E0, v0 = parse_davidson_output(out, dim=dim, nroots=1)
        E_var = float(E0)

        psiS = v0.reshape(na, nb)
        psi = normalize(apply_mask(psiS, S))

        # Selection from v_full = H psi
        v_full = H_contract(psi)

        denom = np.abs(E_var - hdiag)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        score = (np.abs(v_full) ** 2) / denom

        score_ext = score.copy()
        score_ext[S] = 0.0

        add_mask = np.zeros_like(S, dtype=bool)
        if nadd is not None and nadd > 0:
            flat = score_ext.ravel()
            pos = np.where(flat > 0)[0]
            if pos.size > 0:
                nsel = min(nadd, pos.size)
                vals = flat[pos]
                top_local = np.argpartition(vals, -nsel)[-nsel:]
                idx = pos[top_local]
                sflat = np.zeros_like(flat, dtype=bool)
                sflat[idx] = True
                add_mask = sflat.reshape(S.shape)

        topv_mask = compress_keep_top_mask(v_full, nkeep=Kv) if (Kv is not None and Kv > 0) else None

        S_new = S | add_mask
        if topv_mask is not None:
            S_new |= topv_mask

        if ndet_max is not None and S_new.sum() > ndet_max:
            tmp = np.zeros_like(psi)
            tmp[S_new] = psi[S_new]
            keep_mask = compress_keep_top_mask(tmp, nkeep=ndet_max)

            keep_mask |= S
            if topv_mask is not None:
                keep_mask |= topv_mask

            if keep_mask.sum() > ndet_max:
                tmp2 = np.zeros_like(psi)
                tmp2[keep_mask] = psi[keep_mask]
                keep_mask = compress_keep_top_mask(tmp2, nkeep=ndet_max)
                if topv_mask is not None:
                    keep_mask |= topv_mask

            S = keep_mask
        else:
            S = S_new

        if verbose:
            print(f"    [CIPSI it={it:02d}] E_var={E_var:+.10f}  |S|={int(S.sum())}")

        if E_prev is not None and abs(E_var - E_prev) < 1e-10:
            break
        E_prev = E_var

    return float(E_var)


# -----------------------------
# Scan + compare
# -----------------------------
def run_scan():
    Rs = np.linspace(0.8, 3.0, 16)

    basis = "6-31g"
    ncore = 2
    ncas = 10
    nelecas = 10

    # Shared knobs
    nadd = 2000
    nkeep = 10000
    Kv = 3000

    # CCIK knobs
    m_cycle = 25
    ncycles = 10
    nroot = 3
    tol_ccik = 1e-6

    # CIPSI knobs
    niter_cipsi = 10

    E_fci_tot = []
    E_ccik_tot = []
    E_cipsi_tot = []

    for R in Rs:
        mol = make_n2_mol(R, basis=basis)
        h1eff, eri8, ecore, ncas_, nelec_cas = build_cas_hamiltonian(
            mol, ncas=ncas, nelecas=nelecas, ncore=ncore
        )

        e_fci_cas = exact_cas_fci_energy(h1eff, eri8, ncas_, nelec_cas)
        e_fci = ecore + e_fci_cas

        e_ccik_cas = ccik_ground_energy_dense_thick_restart(
            h1eff, eri8, ncas_, nelec_cas,
            m_cycle=m_cycle, ncycles=ncycles, nroot=nroot, tol=tol_ccik,
            nadd=nadd, nkeep=nkeep, Kv=Kv,
            verbose=False
        )
        e_ccik = ecore + e_ccik_cas

        e_cipsi_cas = cipsi_dense_variational(
            h1eff, eri8, ncas_, nelec_cas,
            niter=niter_cipsi,
            nadd=nadd, ndet_max=nkeep, Kv=Kv,
            davidson_tol=1e-10,
            verbose=False
        )
        e_cipsi = ecore + e_cipsi_cas

        E_fci_tot.append(e_fci)
        E_ccik_tot.append(e_ccik)
        E_cipsi_tot.append(e_cipsi)

        print(
            f"R={R:5.2f} Å | FCI={e_fci:+.10f} | "
            f"CCIK(thick)={e_ccik:+.10f} (Δ={e_ccik-e_fci:+.3e}) | "
            f"CIPSI-var={e_cipsi:+.10f} (Δ={e_cipsi-e_fci:+.3e})"
        )

    Rs = np.array(Rs)
    E_fci_tot = np.array(E_fci_tot)
    E_ccik_tot = np.array(E_ccik_tot)
    E_cipsi_tot = np.array(E_cipsi_tot)

    plt.figure()
    plt.plot(Rs, E_fci_tot, marker="o", label="Exact CAS(10,10)-FCI / 6-31G")
    plt.plot(Rs, E_ccik_tot, marker="s", label="CCIK (dense) + thick restart")
    plt.plot(Rs, E_cipsi_tot, marker="^", label="CIPSI variational (cap=nkeep)")
    plt.xlabel("N–N bond length (Å)")
    plt.ylabel("Total energy (Ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("n2_631g_cas10_10_fci_vs_ccik_vs_cipsi_var.png", dpi=200)
    plt.show()

    print("\nSaved plot: n2_631g_cas10_10_fci_vs_ccik_vs_cipsi_var.png")


if __name__ == "__main__":
    run_scan()
