"""
N2 / 6-31G bond breaking in CAS(10e,10o):
Exact CAS-FCI vs SPARSE CCIK (selected-CI style Krylov)

- Active space: ncore=2 (freeze N 1s on each atom), nelecas=10, ncas=10
- Orbitals: RHF canonical (via CASCI effective Hamiltonian)
- Benchmark: PySCF FCI within CAS
- CCIK: sparse dict wavefunction in spin-orbital determinant basis
        H|psi> built by enumerating connected singles+doubles (Slater–Condon)
        selection uses PT2-like score on components of H|psi> plus top-|v| stabilizer
        generalized Ritz (H y = E S y) in Krylov basis because truncation breaks orthogonality

Requires: pyscf, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo, fci
from pyscf.mcscf import casci


# ============================================================
# Bit ops + sparse vector utilities
# ============================================================
def bit_count(x: int) -> int:
    return x.bit_count()

def iter_set_bits(x: int):
    while x:
        lsb = x & -x
        i = lsb.bit_length() - 1
        yield i
        x ^= lsb

def occ_list(det: int):
    return list(iter_set_bits(det))

def build_hf_det(norb: int, neleca: int, nelecb: int) -> int:
    """
    Spin-orbital indexing:
      spatial p -> alpha: 2p, beta: 2p+1
    HF determinant in the active space: occupy lowest orbitals for alpha and beta.
    """
    det = 0
    for p in range(neleca):
        det |= (1 << (2 * p))      # alpha
    for p in range(nelecb):
        det |= (1 << (2 * p + 1))  # beta
    return det

def sparse_norm2(v: dict) -> float:
    return float(sum((abs(c) ** 2) for c in v.values()))

def sparse_inner(v: dict, w: dict) -> float:
    if len(v) > len(w):
        v, w = w, v
    s = 0.0 + 0.0j
    for k, cv in v.items():
        cw = w.get(k)
        if cw is not None:
            s += np.conjugate(cv) * cw
    return float(np.real(s))

def sparse_scale(v: dict, a: float) -> dict:
    if a == 0.0:
        return {}
    return {k: a * c for k, c in v.items()}

def sparse_axpy(y: dict, a: float, x: dict) -> dict:
    out = dict(y)
    for k, cx in x.items():
        out[k] = out.get(k, 0.0) + a * cx
        if abs(out[k]) < 1e-16:
            out.pop(k, None)
    return out

def sparse_normalize(v: dict) -> dict:
    n2 = sparse_norm2(v)
    if n2 == 0.0:
        return v
    return sparse_scale(v, 1.0 / np.sqrt(n2))

def sparse_topk(v: dict, k: int) -> dict:
    if k is None or k <= 0 or len(v) <= k:
        return v
    items = sorted(v.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return dict(items)


# ============================================================
# Spin-orbital integrals from spatial CAS integrals
# ============================================================
class SpinOrbIntegrals:
    """
    Provide h(p,q) and <pq||rs> in spin-orbital basis
    from spatial-orbital h1[p,q] and eri[p,q,r,s] = (pq|rs) (chemist).
    """
    def __init__(self, h1_spatial: np.ndarray, eri_spatial_4: np.ndarray):
        self.h1 = np.asarray(h1_spatial)
        self.eri = np.asarray(eri_spatial_4)  # (pq|rs)
        self.norb = self.h1.shape[0]
        self.nso = 2 * self.norb

    def h(self, p: int, q: int) -> float:
        if (p & 1) != (q & 1):
            return 0.0
        P, Q = p >> 1, q >> 1
        return float(self.h1[P, Q])

    def coul(self, p: int, q: int, r: int, s: int) -> float:
        # <pq|rs> nonzero only if spin(p)=spin(r) and spin(q)=spin(s)
        if ((p & 1) != (r & 1)) or ((q & 1) != (s & 1)):
            return 0.0
        P, Q, R, S = p >> 1, q >> 1, r >> 1, s >> 1
        return float(self.eri[P, Q, R, S])

    def exch(self, p: int, q: int, r: int, s: int) -> float:
        # <pq|sr> nonzero only if spin(p)=spin(s) and spin(q)=spin(r)
        if ((p & 1) != (s & 1)) or ((q & 1) != (r & 1)):
            return 0.0
        P, Q, S, R = p >> 1, q >> 1, s >> 1, r >> 1
        return float(self.eri[P, Q, S, R])

    def g_antisym(self, p: int, q: int, r: int, s: int) -> float:
        return self.coul(p, q, r, s) - self.exch(p, q, r, s)


# ============================================================
# Slater–Condon phases and matrix elements
# ============================================================
def sign_remove(det: int, p: int) -> int:
    below = det & ((1 << p) - 1)
    return -1 if (bit_count(below) % 2) else 1

def sign_add(det: int, p: int) -> int:
    below = det & ((1 << p) - 1)
    return -1 if (bit_count(below) % 2) else 1

def det_diagonal_energy(det: int, ints: SpinOrbIntegrals) -> float:
    occ = occ_list(det)
    e1 = sum(ints.h(i, i) for i in occ)
    e2 = 0.0
    for i in occ:
        for j in occ:
            e2 += ints.g_antisym(i, j, i, j)
    return float(e1 + 0.5 * e2)

def single_me(det: int, i: int, a: int, ints: SpinOrbIntegrals) -> float:
    occ = occ_list(det)
    val = ints.h(a, i)
    for j in occ:
        if j == i:
            continue
        val += ints.g_antisym(a, j, i, j)
    return float(val)

def double_me(i: int, j: int, a: int, b: int, ints: SpinOrbIntegrals) -> float:
    return float(ints.g_antisym(a, b, i, j))


# ============================================================
# Sparse matvec: v = H psi (selected-CI style)
# ============================================================
def sparse_matvec(psi: dict, ints: SpinOrbIntegrals) -> dict:
    """
    Build v = H psi by enumerating connected singles+doubles in spin-orbital basis.
    Keeps electron number and spin counts automatically by exciting within alpha/beta channels.
    """
    v = {}
    nso = ints.nso

    for det, c in psi.items():
        if abs(c) < 1e-18:
            continue

        # diagonal
        Hii = det_diagonal_energy(det, ints)
        v[det] = v.get(det, 0.0) + Hii * c

        occ = occ_list(det)
        occ_set = set(occ)

        occ_alpha = [p for p in occ if (p & 1) == 0]
        occ_beta  = [p for p in occ if (p & 1) == 1]
        vir_alpha = [p for p in range(0, nso, 2) if p not in occ_set]
        vir_beta  = [p for p in range(1, nso, 2) if p not in occ_set]

        # singles (alpha)
        for i in occ_alpha:
            si = sign_remove(det, i)
            det_i = det ^ (1 << i)
            for a in vir_alpha:
                val = single_me(det, i, a, ints)
                if abs(val) < 1e-14:
                    continue
                sa = sign_add(det_i, a)
                det1 = det_i | (1 << a)
                v[det1] = v.get(det1, 0.0) + (si * sa) * val * c

        # singles (beta)
        for i in occ_beta:
            si = sign_remove(det, i)
            det_i = det ^ (1 << i)
            for a in vir_beta:
                val = single_me(det, i, a, ints)
                if abs(val) < 1e-14:
                    continue
                sa = sign_add(det_i, a)
                det1 = det_i | (1 << a)
                v[det1] = v.get(det1, 0.0) + (si * sa) * val * c

        # doubles alpha-alpha
        for idx_i in range(len(occ_alpha)):
            i = occ_alpha[idx_i]
            si = sign_remove(det, i)
            det_i = det ^ (1 << i)
            for idx_j in range(idx_i + 1, len(occ_alpha)):
                j = occ_alpha[idx_j]
                sj = sign_remove(det_i, j)
                det_ij = det_i ^ (1 << j)

                for ai in range(len(vir_alpha)):
                    a = vir_alpha[ai]
                    sa = sign_add(det_ij, a)
                    det_a = det_ij | (1 << a)
                    for bi in range(ai + 1, len(vir_alpha)):
                        b = vir_alpha[bi]
                        sb = sign_add(det_a, b)
                        val = double_me(i, j, a, b, ints)
                        if abs(val) < 1e-14:
                            continue
                        det2 = det_a | (1 << b)
                        v[det2] = v.get(det2, 0.0) + (si * sj * sa * sb) * val * c

        # doubles beta-beta
        for idx_i in range(len(occ_beta)):
            i = occ_beta[idx_i]
            si = sign_remove(det, i)
            det_i = det ^ (1 << i)
            for idx_j in range(idx_i + 1, len(occ_beta)):
                j = occ_beta[idx_j]
                sj = sign_remove(det_i, j)
                det_ij = det_i ^ (1 << j)

                for ai in range(len(vir_beta)):
                    a = vir_beta[ai]
                    sa = sign_add(det_ij, a)
                    det_a = det_ij | (1 << a)
                    for bi in range(ai + 1, len(vir_beta)):
                        b = vir_beta[bi]
                        sb = sign_add(det_a, b)
                        val = double_me(i, j, a, b, ints)
                        if abs(val) < 1e-14:
                            continue
                        det2 = det_a | (1 << b)
                        v[det2] = v.get(det2, 0.0) + (si * sj * sa * sb) * val * c

        # doubles alpha-beta
        for i in occ_alpha:
            si = sign_remove(det, i)
            det_i = det ^ (1 << i)
            for j in occ_beta:
                sj = sign_remove(det_i, j)
                det_ij = det_i ^ (1 << j)

                for a in vir_alpha:
                    sa = sign_add(det_ij, a)
                    det_a = det_ij | (1 << a)
                    for b in vir_beta:
                        sb = sign_add(det_a, b)
                        val = double_me(i, j, a, b, ints)
                        if abs(val) < 1e-14:
                            continue
                        det2 = det_a | (1 << b)
                        v[det2] = v.get(det2, 0.0) + (si * sj * sa * sb) * val * c

    # prune tiny
    v = {k: vk for k, vk in v.items() if abs(vk) > 1e-15}
    return v


# ============================================================
# Generalized eigenproblem for Ritz
# ============================================================
def generalized_eig(H: np.ndarray, S: np.ndarray, eps=1e-12):
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


# ============================================================
# Sparse CCIK (CIPSI-scored inexact Krylov)
# ============================================================
def ccik_sparse_ground_energy(
    ints: SpinOrbIntegrals,
    neleca: int,
    nelecb: int,
    m: int = 40,
    nkeep_q: int = 20000,
    nkeep_v: int = 60000,
    nadd_pt2: int = 30000,
    Kv: int = 20000,
    verbose: bool = False,
):
    """
    Returns ground (electronic) energy via sparse CCIK in the CAS Hamiltonian.

    Selection/truncation rule for each step:
      keep_set = support(qk) U topKv(|Hq|) U topN(PT2score)
      then truncate v to top nkeep_v by |coeff|
      build q_{k+1} via full reorthogonalization, then truncate to top nkeep_q
      generalized Ritz at end
    """
    norb = ints.norb
    det0 = build_hf_det(norb, neleca, nelecb)
    q0 = sparse_normalize({det0: 1.0})

    Q = [q0]
    Hq_list = []

    def hdiag_det(det: int) -> float:
        return det_diagonal_energy(det, ints)

    for k in range(m):
        qk = Q[-1]
        v_full = sparse_matvec(qk, ints)

        Ek = sparse_inner(qk, v_full) / max(sparse_norm2(qk), 1e-30)

        # top-|v|
        topv = sorted(v_full.items(), key=lambda kv: abs(kv[1]), reverse=True)
        keep_set = set(qk.keys())
        if Kv is not None and Kv > 0:
            keep_set |= set(det for det, _ in topv[:Kv])

        # PT2-like score: |v(det)|^2 / |E - Hdiag(det)|
        scores = []
        for det, val in v_full.items():
            denom = abs(Ek - hdiag_det(det))
            denom = max(denom, 1e-12)
            scores.append((det, (abs(val) ** 2) / denom))
        scores.sort(key=lambda x: x[1], reverse=True)
        if nadd_pt2 is not None and nadd_pt2 > 0:
            keep_set |= set(det for det, _ in scores[:nadd_pt2])

        v_keep = {det: v_full[det] for det in keep_set if det in v_full}
        v_keep = sparse_topk(v_keep, nkeep_v)

        if verbose:
            print(f"[k={k:02d}] Ek~{Ek:+.10f} |q|={len(qk):6d} |Hq(full)|={len(v_full):7d} |Hq(keep)|={len(v_keep):6d}")

        Hq_list.append(v_keep)

        # r = v_keep - sum_j <Qj|v_keep> Qj
        r = dict(v_keep)
        for j, Qj in enumerate(Q):
            proj = sparse_inner(Qj, r) / max(sparse_norm2(Qj), 1e-30)
            if abs(proj) > 0:
                r = sparse_axpy(r, -proj, Qj)

        if sparse_norm2(r) < 1e-24:
            if verbose:
                print("  [break] Krylov breakdown (near linear dependence).")
            break

        q_next = sparse_normalize(r)
        q_next = sparse_topk(q_next, nkeep_q)
        q_next = sparse_normalize(q_next)
        Q.append(q_next)

    # Build projected H and S and solve generalized Ritz
    m_eff = len(Hq_list)
    Hproj = np.zeros((m_eff, m_eff))
    Sproj = np.zeros((m_eff, m_eff))

    for i in range(m_eff):
        for j in range(m_eff):
            Sproj[i, j] = sparse_inner(Q[i], Q[j])
            Hproj[i, j] = sparse_inner(Q[i], Hq_list[j])

    evals, _ = generalized_eig(Hproj, Sproj)
    return float(evals[0])


# ============================================================
# Build CAS(10,10) Hamiltonian for N2 / 6-31G (same as before)
# ============================================================
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
    """
    Returns:
      h1eff (ncas,ncas), eri4 (ncas,ncas,ncas,ncas) chemist (pq|rs), ecore
    Total energy = ecore + E_cas_elec
    """
    mf = scf.RHF(mol).run()

    mc = casci.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.mo_coeff = mf.mo_coeff

    h1eff, ecore = mc.get_h1eff()
    eri_packed = mc.get_h2eff()
    eri4 = ao2mo.restore(1, eri_packed, ncas)  # full (pq|rs)

    neleca = nelecas // 2
    nelecb = nelecas - neleca
    nelec_cas = (neleca, nelecb)

    return np.asarray(h1eff), np.asarray(eri4), float(ecore), ncas, nelec_cas

def exact_cas_fci_energy(h1eff, eri4, ncas, nelec_cas):
    # PySCF FCI wants eri in "eri8" (symm 8-fold) with absorb_h1e done internally by solver
    eri8 = ao2mo.restore(8, ao2mo.restore(4, eri4, ncas), ncas)  # safe but a bit redundant
    # Better: pack then restore(8) — easiest:
    eri_packed = ao2mo.restore(4, eri4, ncas)  # 4-fold packed
    eri8 = ao2mo.restore(8, eri_packed, ncas)

    cisolver = fci.direct_spin1.FCI()
    e_elec, _ = cisolver.kernel(h1eff, eri8, ncas, nelec_cas)
    return float(e_elec)


# ============================================================
# Scan + plot: keep the same system and include FCI
# ============================================================
def run_scan():
    basis = "6-31g"
    ncore = 2
    ncas = 10
    nelecas = 10
    neleca = nelecas // 2
    nelecb = nelecas - neleca

    Rs = np.linspace(3.0,3.0,1)

    # Sparse CCIK knobs (start here; increase if dissociation still off)
    # These control how many determinants you keep in q and Hq.
    m = 45
    nkeep_q = 30000
    nkeep_v = 90000
    nadd_pt2 = 40000
    Kv = 25000

    E_fci_tot = []
    E_ccik_tot = []

    for R in Rs:
        mol = make_n2_mol(R, basis=basis)

        h1eff, eri4, ecore, ncas_, nelec_cas = build_cas_hamiltonian(
            mol, ncas=ncas, nelecas=nelecas, ncore=ncore
        )

        # Exact CAS-FCI (benchmark)
        e_fci_cas = exact_cas_fci_energy(h1eff, eri4, ncas_, nelec_cas)
        e_fci_tot = ecore + e_fci_cas
        print("exact energy=", e_fci_tot)

        # Sparse CCIK
        ints = SpinOrbIntegrals(h1eff, eri4)
        e_ccik_cas = ccik_sparse_ground_energy(
            ints, neleca, nelecb,
            m=m,
            nkeep_q=nkeep_q,
            nkeep_v=nkeep_v,
            nadd_pt2=nadd_pt2,
            Kv=Kv,
            verbose=True
        )
        e_ccik_tot = ecore + e_ccik_cas

        E_fci_tot.append(e_fci_tot)
        E_ccik_tot.append(e_ccik_tot)

        print(f"R={R:5.2f} Å | CAS-FCI={e_fci_tot:+.10f} | sparse-CCIK={e_ccik_tot:+.10f} | Δ={e_ccik_tot-e_fci_tot:+.3e}")

    Rs = np.array(Rs)
    E_fci_tot = np.array(E_fci_tot)
    E_ccik_tot = np.array(E_ccik_tot)

    plt.figure()
    plt.plot(Rs, E_fci_tot, marker="o", label="Exact CAS(10,10)-FCI / 6-31G")
    plt.plot(Rs, E_ccik_tot, marker="s", label="Sparse CCIK (selected-CI matvec)")
    plt.xlabel("N–N bond length (Å)")
    plt.ylabel("Total energy (Ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("n2_631g_cas10_10_fci_vs_sparse_ccik.png", dpi=200)
    plt.show()

    print("\nSaved plot: n2_631g_cas10_10_fci_vs_sparse_ccik.png")


if __name__ == "__main__":
    run_scan()
