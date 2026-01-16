import pytest


pyscf = pytest.importorskip("pyscf")

import numpy as np

from ccik import (
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_ai_selector_krylov,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_fciqmc_krylov,
    cipsi_dense_variational,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.ai_selector_krylov import AISelectorKrylovParams
from ccik.params import CCIKParams, CCIKThickRestartParams, CIPSIParams, FCIQMCKrylovParams
from ccik.pyscf_cas import CASSpec


def _build_h2_cas() -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int], float]:
    """Return (h1, eri8, ecore, ncas, nelec, e_fci_tot) for a tiny H2 CAS."""

    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )

    cas = CASSpec(ncas=2, nelecas=2, ncore=0)
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    e_fci_cas = exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec)
    e_fci_tot = float(ecore + e_fci_cas)
    return h1, eri8, float(ecore), int(ncas), nelec, e_fci_tot


def _build_h4_chain_cas(r_ang: float = 1.5) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int], float]:
    """Return (h1, eri8, ecore, ncas, nelec, e_fci_tot) for a small H4 chain CAS.

    Chosen to be small but nontrivial (determinant dimension = 36 for CAS(4,4) singlet).
    """

    atom = "; ".join(
        [
            f"H 0 0 {0.0 * r_ang}",
            f"H 0 0 {1.0 * r_ang}",
            f"H 0 0 {2.0 * r_ang}",
            f"H 0 0 {3.0 * r_ang}",
        ]
    )
    mol = make_mol_pyscf(
        atom=atom,
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )

    cas = CASSpec(ncas=4, nelecas=4, ncore=0)
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    e_fci_cas = exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec)
    e_fci_tot = float(ecore + e_fci_cas)
    return h1, eri8, float(ecore), int(ncas), nelec, e_fci_tot


def test_h2_smoke_ccik_thick_and_fciqmc_krylov() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    thick_params = CCIKThickRestartParams(
        m_cycle=10,
        ncycles=1,
        nroot=2,
        tol=1e-10,
        nadd=50,
        nkeep=200,
        Kv=50,
        orth_tol=1e-12,
        verbose=False,
    )
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))

    fciqmc_params = FCIQMCKrylovParams(
        m=12,
        nadd=50,
        nkeep=200,
        Kv=50,
        n_walkers=5000,
        seed=0,
        parent_power=1.0,
        p_double=0.6,
        mixed_double_weight=1.0,
        eps_denom=1e-12,
        orth_tol=1e-12,
        verbose=False,
    )
    st: dict[str, int] = {}
    e_fqmc = float(
        ecore + ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=fciqmc_params, stats=st)
    )

    # Thick restart should be very accurate for this tiny CAS.
    assert abs(e_thick - e_fci_tot) < 1e-6

    # FCIQMC-Krylov is stochastic in its selection, but with a seed and this tiny system
    # it should still be close to FCI.
    assert abs(e_fqmc - e_fci_tot) < 1e-3

    assert "m_eff" in st
    assert "ndet_sum" in st
    assert "ndet_union" in st


def test_h2_all_algorithms_agree_with_fci_when_full_space_kept() -> None:
    """End-to-end regression for the three algorithm families.

    We choose parameters so the selected/compressed support includes the entire (tiny) CAS space,
    making the answer essentially exact and stable across versions.
    """

    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    # Full CI tensor dimension for CAS(2,2) is 4; set knobs to comfortably include all.
    ccik_params = CCIKParams(m=8, nadd=50, nkeep=200, Kv=50, verbose=False, orth_tol=1e-12)
    thick_params = CCIKThickRestartParams(
        m_cycle=8,
        ncycles=1,
        nroot=2,
        tol=1e-12,
        nadd=50,
        nkeep=200,
        Kv=50,
        verbose=False,
        orth_tol=1e-12,
    )
    cipsi_params = CIPSIParams(niter=5, nadd=50, ndet_max=200, Kv=50, davidson_tol=1e-12, verbose=False)
    fciqmc_params = FCIQMCKrylovParams(
        m=8,
        nadd=50,
        nkeep=200,
        Kv=50,
        n_walkers=2000,
        seed=0,
        parent_power=1.0,
        p_double=0.6,
        mixed_double_weight=1.0,
        eps_denom=1e-12,
        orth_tol=1e-12,
        verbose=False,
    )

    st_ccik: dict[str, int] = {}
    st_fqmc: dict[str, int] = {}
    st_cipsi: dict[str, int] = {}

    e_ccik = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=ccik_params, stats=st_ccik))
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))
    e_cipsi = float(ecore + cipsi_dense_variational(h1, eri8, ncas, nelec, params=cipsi_params, stats=st_cipsi))
    e_fqmc = float(ecore + ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=fciqmc_params, stats=st_fqmc))

    # With full-space retention on this tiny system, everything should be essentially exact.
    assert abs(e_ccik - e_fci_tot) < 1e-10
    assert abs(e_thick - e_fci_tot) < 1e-10
    assert abs(e_cipsi - e_fci_tot) < 1e-10
    assert abs(e_fqmc - e_fci_tot) < 1e-10

    assert st_ccik["m_eff"] >= 1
    assert st_ccik["ndet_sum"] >= st_ccik["ndet_union"]
    assert st_fqmc["m_eff"] >= 1
    assert st_fqmc["ndet_sum"] >= st_fqmc["ndet_union"]
    assert st_cipsi["ndet"] >= 1


def test_h2_ai_selector_krylov_backends_smoke() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    p = AISelectorKrylovParams(
        m=10,
        nadd=50,
        nkeep=200,
        Kv=50,
        n_walkers=2000,
        seed=0,
        verbose=False,
    )

    st1: dict[str, int] = {}
    e_cipsi_sel = float(
        ecore
        + ccik_ground_energy_ai_selector_krylov(
            h1,
            eri8,
            ncas,
            nelec,
            selector_backend="cipsi",
            params=p,
            stats=st1,
        )
    )
    assert abs(e_cipsi_sel - e_fci_tot) < 1e-6
    assert "m_eff" in st1

    st2: dict[str, int] = {}
    e_gnn_sel = float(
        ecore
        + ccik_ground_energy_ai_selector_krylov(
            h1,
            eri8,
            ncas,
            nelec,
            selector_backend="gnn",
            params=p,
            stats=st2,
        )
    )
    # With fallback heuristic + tiny system, still should be essentially exact.
    assert abs(e_gnn_sel - e_fci_tot) < 1e-6
    assert "m_eff" in st2


def test_fciqmc_krylov_reproducible_with_seed() -> None:
    h1, eri8, ecore, ncas, nelec, _e_fci_tot = _build_h2_cas()

    params = FCIQMCKrylovParams(
        m=10,
        nadd=10,
        nkeep=50,
        Kv=10,
        n_walkers=5000,
        seed=123,
        verbose=False,
        orth_tol=1e-12,
    )

    e1 = float(ecore + ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=params))
    e2 = float(ecore + ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=params))
    assert np.isclose(e1, e2, atol=0.0, rtol=0.0)


def test_h4_medium_selection_algorithms_close_to_fci() -> None:
    """Exercise selection/compression on a slightly larger CAS.

    This test is meant to catch regressions in:
    - selection masks and masking logic,
    - Krylov restart plumbing,
    - CIPSI davidson projection,
    while keeping runtime small.
    """

    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h4_chain_cas(r_ang=1.5)

    thick_params = CCIKThickRestartParams(
        m_cycle=12,
        ncycles=1,
        nroot=3,
        tol=1e-10,
        nadd=12,
        nkeep=18,
        Kv=10,
        orth_tol=1e-12,
        verbose=False,
    )
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))

    cipsi_params = CIPSIParams(
        niter=6,
        nadd=12,
        ndet_max=24,
        Kv=10,
        davidson_tol=1e-10,
        verbose=False,
    )
    e_cipsi = float(ecore + cipsi_dense_variational(h1, eri8, ncas, nelec, params=cipsi_params))

    fciqmc_params = FCIQMCKrylovParams(
        m=18,
        nadd=12,
        nkeep=18,
        Kv=10,
        n_walkers=10000,
        seed=0,
        parent_power=1.0,
        p_double=0.6,
        mixed_double_weight=1.0,
        eps_denom=1e-12,
        orth_tol=1e-12,
        verbose=False,
    )
    e_fqmc = float(ecore + ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=fciqmc_params))

    # CIPSI is variational (Rayleigh-Ritz in a subspace).
    assert e_cipsi >= (e_fci_tot - 1e-10)

    # The Krylov methods are not strictly variational under compression; check closeness.
    assert abs(e_thick - e_fci_tot) < 2e-2
    assert abs(e_fqmc - e_fci_tot) < 2e-2


def test_h4_ai_selector_krylov_gnn_fallback_close_to_fci() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h4_chain_cas(r_ang=1.5)

    p = AISelectorKrylovParams(
        m=18,
        nadd=12,
        nkeep=18,
        Kv=10,
        n_walkers=10000,
        seed=0,
        verbose=False,
    )
    e = float(
        ecore
        + ccik_ground_energy_ai_selector_krylov(
            h1,
            eri8,
            ncas,
            nelec,
            selector_backend="gnn",
            params=p,
        )
    )
    assert abs(e - e_fci_tot) < 2e-2
