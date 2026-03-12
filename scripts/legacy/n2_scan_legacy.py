"""Legacy-style N2 scan preserved for reference, limited to the three active solvers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (
    build_cas_hamiltonian_pyscf,
    cas_spec_from_dict,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_stochastic,
    ccik_params_from_dict,
    ccik_stochastic_params_from_dict,
    ccik_thick_restart_params_from_dict,
    exact_cas_fci_energy_pyscf,
    load_config,
    make_mol_pyscf,
    run_method_from_dict,
)


def make_n2_mol(R_ang: float, basis: str = "6-31g"):
    z = R_ang / 2.0
    atom = f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """
    return make_mol_pyscf(atom=atom, basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)


def run_scan() -> None:
    parser = argparse.ArgumentParser(description="Legacy N2 scan (three active CCIK solvers)")
    parser.add_argument("--config", default=None, help="Path to TOML config")
    args, _ = parser.parse_known_args()

    config_path = args.config
    if config_path is None:
        local_default = _ROOT / "configs" / "config.toml"
        if local_default.exists():
            config_path = str(local_default)

    cfg = load_config(config_path)
    scan = cfg.get("scan", {})
    Rs = np.linspace(float(scan.get("R_min", 0.8)), float(scan.get("R_max", 3.0)), int(scan.get("n_points", 16)))

    basis = str(cfg.get("molecule", {}).get("basis", "6-31g"))
    method = run_method_from_dict(cfg.get("run", {})).strip()

    cas = cas_spec_from_dict(cfg.get("cas", {}))
    ccik_params = ccik_params_from_dict(cfg.get("ccik", {}))
    thick_params = ccik_thick_restart_params_from_dict(cfg.get("ccik_thick", {}))
    stochastic_params = ccik_stochastic_params_from_dict(cfg.get("ccik_stochastic", {}))

    e_fci_totals = []
    e_alg_totals = []

    for R in Rs:
        mol = make_n2_mol(float(R), basis=basis)
        h1eff, eri8, ecore, ncas, nelec_cas = build_cas_hamiltonian_pyscf(mol, cas=cas)
        e_fci_tot = float(ecore + exact_cas_fci_energy_pyscf(h1eff, eri8, ncas, nelec_cas))

        if method == "ccik":
            e_alg_cas = ccik_ground_energy_dense(h1eff, eri8, ncas, nelec_cas, params=ccik_params)
            label = "CCIK"
        elif method in ("ccik_thick", "ccik_thick_restart", "reuse_ritz", "reuseRitz"):
            e_alg_cas = ccik_ground_energy_dense_thick_restart(h1eff, eri8, ncas, nelec_cas, params=thick_params)
            label = "CCIK(thick)"
        elif method in ("ccik_stochastic", "stochastic", "ccikstochastic", "fciqmckrylov", "fciqmc_krylov"):
            e_alg_cas = ccik_ground_energy_stochastic(h1eff, eri8, ncas, nelec_cas, params=stochastic_params)
            label = "CCIK-stochastic"
        else:
            raise ValueError(f"Unknown run.method={method!r}. Expected ccik, ccik_thick, or ccik_stochastic")

        e_alg_tot = float(ecore + e_alg_cas)
        e_fci_totals.append(e_fci_tot)
        e_alg_totals.append(e_alg_tot)
        print(f"R={R:5.2f} Å | CAS-FCI={e_fci_tot:+.10f} | {label}={e_alg_tot:+.10f} | Δ={e_alg_tot - e_fci_tot:+.3e}")

    plt.figure()
    plt.plot(Rs, np.asarray(e_fci_totals), marker="o", label="Exact CAS-FCI")
    plt.plot(Rs, np.asarray(e_alg_totals), marker="s", label=label)
    plt.xlabel("N–N bond length (Å)")
    plt.ylabel("Total energy (Ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("n2_legacy_scan.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    run_scan()
