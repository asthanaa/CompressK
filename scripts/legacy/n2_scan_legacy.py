"""
N2 / 6-31G bond breaking in CAS(10e,10o):
Exact CAS-FCI vs DENSE CCIK (full-Hilbert-space matvec via PySCF direct_spin1)

This is the "previous working" style:
  - CCIK uses full CI tensors in the CAS space (na x nb)
  - matvec uses direct_spin1.contract_2e (fast, in compiled code)
  - generalized Rayleigh–Ritz to fix non-orthogonality after compression
  - scans bond lengths and plots FCI vs CCIK

Active space:
  - ncore = 2 (freeze N 1s on each N)
  - nelecas = 10, ncas = 10
Orbitals:
  - RHF canonical orbitals (CASCI effective Hamiltonian)

Requires: pyscf, numpy, matplotlib
"""

# PDF mapping:
# - Theory/notation: `krylov_cipsi.pdf`
# - Equation-by-equation implementation comments: `src/ccik/core.py`

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Allow running without installing the package (uses src/ layout)
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (
  build_cas_hamiltonian_pyscf,
  ccik_ground_energy_dense,
  ccik_ground_energy_dense_thick_restart,
  cipsi_dense_variational,
  exact_cas_fci_energy_pyscf,
  make_mol_pyscf,
)
from ccik.pyscf_cas import CASSpec
from ccik.config import (
  cas_spec_from_dict,
  ccik_params_from_dict,
  ccik_thick_restart_params_from_dict,
  cipsi_params_from_dict,
  load_config,
  run_method_from_dict,
)


def make_n2_mol(R_ang, basis="6-31g"):
    z = R_ang / 2.0
    atom = f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """
    return make_mol_pyscf(atom=atom, basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)


# -----------------------------
# Bond scan + plot
# -----------------------------


def run_scan():
  parser = argparse.ArgumentParser(description="Legacy N2 scan (dense CCIK)")
  parser.add_argument(
    "--config",
    default=None,
    help="Path to TOML config (default: packaged ccik/defaults/config.toml)",
  )
  args, _ = parser.parse_known_args()

  config_path = args.config
  if config_path is None:
    local_default = _ROOT / "configs" / "config.toml"
    if local_default.exists():
      config_path = str(local_default)

  cfg = load_config(config_path)
  scan = cfg.get("scan", {})
  Rs = np.linspace(
    float(scan.get("R_min", 0.8)),
    float(scan.get("R_max", 3.0)),
    int(scan.get("n_points", 16)),
  )

  molcfg = cfg.get("molecule", {})
  basis = str(molcfg.get("basis", "6-31g"))

  run_cfg = cfg.get("run", {})
  method = run_method_from_dict(run_cfg)

  cas = cas_spec_from_dict(cfg.get("cas", {}))
  ccik_params = ccik_params_from_dict(cfg.get("ccik", {}))
  thick_params = ccik_thick_restart_params_from_dict(cfg.get("ccik_thick", {}))
  cipsi_params = cipsi_params_from_dict(cfg.get("cipsi", {}))

  E_fci_tot = []
  E_ccik_tot = []

  for R in Rs:
    mol = make_n2_mol(R, basis=basis)

    h1eff, eri8, ecore, ncas_, nelec_cas = build_cas_hamiltonian_pyscf(mol, cas=cas)

    # exact CAS-FCI
    e_fci_cas = exact_cas_fci_energy_pyscf(h1eff, eri8, ncas_, nelec_cas)
    e_fci_tot = ecore + e_fci_cas

    if method == "ccik":
      e_alg_cas = ccik_ground_energy_dense(h1eff, eri8, ncas_, nelec_cas, params=ccik_params)
      label = "CCIK"
    elif method in ("ccik_thick", "ccik_thick_restart", "reuse_ritz", "reuseRitz"):
      e_alg_cas = ccik_ground_energy_dense_thick_restart(
        h1eff,
        eri8,
        ncas_,
        nelec_cas,
        params=thick_params,
      )
      label = "CCIK(thick)"
    elif method in ("cipsi", "cipsi_var", "cipsi_variational"):
      e_alg_cas = cipsi_dense_variational(h1eff, eri8, ncas_, nelec_cas, params=cipsi_params)
      label = "CIPSI-var"
    else:
      raise ValueError(
        f"Unknown run.method={method!r}. Expected one of: ccik, ccik_thick, cipsi_var"
      )

    e_alg_tot = ecore + float(e_alg_cas)

    E_fci_tot.append(e_fci_tot)
    E_ccik_tot.append(e_alg_tot)

    print(
      f"R={R:5.2f} Å | CAS-FCI={e_fci_tot:+.10f} | {label}={e_alg_tot:+.10f} | Δ={e_alg_tot-e_fci_tot:+.3e}"
    )

  Rs = np.array(Rs)
  E_fci_tot = np.array(E_fci_tot)
  E_ccik_tot = np.array(E_ccik_tot)

  plt.figure()
  plt.plot(Rs, E_fci_tot, marker="o", label="Exact CAS(10,10)-FCI / 6-31G")
  plt.plot(Rs, E_ccik_tot, marker="s", label="Dense CCIK (full-CAS matvec)")
  plt.xlabel("N–N bond length (Å)")
  plt.ylabel("Total energy (Ha)")
  plt.legend()
  plt.tight_layout()
  plt.savefig("n2_631g_cas10_10_fci_vs_dense_ccik.png", dpi=200)
  plt.show()

  print("\nSaved plot: n2_631g_cas10_10_fci_vs_dense_ccik.png")


if __name__ == "__main__":
    run_scan()
