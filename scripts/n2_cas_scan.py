"""Example external driver: N2 bond scan using the `ccik` package.

Run from repo root:
  python scripts/n2_cas_scan.py

This script intentionally lives outside the package so you can copy/modify it
for other systems and data collection.

PDF mapping:
- Algorithm theory is in `krylov_cipsi.pdf`
- Equation-level mapping comments are in `src/ccik/core.py`
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time

import numpy as np

# Allow running without installing the package (uses src/ layout)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
import sys

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    ai_selector_krylov_params_from_dict,
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_ai_selector_krylov,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_fciqmc_krylov,
    cipsi_dense_variational,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.pyscf_cas import CASSpec  # noqa: E402
from ccik.config import (  # noqa: E402
    cas_spec_from_dict,
    ccik_params_from_dict,
    ccik_thick_restart_params_from_dict,
    cipsi_params_from_dict,
    fciqmc_krylov_params_from_dict,
    load_config,
    run_method_from_dict,
    run_methods_from_dict,
)

from n2_utils import make_n2_atom


def _canonical_method(method: str) -> str:
    m = method.strip()
    if m == "ccik":
        return "ccik"
    if m in ("ccik_thick", "ccik_thick_restart", "reuse_ritz", "reuseRitz"):
        return "ccik_thick"
    if m in ("cipsi", "cipsi_var", "cipsi_variational"):
        return "cipsi_var"
    if m in ("fciqmckrylov", "fciqmc_krylov", "fciqmc-krylov"):
        return "fciqmckrylov"
    if m in ("ai_selector_krylov", "ai-selector-krylov", "ai_krylov", "ai"):
        return "ai_selector_krylov"
    return m
def main() -> None:
    parser = argparse.ArgumentParser(description="N2 CAS scan using dense CCIK")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config (default: packaged ccik/defaults/config.toml)",
    )
    parser.add_argument(
        "--gnn-model",
        default=None,
        help="Optional torch checkpoint path for AI-selector Krylov (GNN backend).",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        local_default = _ROOT / "configs" / "config.toml"
        if local_default.exists():
            config_path = str(local_default)

    cfg = load_config(config_path)

    run_cfg = cfg.get("run", {})
    methods = [_canonical_method(m) for m in run_methods_from_dict(run_cfg)]

    scan = cfg.get("scan", {})
    Rs = np.linspace(
        float(scan.get("R_min", 2.0)),
        float(scan.get("R_max", 3.0)),
        int(scan.get("n_points", 4)),
    )

    molcfg = cfg.get("molecule", {})
    basis = str(molcfg.get("basis", "6-31g"))
    unit = str(molcfg.get("unit", "Angstrom"))
    charge = int(molcfg.get("charge", 0))
    spin = int(molcfg.get("spin", 0))
    verbose = int(molcfg.get("verbose", 0))

    cas = cas_spec_from_dict(cfg.get("cas", {}))
    ccik_params = ccik_params_from_dict(cfg.get("ccik", {}))
    thick_params = ccik_thick_restart_params_from_dict(cfg.get("ccik_thick", {}))
    cipsi_params = cipsi_params_from_dict(cfg.get("cipsi", {}))
    fciqmc_krylov_params = fciqmc_krylov_params_from_dict(cfg.get("fciqmckrylov", {}))
    ai_params = ai_selector_krylov_params_from_dict(cfg.get("ai_selector_krylov", {}))

    gnn_model = None
    model_path = args.gnn_model
    if model_path is None:
        model_path = cfg.get("ai_selector_krylov", {}).get("gnn_model_path")
    if model_path:
        from ccik.feature_mlp import load_feature_mlp_checkpoint

        gnn_model = load_feature_mlp_checkpoint(model_path, map_location="cpu")
        print(f"Loaded GNN model checkpoint: {model_path}")

    # Prepare header + row layout
    multi = len(methods) > 1
    if multi:
        header_cols = ["R_Ang", "e_fci_total"]
        for m in methods:
            header_cols += [f"{m}_total", f"{m}_delta"]
        header = "R_Ang,e_fci_total," + ",".join(header_cols[2:])
        rows = []
    else:
        header = f"method={methods[0]}\nR_Ang,e_fci_total,e_alg_total,delta"
        rows = []

    for R in Rs:
        t_R0 = time.perf_counter()
        print(f"\n=== R={float(R):.4f} Å ===", flush=True)
        mol = make_mol_pyscf(
            atom=make_n2_atom(float(R)),
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            verbose=verbose,
        )

        # Helpful sanity check when modifying (ncas, nelecas, ncore).
        # For a closed-shell singlet, total electrons must satisfy: nelectron = 2*ncore + nelecas.
        # If you change nelecas (e.g. to 12), you must adjust ncore accordingly.
        if spin == 0:
            expected_ne = 2 * int(cas.ncore) + int(cas.nelecas)
            if int(mol.nelectron) != int(expected_ne):
                raise ValueError(
                    "CAS electron mismatch for singlet: expected mol.nelectron == 2*ncore + nelecas. "
                    f"Got mol.nelectron={mol.nelectron}, ncore={cas.ncore}, nelecas={cas.nelecas} (2*ncore+nelecas={expected_ne})."
                )
        h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

        t0 = time.perf_counter()
        e_fci_cas = exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec)
        e_fci_tot = ecore + e_fci_cas
        print(f"CAS-FCI done in {time.perf_counter()-t0:.2f}s | E={e_fci_tot:+.10f}", flush=True)

        if multi:
            row = [float(R), float(e_fci_tot)]
            msg_parts = [f"R={R:5.2f} Å", f"CAS-FCI={e_fci_tot:+.10f}"]
            for method in methods:
                t_method0 = time.perf_counter()
                if method == "ccik":
                    st: dict[str, int] = {}
                    print("  -> running CCIK...", flush=True)
                    e_alg_cas = ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=ccik_params, stats=st)
                    label = "CCIK"
                    ndet_info = f"nparam={st.get('ndet_sum', -1)}"
                elif method == "ccik_thick":
                    st = {}
                    print("  -> running CCIK(thick)...", flush=True)
                    e_alg_cas = ccik_ground_energy_dense_thick_restart(
                        h1,
                        eri8,
                        ncas,
                        nelec,
                        params=thick_params,
                        stats=st,
                    )
                    label = "CCIK(thick)"
                    ndet_info = f"nparam={st.get('ndet_sum', -1)}"
                elif method == "cipsi_var":
                    st = {}
                    print("  -> running CIPSI-var...", flush=True)
                    e_alg_cas = cipsi_dense_variational(h1, eri8, ncas, nelec, params=cipsi_params, stats=st)
                    label = "CIPSI-var"
                    ndet_info = f"nparam={st.get('ndet', -1)}"
                elif method == "fciqmckrylov":
                    st = {}
                    print("  -> running FCIQMC-Krylov...", flush=True)
                    e_alg_cas = ccik_ground_energy_fciqmc_krylov(
                        h1,
                        eri8,
                        ncas,
                        nelec,
                        params=fciqmc_krylov_params,
                        stats=st,
                    )
                    label = "FCIQMC-Krylov"
                    ndet_info = f"nparam={st.get('ndet_sum', -1)}"
                elif method == "ai_selector_krylov":
                    st = {}
                    print("  -> running AI-selector Krylov (GNN backend)...", flush=True)
                    e_alg_cas = ccik_ground_energy_ai_selector_krylov(
                        h1,
                        eri8,
                        ncas,
                        nelec,
                        selector_backend="gnn",
                        gnn_model=gnn_model,
                        params=ai_params,
                        stats=st,
                    )
                    label = "AI-selector"
                    ndet_info = f"m_eff={st.get('m_eff', -1)}"
                else:
                    raise ValueError(
                        "Unknown run.methods entry={!r}. Expected: ccik, ccik_thick, cipsi_var, fciqmckrylov, ai_selector_krylov".format(
                            method
                        )
                    )

                dt_method = time.perf_counter() - t_method0

                e_alg_tot = ecore + float(e_alg_cas)
                delta = e_alg_tot - e_fci_tot
                row += [float(e_alg_tot), float(delta)]
                msg_parts.append(f"{label}={e_alg_tot:+.10f} (Δ={delta:+.3e}, {ndet_info}, t={dt_method:.2f}s)")

            rows.append(row)
            print(" | ".join(msg_parts))
        else:
            method = methods[0]
            t_method0 = time.perf_counter()
            if method == "ccik":
                st: dict[str, int] = {}
                print("  -> running CCIK...", flush=True)
                e_alg_cas = ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=ccik_params, stats=st)
                label = "CCIK"
                ndet_info = f"nparam={st.get('ndet_sum', -1)}"
            elif method == "ccik_thick":
                st = {}
                print("  -> running CCIK(thick)...", flush=True)
                e_alg_cas = ccik_ground_energy_dense_thick_restart(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=thick_params,
                    stats=st,
                )
                label = "CCIK(thick)"
                ndet_info = f"nparam={st.get('ndet_sum', -1)}"
            elif method == "cipsi_var":
                st = {}
                print("  -> running CIPSI-var...", flush=True)
                e_alg_cas = cipsi_dense_variational(h1, eri8, ncas, nelec, params=cipsi_params, stats=st)
                label = "CIPSI-var"
                ndet_info = f"nparam={st.get('ndet', -1)}"
            elif method == "fciqmckrylov":
                st = {}
                print("  -> running FCIQMC-Krylov...", flush=True)
                e_alg_cas = ccik_ground_energy_fciqmc_krylov(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    params=fciqmc_krylov_params,
                    stats=st,
                )
                label = "FCIQMC-Krylov"
                ndet_info = f"nparam={st.get('ndet_sum', -1)}"
            elif method == "ai_selector_krylov":
                st = {}
                print("  -> running AI-selector Krylov (GNN backend)...", flush=True)
                e_alg_cas = ccik_ground_energy_ai_selector_krylov(
                    h1,
                    eri8,
                    ncas,
                    nelec,
                    selector_backend="gnn",
                    params=ai_params,
                    stats=st,
                )
                label = "AI-selector"
                ndet_info = f"m_eff={st.get('m_eff', -1)}"
            else:
                raise ValueError(
                    "Unknown run.method={!r}. Expected one of: ccik, ccik_thick, cipsi_var, fciqmckrylov, ai_selector_krylov".format(
                        method
                    )
                )

            dt_method = time.perf_counter() - t_method0

            e_alg_tot = ecore + float(e_alg_cas)
            rows.append((float(R), e_fci_tot, e_alg_tot, e_alg_tot - e_fci_tot))
            print(
                f"R={R:5.2f} Å | CAS-FCI={e_fci_tot:+.10f} | {label}={e_alg_tot:+.10f} | Δ={e_alg_tot-e_fci_tot:+.3e} | {ndet_info} | t={dt_method:.2f}s"
            )

        print(f"R={float(R):.4f} Å done in {time.perf_counter()-t_R0:.2f}s", flush=True)

    rows = np.asarray(rows)
    out = _ROOT / "out_n2_cas_scan.csv"
    np.savetxt(out, rows, delimiter=",", header=header, comments="")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
