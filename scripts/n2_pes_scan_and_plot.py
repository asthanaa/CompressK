"""N2 PES scan + plots for CCIK(thick) and FCIQMC-Krylov.

Creates 4 plots (2x2 figure):
  (1) PES: CAS-FCI vs CCIK(thick)
  (2) PES: CAS-FCI vs FCIQMC-Krylov
  (3) log(|ΔE|): CCIK(thick)
  (4) log(|ΔE|): FCIQMC-Krylov

Run from repo root:
  python scripts/n2_pes_scan_and_plot.py --config configs/config.toml --npoints 20

Matplotlib is an optional dependency (ccik[plot]).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Allow running without installing the package (uses src/ layout)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
import sys

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_fciqmc_krylov,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.config import (  # noqa: E402
    cas_spec_from_dict,
    ccik_thick_restart_params_from_dict,
    fciqmc_krylov_params_from_dict,
    load_config,
)


def make_n2_atom(R_ang: float) -> str:
    z = R_ang / 2.0
    return f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """


def _safe_log_err(delta: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    err = np.abs(np.asarray(delta, dtype=float))
    return np.maximum(err, float(floor))


def main() -> None:
    parser = argparse.ArgumentParser(description="N2 PES scan + 4-panel plot (CCIK thick vs FCIQMC-Krylov)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config (default: configs/config.toml if it exists)",
    )
    parser.add_argument("--Rmin", type=float, default=None, help="Scan start (Å). Overrides config.")
    parser.add_argument("--Rmax", type=float, default=None, help="Scan end (Å). Overrides config.")
    parser.add_argument(
        "--npoints",
        type=int,
        default=20,
        help="Number of scan points (inclusive endpoints). Overrides config.",
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "out_n2_pes"),
        help="Output prefix (writes <out>.csv and <out>.png)",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        local_default = _ROOT / "configs" / "config.toml"
        if local_default.exists():
            config_path = str(local_default)

    cfg = load_config(config_path)

    scan = cfg.get("scan", {})
    R_min = float(scan.get("R_min", 0.8)) if args.Rmin is None else float(args.Rmin)
    R_max = float(scan.get("R_max", 3.0)) if args.Rmax is None else float(args.Rmax)
    n_points = int(args.npoints)

    Rs = np.linspace(R_min, R_max, n_points)

    molcfg = cfg.get("molecule", {})
    basis = str(molcfg.get("basis", "6-31g"))
    unit = str(molcfg.get("unit", "Angstrom"))
    charge = int(molcfg.get("charge", 0))
    spin = int(molcfg.get("spin", 0))
    verbose = int(molcfg.get("verbose", 0))

    cas = cas_spec_from_dict(cfg.get("cas", {}))
    thick_params = ccik_thick_restart_params_from_dict(cfg.get("ccik_thick", {}))
    fciqmc_params = fciqmc_krylov_params_from_dict(cfg.get("fciqmckrylov", {}))

    out_prefix = Path(args.out)
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    rows = []

    print(f"Scanning N2: R in [{R_min:.3f}, {R_max:.3f}] Å with n_points={n_points}")
    print("Methods: CAS-FCI (ref), CCIK(thick), FCIQMC-Krylov")

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
        e_fci_tot = float(ecore + e_fci_cas)
        print(f"CAS-FCI done in {time.perf_counter()-t0:.2f}s | E={e_fci_tot:+.10f}", flush=True)

        t1 = time.perf_counter()
        e_thick_cas = ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params)
        e_thick_tot = float(ecore + e_thick_cas)
        d_thick = float(e_thick_tot - e_fci_tot)
        print(
            f"CCIK(thick) done in {time.perf_counter()-t1:.2f}s | E={e_thick_tot:+.10f} | Δ={d_thick:+.3e}",
            flush=True,
        )

        t2 = time.perf_counter()
        e_fqmc_cas = ccik_ground_energy_fciqmc_krylov(h1, eri8, ncas, nelec, params=fciqmc_params)
        e_fqmc_tot = float(ecore + e_fqmc_cas)
        d_fqmc = float(e_fqmc_tot - e_fci_tot)
        print(
            f"FCIQMC-Krylov done in {time.perf_counter()-t2:.2f}s | E={e_fqmc_tot:+.10f} | Δ={d_fqmc:+.3e}",
            flush=True,
        )

        rows.append((float(R), e_fci_tot, e_thick_tot, d_thick, e_fqmc_tot, d_fqmc))
        print(f"R={float(R):.4f} Å done in {time.perf_counter()-t_R0:.2f}s", flush=True)

    data = np.asarray(rows, dtype=float)
    header = "R_Ang,e_fci_total,ccik_thick_total,ccik_thick_delta,fciqmckrylov_total,fciqmckrylov_delta"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"\nWrote: {out_csv}")

    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        print("\nMatplotlib is required for plotting.")
        print("Install with: pip install 'ccik[plot]'  (or: pip install matplotlib)")
        raise

    # Publication-quality defaults (vector-friendly; consistent typography).
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    R = data[:, 0]
    e_fci = data[:, 1]
    e_thick = data[:, 2]
    d_thick = data[:, 3]
    e_fqmc = data[:, 4]
    d_fqmc = data[:, 5]

    # 2-column friendly width; height tuned for a 2x2 grid.
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)

    def panel_label(a, s: str) -> None:
        a.text(
            0.02,
            0.96,
            s,
            transform=a.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # (1) PES: FCI vs CCIK(thick)
    ax[0, 0].plot(R, e_fci, "k-", lw=2, label="CAS-FCI")
    ax[0, 0].plot(R, e_thick, "C0o-", ms=4, label="CCIK(thick)")
    panel_label(ax[0, 0], "(a)")
    ax[0, 0].set_xlabel(r"$R$ (\AA)")
    ax[0, 0].set_ylabel(r"$E_{\mathrm{tot}}$ (E$_h$)")
    ax[0, 0].legend(frameon=False)
    ax[0, 0].minorticks_on()
    ax[0, 0].tick_params(top=True, right=True)

    # (2) PES: FCI vs FCIQMC-Krylov
    ax[0, 1].plot(R, e_fci, "k-", lw=2, label="CAS-FCI")
    ax[0, 1].plot(R, e_fqmc, "C1o-", ms=4, label="FCIQMC-Krylov")
    panel_label(ax[0, 1], "(b)")
    ax[0, 1].set_xlabel(r"$R$ (\AA)")
    ax[0, 1].set_ylabel(r"$E_{\mathrm{tot}}$ (E$_h$)")
    ax[0, 1].legend(frameon=False)
    ax[0, 1].minorticks_on()
    ax[0, 1].tick_params(top=True, right=True)

    # (3) log error: CCIK(thick)
    ax[1, 0].plot(R, _safe_log_err(d_thick), "C0o-", ms=4)
    ax[1, 0].set_yscale("log")
    panel_label(ax[1, 0], "(c)")
    ax[1, 0].set_xlabel(r"$R$ (\AA)")
    ax[1, 0].set_ylabel(r"$|\Delta E|$ (E$_h$)")
    ax[1, 0].minorticks_on()
    ax[1, 0].tick_params(top=True, right=True)

    # (4) log error: FCIQMC-Krylov
    ax[1, 1].plot(R, _safe_log_err(d_fqmc), "C1o-", ms=4)
    ax[1, 1].set_yscale("log")
    panel_label(ax[1, 1], "(d)")
    ax[1, 1].set_xlabel(r"$R$ (\AA)")
    ax[1, 1].set_ylabel(r"$|\Delta E|$ (E$_h$)")
    ax[1, 1].minorticks_on()
    ax[1, 1].tick_params(top=True, right=True)

    # Save both raster (quick view) and vector (publication).
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
