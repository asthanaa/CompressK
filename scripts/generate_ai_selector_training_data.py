"""Generate training data for the AI-selector Krylov GNN/MLP backend.

This script runs the AI-selector Krylov routine (with exact dense matvecs) and
records *candidate-pool* examples:
  - features: the same 5 engineered features used by GNNSelector fallback
  - target: log(eps + |(H q_k)[det]|)

Output is a NumPy .npz with arrays:
  - X: (M, 5)
  - y: (M,)

Run from repo root:
  python scripts/generate_ai_selector_training_data.py --config configs/config.toml --out data/n2_train.npz

You can optionally restrict to a single geometry:
  python scripts/generate_ai_selector_training_data.py --R 1.10

Notes
- Requires PySCF.
- Does not require torch/torch_geometric (data is saved as NumPy).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    ai_selector_krylov_params_from_dict,
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_ai_selector_krylov,
    make_mol_pyscf,
)
from ccik.config import cas_spec_from_dict, load_config  # noqa: E402


def make_n2_atom(R_ang: float) -> str:
    z = R_ang / 2.0
    return f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """


def _features_for(det, *, E_k: float, hdiag: np.ndarray, cand_parent_w: dict, iteration: int, eps_denom: float):
    ia, ib = det
    w = float(cand_parent_w.get(det, 0.0))
    denom = float(abs(float(E_k) - float(hdiag[int(ia), int(ib)])))
    return np.array(
        [
            w,
            np.log(w + 1e-30),
            denom,
            1.0 / (denom + float(eps_denom)),
            float(iteration),
        ],
        dtype=np.float32,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate AI-selector training data (candidate pool)")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=str(_ROOT / "data" / "ai_selector_train.npz"))
    p.add_argument("--R", type=float, default=None, help="If set, only generate data at one bond length (Å).")
    p.add_argument("--Rmin", type=float, default=None)
    p.add_argument("--Rmax", type=float, default=None)
    p.add_argument("--npoints", type=int, default=10)
    p.add_argument("--target-eps", type=float, default=1e-12, help="eps in log(eps+|v|)")
    args = p.parse_args()

    config_path = args.config
    if config_path is None:
        local_default = _ROOT / "configs" / "config.toml"
        if local_default.exists():
            config_path = str(local_default)

    cfg = load_config(config_path)

    scan = cfg.get("scan", {})
    if args.R is not None:
        Rs = np.array([float(args.R)], dtype=float)
    else:
        R_min = float(scan.get("R_min", 1.0)) if args.Rmin is None else float(args.Rmin)
        R_max = float(scan.get("R_max", 3.0)) if args.Rmax is None else float(args.Rmax)
        Rs = np.linspace(R_min, R_max, int(args.npoints))

    molcfg = cfg.get("molecule", {})
    basis = str(molcfg.get("basis", "6-31g"))
    unit = str(molcfg.get("unit", "Angstrom"))
    charge = int(molcfg.get("charge", 0))
    spin = int(molcfg.get("spin", 0))
    verbose = int(molcfg.get("verbose", 0))

    cas = cas_spec_from_dict(cfg.get("cas", {}))
    ai_params = ai_selector_krylov_params_from_dict(cfg.get("ai_selector_krylov", {}))
    eps_denom = float(cfg.get("ai_selector_krylov", {}).get("eps_denom", 1e-12))

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    def collector(**kw):
        k = int(kw["k"])
        E_k = float(kw["E_k"])
        hdiag = kw["hdiag"]
        v_full = kw["v_full"]
        supp_k = kw["supp_k"]
        cand_ids = kw["cand_ids"]
        cand_parent_w = kw["cand_parent_w"]

        feats = []
        tgts = []
        for det in cand_ids:
            ia, ib = det
            if supp_k[int(ia), int(ib)]:
                continue
            feats.append(_features_for(det, E_k=E_k, hdiag=hdiag, cand_parent_w=cand_parent_w, iteration=k, eps_denom=eps_denom))
            tgts.append(np.log(float(args.target_eps) + abs(float(v_full[int(ia), int(ib)]))))

        if feats:
            X_rows.append(np.vstack(feats))
            y_rows.append(np.asarray(tgts, dtype=np.float32))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for R in Rs:
        mol = make_mol_pyscf(
            atom=make_n2_atom(float(R)),
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            verbose=verbose,
        )

        h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

        # Run once, purely to trigger collector callbacks.
        _ = ccik_ground_energy_ai_selector_krylov(
            h1,
            eri8,
            ncas,
            nelec,
            selector_backend="cipsi",  # selection backend doesn't matter for labels; we always compute v_full
            params=ai_params,
            iteration_collector=collector,
        )

    if not X_rows:
        raise RuntimeError("No training examples collected. Try increasing --npoints or ai_selector_krylov.n_walkers.")

    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)

    np.savez_compressed(out_path, X=X, y=y)
    print(f"Wrote: {out_path}  (X={X.shape}, y={y.shape})")


if __name__ == "__main__":
    main()
