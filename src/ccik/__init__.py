"""Krylov-CIPSI (dense matvec) utilities.

This package exposes:
- `ccik_ground_energy_dense`: dense CCIK ground-state energy in a given Hamiltonian.
- `build_cas_hamiltonian_pyscf`: helper to build CAS Hamiltonians using PySCF.

The code is extracted from the original scripts in `legacy/` so you can
import and run sweeps from external driver scripts.

For a mapping from PDF equations to implementation lines, see:
- `krylov_cipsi.pdf` (theory)
- `src/ccik/core.py` (equation-by-equation comments)
"""

from .core import ccik_ground_energy_dense
from .cipsi import cipsi_dense_variational
from .config import (
    ai_selector_krylov_params_from_dict,
    cas_spec_from_dict,
    ccik_params_from_dict,
    ccik_thick_restart_params_from_dict,
    cipsi_params_from_dict,
    fciqmc_krylov_params_from_dict,
    load_config,
    load_default_toml,
    load_toml,
    run_method_from_dict,
    run_methods_from_dict,
)
from .params import CCIKParams, CCIKThickRestartParams, CIPSIParams, FCIQMCKrylovParams
from .pyscf_cas import (
    build_cas_hamiltonian_pyscf,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from .thick_restart import ccik_ground_energy_dense_thick_restart
from .fciqmc_krylov import ccik_ground_energy_fciqmc_krylov
from .ai_selector_krylov import (
    AISelectorKrylovParams,
    CIPSISelector,
    GNNSelector,
    Selector,
    ccik_ground_energy_ai_selector_krylov,
)
from .operator_nn import OperatorNN, OperatorNNConfig, select_topk_nodes, topk_recall
from .feature_mlp import FeatureMLP, FeatureMLPConfig, load_feature_mlp_checkpoint

__all__ = [
    "CCIKParams",
    "CCIKThickRestartParams",
    "CIPSIParams",
    "FCIQMCKrylovParams",
    "ccik_ground_energy_dense",
    "ccik_ground_energy_dense_thick_restart",
    "ccik_ground_energy_fciqmc_krylov",
    "ccik_ground_energy_ai_selector_krylov",
    "Selector",
    "CIPSISelector",
    "GNNSelector",
    "AISelectorKrylovParams",
    "OperatorNN",
    "OperatorNNConfig",
    "FeatureMLP",
    "FeatureMLPConfig",
    "load_feature_mlp_checkpoint",
    "select_topk_nodes",
    "topk_recall",
    "cipsi_dense_variational",
    "load_toml",
    "load_default_toml",
    "load_config",
    "ccik_params_from_dict",
    "ccik_thick_restart_params_from_dict",
    "cipsi_params_from_dict",
    "fciqmc_krylov_params_from_dict",
    "ai_selector_krylov_params_from_dict",
    "run_method_from_dict",
    "run_methods_from_dict",
    "cas_spec_from_dict",
    "build_cas_hamiltonian_pyscf",
    "exact_cas_fci_energy_pyscf",
    "make_mol_pyscf",
]
