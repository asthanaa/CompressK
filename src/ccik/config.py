from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams
from .pyscf_cas import CASSpec


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file using the Python 3.11 stdlib parser."""

    import tomllib

    p = Path(path)
    with p.open("rb") as f:
        return tomllib.load(f)


def load_default_toml() -> dict[str, Any]:
    """Load the packaged default configuration."""

    import tomllib
    from importlib import resources

    data = resources.files("ccik.defaults").joinpath("config.toml")
    with data.open("rb") as f:
        return tomllib.load(f)


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load an explicit config file or the packaged default."""

    if path is None:
        return load_default_toml()
    return load_toml(path)


def ccik_params_from_dict(d: dict[str, Any]) -> CCIKParams:
    """Create ``CCIKParams`` from a parsed TOML section."""

    base = asdict(CCIKParams())
    for key in ("m", "nadd", "nkeep", "Kv"):
        if key in d:
            base[key] = int(d[key])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    return CCIKParams(**base)


def ccik_thick_restart_params_from_dict(d: dict[str, Any]) -> CCIKThickRestartParams:
    """Create ``CCIKThickRestartParams`` from a parsed TOML section."""

    base = asdict(CCIKThickRestartParams())
    for key in ("m_cycle", "ncycles", "nroot", "nadd", "nkeep", "Kv"):
        if key in d:
            base[key] = int(d[key])
    if "tol" in d:
        base["tol"] = float(d["tol"])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    return CCIKThickRestartParams(**base)


def ccik_stochastic_params_from_dict(d: dict[str, Any]) -> CCIKStochasticParams:
    """Create ``CCIKStochasticParams`` from a parsed TOML section."""

    base = asdict(CCIKStochasticParams())
    for key in ("m", "nadd", "nkeep", "Kv", "n_walkers"):
        if key in d:
            base[key] = int(d[key])
    if "seed" in d:
        base["seed"] = None if d["seed"] is None else int(d["seed"])
    for key in ("parent_power", "p_double", "mixed_double_weight", "eps_denom"):
        if key in d:
            base[key] = float(d[key])
    if "verbose" in d:
        base["verbose"] = bool(d["verbose"])
    if "orth_tol" in d:
        base["orth_tol"] = float(d["orth_tol"])
    return CCIKStochasticParams(**base)


def run_method_from_dict(d: dict[str, Any]) -> str:
    """Return the requested method name, defaulting to ``ccik``."""

    return str(d.get("method", "ccik")).strip()


def run_methods_from_dict(d: dict[str, Any]) -> list[str]:
    """Return the requested method list.

    Supported shapes:
    - ``method = "ccik"``
    - ``methods = ["ccik", "ccik_thick"]``
    """

    if "methods" in d and d["methods"] is not None:
        raw = d["methods"]
        if not isinstance(raw, (list, tuple)):
            raise TypeError("run.methods must be a list of strings")
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else ["ccik"]

    if "method" in d and isinstance(d["method"], (list, tuple)):
        raw = d["method"]
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else ["ccik"]

    return [run_method_from_dict(d)]


def cas_spec_from_dict(d: dict[str, Any]) -> CASSpec:
    """Create ``CASSpec`` from a parsed TOML section."""

    return CASSpec(
        ncas=int(_require(d, "ncas")),
        nelecas=int(_require(d, "nelecas")),
        ncore=int(d.get("ncore", 0)),
    )


def as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion for dataclasses to plain dictionaries."""

    try:
        return asdict(obj)
    except Exception:
        return dict(obj)
