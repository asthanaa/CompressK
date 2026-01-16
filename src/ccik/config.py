from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .ai_selector_krylov import AISelectorKrylovParams
from .params import CCIKParams, CCIKThickRestartParams, CIPSIParams, FCIQMCKrylovParams
from .pyscf_cas import CASSpec


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file using the stdlib (Python 3.11+: tomllib)."""
    import tomllib

    p = Path(path)
    with p.open("rb") as f:
        return tomllib.load(f)


def load_default_toml() -> dict[str, Any]:
    """Load the packaged default config TOML.

    This allows driver scripts to run without requiring an external config file.
    """

    import tomllib
    from importlib import resources

    data = resources.files("ccik.defaults").joinpath("config.toml")
    with data.open("rb") as f:
        return tomllib.load(f)


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load config from an explicit path, or fall back to the packaged default."""

    if path is None:
        return load_default_toml()
    return load_toml(path)


def ccik_params_from_dict(d: dict[str, Any]) -> CCIKParams:
    """Create CCIKParams from a dict (typically parsed from TOML)."""
    base = CCIKParams()
    out = asdict(base)

    if "m" in d:
        out["m"] = int(d["m"])
    if "nadd" in d:
        out["nadd"] = int(d["nadd"])
    if "nkeep" in d:
        out["nkeep"] = int(d["nkeep"])
    if "Kv" in d:
        out["Kv"] = int(d["Kv"])
    if "verbose" in d:
        out["verbose"] = bool(d["verbose"])

    if "orth_tol" in d:
        out["orth_tol"] = float(d["orth_tol"])

    return CCIKParams(**out)


def ccik_thick_restart_params_from_dict(d: dict[str, Any]) -> CCIKThickRestartParams:
    """Create CCIKThickRestartParams from a dict (typically parsed from TOML)."""
    base = CCIKThickRestartParams()
    out = asdict(base)

    for k in ("m_cycle", "ncycles", "nroot"):
        if k in d:
            out[k] = int(d[k])
    if "tol" in d:
        out["tol"] = float(d["tol"])

    for k in ("nadd", "nkeep", "Kv"):
        if k in d:
            out[k] = int(d[k])
    if "verbose" in d:
        out["verbose"] = bool(d["verbose"])

    if "orth_tol" in d:
        out["orth_tol"] = float(d["orth_tol"])

    return CCIKThickRestartParams(**out)


def cipsi_params_from_dict(d: dict[str, Any]) -> CIPSIParams:
    """Create CIPSIParams from a dict (typically parsed from TOML)."""
    base = CIPSIParams()
    out = asdict(base)

    for k in ("niter", "nadd", "ndet_max", "Kv"):
        if k in d:
            out[k] = int(d[k])
    if "davidson_tol" in d:
        out["davidson_tol"] = float(d["davidson_tol"])
    if "verbose" in d:
        out["verbose"] = bool(d["verbose"])

    return CIPSIParams(**out)


def fciqmc_krylov_params_from_dict(d: dict[str, Any]) -> FCIQMCKrylovParams:
    """Create FCIQMCKrylovParams from a dict (typically parsed from TOML)."""
    base = FCIQMCKrylovParams()
    out = asdict(base)

    for k in ("m", "nadd", "nkeep", "Kv", "n_walkers"):
        if k in d:
            out[k] = int(d[k])
    if "seed" in d:
        out["seed"] = None if d["seed"] is None else int(d["seed"])

    for k in ("parent_power", "p_double", "mixed_double_weight", "eps_denom"):
        if k in d:
            out[k] = float(d[k])

    if "verbose" in d:
        out["verbose"] = bool(d["verbose"])
    if "orth_tol" in d:
        out["orth_tol"] = float(d["orth_tol"])

    return FCIQMCKrylovParams(**out)


def ai_selector_krylov_params_from_dict(d: dict[str, Any]) -> AISelectorKrylovParams:
    """Create AISelectorKrylovParams from a dict (typically parsed from TOML)."""

    base = AISelectorKrylovParams()
    out = asdict(base)

    for k in ("m", "nadd", "nkeep", "Kv", "n_walkers"):
        if k in d:
            out[k] = int(d[k])
    if "seed" in d:
        out["seed"] = None if d["seed"] is None else int(d["seed"])

    for k in ("parent_power", "p_double", "mixed_double_weight", "eps_denom"):
        if k in d:
            out[k] = float(d[k])

    if "orth_tol" in d:
        out["orth_tol"] = float(d["orth_tol"])
    if "verbose" in d:
        out["verbose"] = bool(d["verbose"])

    return AISelectorKrylovParams(**out)


def run_method_from_dict(d: dict[str, Any]) -> str:
    """Return the requested method name (defaults to 'ccik')."""
    method = str(d.get("method", "ccik")).strip()
    return method


def run_methods_from_dict(d: dict[str, Any]) -> list[str]:
    """Return a list of requested methods.

    Supported TOML shapes:
    - method = "ccik"              -> ["ccik"]
    - methods = ["ccik", "ccik_thick"]
    If neither is provided, defaults to ["ccik"].
    """

    if "methods" in d and d["methods"] is not None:
        raw = d["methods"]
        if not isinstance(raw, (list, tuple)):
            raise TypeError("run.methods must be a list of strings")
        out = [str(x).strip() for x in raw]
        out = [x for x in out if x]
        return out if out else ["ccik"]

    # Convenience: allow `method = [..]` as well.
    if "method" in d and isinstance(d["method"], (list, tuple)):
        raw = d["method"]
        out = [str(x).strip() for x in raw]
        out = [x for x in out if x]
        return out if out else ["ccik"]

    return [run_method_from_dict(d)]


def cas_spec_from_dict(d: dict[str, Any]) -> CASSpec:
    """Create CASSpec from a dict (typically parsed from TOML)."""
    return CASSpec(
        ncas=int(_require(d, "ncas")),
        nelecas=int(_require(d, "nelecas")),
        ncore=int(d.get("ncore", 0)),
    )


def as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion for dataclasses to plain dicts (debug/printing)."""
    try:
        return asdict(obj)
    except Exception:
        return dict(obj)
