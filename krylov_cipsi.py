"""Compatibility wrapper (legacy entry point).

The original monolithic script has been moved to `legacy/krylov_cipsi.py`.
Prefer using the importable package in `src/ccik/` and the drivers in `scripts/`.

Example modern run:
  python scripts/n2_cas_scan.py --config configs/config.toml
"""

from __future__ import annotations

from legacy.krylov_cipsi import run_scan


if __name__ == "__main__":
    run_scan()
