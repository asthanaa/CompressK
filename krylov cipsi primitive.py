"""Compatibility wrapper (legacy entry point).

This file used to be the main runnable script. The historical version now lives at:
- `legacy/krylov_cipsi_primitive.py`

Prefer running the modern driver:
  python scripts/legacy/n2_scan_legacy.py
or the current package-based scan:
  python scripts/n2_cas_scan.py --config configs/config.toml
"""

from __future__ import annotations

from legacy.krylov_cipsi_primitive import run


if __name__ == "__main__":
    run()
