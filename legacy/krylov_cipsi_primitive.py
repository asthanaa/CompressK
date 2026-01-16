"""Legacy compatibility wrapper.

This script originally lived at repo root as `krylov cipsi primitive.py`.
It is kept here for history; the repo root now contains a tiny shim that imports
and runs this module.

Prefer running:
  python scripts/legacy/n2_scan_legacy.py
"""

from __future__ import annotations


def run() -> None:
    from scripts.legacy.n2_scan_legacy import run_scan

    run_scan()


if __name__ == "__main__":
    run()
