"""Compatibility wrapper.

This file used to be the main runnable script. It is kept as a thin wrapper so
existing notes/commands still work after the repo was reorganized as `ccik`.

Prefer running:
  python scripts/legacy/n2_scan_legacy.py
"""

from __future__ import annotations

from scripts.legacy.n2_scan_legacy import run_scan


if __name__ == "__main__":
    run_scan()
