from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure `src/` layout imports work when running tests without installing.

    This keeps `pytest` usable in fresh environments (e.g., CI) without requiring
    `pip install -e .`.
    """

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
