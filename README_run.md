
## Package-style usage

Reusable code lives in `src/ccik/`.

See `src/ccik/README.md` for an algorithm description mapped to `krylov_cipsi.pdf`.

You can either:

1) run scripts that add `src/` to `sys.path` (see `scripts/n2_cas_scan.py`), or
2) install the package in editable mode (recommended for long-term use).

### Run an external sweep script (no install)

From repo root:

```bash
python scripts/n2_cas_scan.py
```

This writes `out_n2_cas_scan.csv`.

### Import from your own driver

Create a new script anywhere and do:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[0]  # adjust as needed
sys.path.insert(0, str(ROOT / "src"))

from ccik import ccik_ground_energy_dense, CCIKParams
```

Then build integrals for your system (e.g. via `ccik.pyscf_cas`).

