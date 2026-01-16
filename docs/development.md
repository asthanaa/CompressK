# Development

## Python version

The project requires Python >= 3.11.

## Install

Editable install:

```bash
pip install -e .
```

For quantum chemistry workflows (PySCF):

```bash
pip install -e ".[quantum]"
```

For tests:

```bash
pip install -e ".[test]"
```

## Running tests

From repo root:

```bash
python -m pytest
```

Notes:

- The test suite is designed to work even if the package is not installed by adding `src/` to `sys.path` from `tests/conftest.py`.
- Some tests are expected to be skipped when optional dependencies are absent.

## Type checking

This repo includes `pyrightconfig.json` and a `[tool.pyright]` section in `pyproject.toml`.

## Optional dependencies

The package keeps heavy dependencies optional so `import ccik` works with only NumPy installed.

Optional extras (see `pyproject.toml`):

- `quantum`: PySCF
- `plot`: matplotlib
- `docs`: MkDocs Material (hosted documentation site)
- `test`: pytest

## Documentation site (MkDocs)

Install docs dependencies:

```bash
pip install -e ".[docs]"
```

Serve locally (live reload):

```bash
mkdocs serve
```

Build static site into `site/`:

```bash
mkdocs build
```

## Adding new drivers

The recommended pattern is:

1. Copy `scripts/n2_cas_scan.py` into a new script.
2. Replace the molecule builder (geometry / basis / charge / spin).
3. Keep using `ccik.config` + `ccik.params` to ensure configs remain consistent.

## Adding a new method

A new method typically requires:

- A params dataclass in `src/ccik/params.py`
- A parser in `src/ccik/config.py`
- A core implementation module in `src/ccik/`
- A driver integration in `scripts/n2_cas_scan.py` (or a new driver)
- A small test or smoke check in `tests/`
