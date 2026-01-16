# Legacy scripts

This repo used to be organized as a set of one-off scripts at the repository root.

Those scripts have been moved into:

- `legacy/`

The repo root still contains small compatibility wrappers (shims) so that older commands continue to work:

- `krylov_cipsi.py`
- `krylov cipsi primitive.py`

New work should use:

- the importable package `src/ccik/`
- the current drivers under `scripts/`

## Why keep legacy code?

- It can be useful for reproducing older experiments.
- It provides historical context for how the package evolved.
- It avoids breaking downstream users that have local tooling expecting those filenames.

## Where to look instead

- Run instructions: `scripts/README.md` and [quickstart.md](quickstart.md)
- Algorithm mapping to the PDF: `src/ccik/README.md`
- Config schema: [configuration.md](configuration.md)
