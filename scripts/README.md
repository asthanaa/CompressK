# scripts/

This folder contains runnable driver scripts that use the importable package in `src/ccik/`.

Below is a step-by-step example showing how to run **FCIQMC-Krylov** and **CCIK(thick restart)** on a small N2 example input.

## 1) Install prerequisites

- Python >= 3.11
- PySCF (required for building CAS Hamiltonians)

From the repo root, install the package (editable) with PySCF:

```bash
pip install -e ".[quantum]"
```

(Optional) to run tests:

```bash
pip install -e ".[quantum,test]"
```

## 2) Use the provided sample N2 input

A small, fast N2 sample config is stored at:

- `tests/n2_sample.toml`

It uses:
- basis: `sto-3g`
- CAS: (6e,6o) with `ncore=4`
- a **single bond length** point (`R_min=R_max=1.10 Å`)

The N2 geometry itself is generated inside the driver (`scripts/n2_cas_scan.py`) using the bond length(s) from the `[scan]` section.

## 3) Run FCIQMC-Krylov (stochastic Krylov selection)

In `tests/n2_sample.toml`, ensure:

```toml
[run]
method = "fciqmckrylov"
```

Then run:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

What you should see:
- For each geometry point, it runs CAS-FCI (reference) and then `FCIQMC-Krylov`.
- It writes a CSV to the repo root:

```text
out_n2_cas_scan.csv
```

## 4) Run CCIK(thick restart)

In `tests/n2_sample.toml`, change:

```toml
[run]
method = "ccik_thick"
```

Then run the same driver:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

This runs the dense compressed-Krylov method but with **thick restart** (it reuses a few lowest-energy Ritz vectors as starting vectors for the next cycle).

## 5) (Optional) Run both methods in one sweep

Instead of `method = ...`, you can use:

```toml
[run]
methods = ["fciqmckrylov", "ccik_thick"]
```

Then run:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

## Notes

- If you increase the active space or basis, CAS-FCI can become expensive quickly.
- All algorithm knobs live in the TOML sections:
  - `[fciqmckrylov]` for FCIQMC-Krylov
  - `[ccik_thick]` for thick-restart CCIK
- The method names accepted by the driver are normalized; the canonical ones are:
  - `fciqmckrylov`
  - `ccik_thick`
