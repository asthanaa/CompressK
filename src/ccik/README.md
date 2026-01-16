# ccik

Dense (full-CAS) implementation of the **Compressed Krylov Subspace Method with Selected-Configuration Matvecs** (called **CCIK** in the PDF).

This package is extracted from the original one-off runner and is meant to be imported from external sweep scripts so you can gather data for multiple systems.

## What this implements (mapped to the PDF)

This code follows the structure in `krylov_cipsi.pdf` (3 pages) titled:

> “Notes on Compressed Krylov Subspace Method with Selected-Configuration Matvecs”

The key objects in the PDF are:

- Krylov space definition (PDF **Sec. II.A**, Eq. (1)):

  In plain text:

  ```
  K_m(H, |q0>) = span{ |q0>, H|q0>, H^2|q0>, ..., H^(m-1)|q0> }
  ```

- Exact Hamiltonian action (PDF **Sec. II.B**, Eq. (4)):

  ```
  H|qk> = sum_I v_I^(k) |D_I>
  ```

- Selection score (PDF **Sec. II.C**, Eq. (6)):

  ```
  Ek = <qk|H|qk>
  eta_I^(k) = |v_I^(k)|^2 / | Ek - H_II |
  ```

- Compressed matvec is a *projection* onto a selected set S_k (PDF **Sec. II.C**, Eq. (7)–(8)):

  ```
  |v_tilde_k> = P_{S_k} (H|qk>)
  ```

- Orthogonalization step (PDF **Sec. II.E**, Eq. (11)–(12)):

  ```
  |rk>   = |v_tilde_k> - sum_{j=0..k} |qj><qj|v_tilde_k>
  |qk+1> = |rk> / sqrt(<rk|rk>)
  ```

- Ritz step (PDF **Sec. II.F**, Eq. (14)–(17))

### Where this appears in the code

Core implementation lives in:

- `src/ccik/core.py`
  - `ccik_ground_energy_dense(...)`: drives the loop “compute H|qk> → select S_k → build |v_tilde_k> → orthogonalize → next |q>”.
  - The selection score is implemented exactly in the PDF form (Eq. (6)), using a small denominator floor for numerical stability.

CAS Hamiltonian helpers live in:

- `src/ccik/pyscf_cas.py`
  - `build_cas_hamiltonian_pyscf(...)`: builds (h1, eri, ecore) via PySCF/CASCI.

## Important implementation notes (differences vs the PDF)

The PDF assumes the Krylov basis is orthonormal (PDF Eq. (13)) so that the overlap matrix in the Ritz problem is S^(m) = I.

This implementation **additionally compresses** each new basis vector to a fixed top-$k$ support (`nkeep`) before continuing. That truncation can slightly break orthogonality in finite precision, so we solve a **generalized** Rayleigh–Ritz problem:

```
H c = E S c
```

instead of assuming $S=I$.

This is why the code uses `generalized_eigh(Hproj, Sproj)` rather than a plain eigenvalue solve.

Also, in addition to selecting determinants by the PT2-like score (PDF Eq. (6)), this code supports a practical “stabilizer”:

- Always keep the top-`Kv` coefficients by $|v_I^{(k)}|$.

  (Plain text: keep the top-Kv determinants by |v|.)

## Install / import

This repo uses a `src/` layout. You can run without installing by adding `src/` to `sys.path` (see `scripts/n2_cas_scan.py`).

From the repo root:

```bash
python scripts/n2_cas_scan.py
```

### Editable parameters (no code edits)

All the tunable knobs (scan range, CAS size, and CCIK parameters like `m`, `nadd`, `nkeep`, `Kv`) live in:

- packaged default: `src/ccik/defaults/config.toml`

Any key you omit falls back to the defaults in `CCIKParams`.

Edit that file and re-run the script. You can also pass a different config:

```bash
python scripts/n2_cas_scan.py --config configs/my_config.toml
```

## Minimal API

```python
from ccik import (
    CCIKParams,
    ccik_ground_energy_dense,
    make_mol_pyscf,
    build_cas_hamiltonian_pyscf,
)
from ccik.pyscf_cas import CASSpec

params = CCIKParams(m=200, nadd=1000, nkeep=5000, Kv=3000, verbose=False)

mol = make_mol_pyscf(atom='N 0 0 -0.5; N 0 0 0.5', basis='6-31g', unit='Angstrom', charge=0, spin=0)
cas = CASSpec(ncas=10, nelecas=10, ncore=2)

h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
E0_cas = ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=params)
E0_total = ecore + E0_cas
```

### Parameters (how to think about them)

- `m`: Krylov dimension (PDF Eq. (1)).
- `nadd`: how many external determinants to add by score $\eta$ each step (PDF Eq. (6) + the set $S_k$ in Eq. (7)).
- `nadd`: how many external determinants to add by score `eta` each step (PDF Eq. (6) + the set S_k in Eq. (7)).
- `nkeep`: how many coefficients to keep when compressing each Krylov basis vector (implementation detail; see “differences vs the PDF”).
- `Kv`: stabilizer size: always keep the top-`Kv` determinants by |v|.

## Output collection

The example script `scripts/n2_cas_scan.py` prints per-geometry energies and writes a CSV:

- `out_n2_cas_scan.csv`

You can duplicate that script and replace the molecule builder to sweep other systems.
