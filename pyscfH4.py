"""Compute H4 ground-state energy in a linear geometry (STO-3G).

This script runs a restricted Hartree–Fock (RHF) calculation in PySCF and
prints the total energy (electronic + nuclear repulsion).
"""

from __future__ import annotations


def main() -> None:
	from pyscf import gto, mcscf, scf

	# Linear H4 chain along x with nearest-neighbor distance r = 1.5 Angstrom.
	r = 1.5
	mol = gto.M(
		atom=[
			("H", (0.0 * r, 0.0, 0.0)),
			("H", (1.0 * r, 0.0, 0.0)),
			("H", (2.0 * r, 0.0, 0.0)),
			("H", (3.0 * r, 0.0, 0.0)),
		],
		basis="sto-6g",
		unit="Angstrom",
		charge=0,
		spin=0,  # 2S = Nalpha - Nbeta; 4 electrons -> singlet
		verbose=4,
	)

	mf = scf.RHF(mol)
	e_hf = mf.kernel()
	print(f"SCF converged: {mf.converged}")
	if not mf.converged:
		raise RuntimeError("SCF did not converge. Try increasing max_cycle or adjusting conv_tol.")

	# Active space: CAS(4,4) = 4 active electrons in 4 active orbitals.
	# For a singlet, specify alpha/beta active electrons explicitly as (2, 2).
	ncas = 4
	nelecas = (2, 2)
	mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
	e_cas = mc.kernel()[0]
	print(f"CASSCF converged: {mc.converged}")
	if not mc.converged:
		raise RuntimeError("CASSCF did not converge. Try increasing max_cycle or tightening/loosening conv_tol.")

	print("\n=== Results ===")
	print(f"RHF/STO-6G total energy (H4 linear, r={r} Å): {e_hf:.12f} Ha")
	print(f"CASSCF(4,4)/STO-6G total energy: {e_cas:.12f} Ha")


if __name__ == "__main__":
	main()
