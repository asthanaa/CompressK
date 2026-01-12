import numpy as np

from ccik.core import apply_mask, compress_keep_top_mask, generalized_eigh, inner, normalize, occ_list_to_bitstring


def test_occ_list_to_bitstring_basic() -> None:
    assert occ_list_to_bitstring([]) == 0
    assert occ_list_to_bitstring([0]) == 1
    assert occ_list_to_bitstring([1]) == 2
    assert occ_list_to_bitstring([0, 2, 5]) == (1 << 0) | (1 << 2) | (1 << 5)


def test_normalize_and_inner() -> None:
    x = np.array([[3.0, 4.0]])
    xn = normalize(x)
    assert np.isclose(np.linalg.norm(xn.ravel()), 1.0)
    assert np.isclose(inner(xn, xn), 1.0)


def test_compress_keep_top_mask_nkeep() -> None:
    ci = np.array([[0.0, -2.0, 1.0], [3.0, 0.0, -0.5]])
    mask = compress_keep_top_mask(ci, nkeep=2)
    # keep |3.0| and |-2.0|
    assert mask.sum() == 2
    kept = np.sort(np.abs(ci[mask]))
    assert np.allclose(kept, np.array([2.0, 3.0]))


def test_compress_keep_top_mask_min_abs_fallback() -> None:
    ci = np.zeros((2, 3))
    mask = compress_keep_top_mask(ci, nkeep=2, min_abs=1.0)
    # no eligible entries; should keep exactly one (the argmax, here any index but deterministic)
    assert mask.sum() == 1


def test_apply_mask_roundtrip() -> None:
    ci = np.arange(6.0).reshape(2, 3)
    mask = np.zeros_like(ci, dtype=bool)
    mask[0, 1] = True
    mask[1, 2] = True
    out = apply_mask(ci, mask)
    assert np.all(out[~mask] == 0.0)
    assert np.all(out[mask] == ci[mask])


def test_generalized_eigh_residual_small() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    B = rng.normal(size=(4, 4))
    H = 0.5 * (B + B.T)

    evals, X = generalized_eigh(H, S)

    # Check residuals: H x ≈ e S x
    for i in range(4):
        x = X[:, i]
        r = H @ x - evals[i] * (S @ x)
        assert np.linalg.norm(r) < 1e-8
