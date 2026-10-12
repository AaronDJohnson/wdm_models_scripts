"""D-C2: BlockTridiagonalCovariance in the ragged (WT) and padded (WG) layouts.

* Both layouts are the dense Gaussian of the assembled block-tridiagonal matrix.
* Strategy cross-check: padded vs ragged <= 1e-12 (spec §3.2).
* Padded diagonal blocks are symmetrised and dummy slots carry the identity.
* A block that is not positive definite sets ``ok = False``.
"""

from __future__ import annotations

import numpy as np
import pytest
from wdm.backend import NUMPY

from domainia.covariance import BlockTridiagonalCovariance
from domainia.covariance.blocktri import BlockTriCovFactor, blocks_to_dense, dummy_slots, to_layout

SIZES = (3, 5, 5, 4, 2)
NU0 = np.zeros(2)


def _blocks(rng: np.random.Generator, sizes=SIZES) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Blocks of M = L L^T, L lower block-bidiagonal: M is SPD and block tri-diagonal."""
    nb = len(sizes)
    Ld = [np.tril(rng.standard_normal((s, s))) + 3 * np.eye(s) for s in sizes]
    Lo = [0.5 * rng.standard_normal((sizes[i + 1], sizes[i])) for i in range(nb - 1)]
    diag = [Ld[0] @ Ld[0].T] + [Lo[i - 1] @ Lo[i - 1].T + Ld[i] @ Ld[i].T for i in range(1, nb)]
    up = [Ld[i] @ Lo[i].T for i in range(nb - 1)]
    return diag, up


def _dense(diag, up) -> np.ndarray:
    return np.asarray(blocks_to_dense(diag, up, NUMPY))


@pytest.mark.parametrize("layout", ["ragged", "padded"])
def test_block_tridiagonal_is_the_dense_gaussian(backend, rng, layout):
    diag, up = _blocks(rng)
    M = _dense(diag, up)
    cov = BlockTridiagonalCovariance(lambda nu, be: ([be.asarray(d) for d in diag], [be.asarray(u) for u in up]),
                                     SIZES, layout=layout)
    assert cov.jittable == (layout == "padded") and cov.n == sum(SIZES)
    F = cov.factor(backend.asarray(NU0), backend)
    assert isinstance(F, BlockTriCovFactor) and bool(F.ok)
    b, B = rng.standard_normal(cov.n), rng.standard_normal((cov.n, 3))
    assert np.allclose(np.asarray(F.solve(backend.asarray(B), be=backend)), np.linalg.solve(M, B), rtol=1e-11, atol=1e-13)
    want = b @ np.linalg.solve(M, b)
    for form in ("solve", "whiten"):
        assert float(F.quad(backend.asarray(b), form=form, be=backend)) == pytest.approx(want, rel=1e-12)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(M)[1], rel=1e-13)
    assert np.allclose(cov.dense(NU0), M, rtol=0, atol=0)


def test_padded_matches_ragged(rng):
    """Strategy cross-check (spec §3.2): <= 1e-12."""
    diag, up = _blocks(rng)
    b = rng.standard_normal(sum(SIZES))
    out = {}
    for layout in ("ragged", "padded"):
        F = BlockTridiagonalCovariance(lambda nu, be: (diag, up), SIZES, layout=layout).factor(NU0)
        out[layout] = (float(F.logdet), float(F.quad(b)), np.asarray(F.solve(b)))
    assert abs(out["padded"][0] - out["ragged"][0]) <= 1e-12 * abs(out["ragged"][0])
    assert abs(out["padded"][1] - out["ragged"][1]) <= 1e-12 * abs(out["ragged"][1])
    assert np.max(np.abs(out["padded"][2] - out["ragged"][2])) <= 1e-12 * np.max(np.abs(out["ragged"][2]))


def test_to_layout_round_trip_and_dummies(rng):
    diag, up = _blocks(rng)
    d, u = to_layout(diag, up, SIZES, "padded", NUMPY)
    B = max(SIZES)
    assert d.shape == (len(SIZES), B, B) and u.shape == (len(SIZES) - 1, B, B)
    for m, s in enumerate(SIZES):
        assert np.array_equal(d[m][s:, s:], np.eye(B - s))
        assert not np.any(d[m][:s, s:]) and not np.any(d[m][s:, :s])
    assert np.array_equal(dummy_slots(SIZES)[1], np.zeros(B)) and dummy_slots(SIZES)[0][3:].tolist() == [1.0, 1.0]
    d2, u2 = to_layout(d, u, SIZES, "ragged", NUMPY)
    assert all(np.array_equal(a, b) for a, b in zip(d2, diag, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(u2, up, strict=True))
    asym = d.copy()
    asym[0, 0, 1] += 1.0                                          # an unsymmetrised (Gram) block
    d3, _ = to_layout(asym, u, SIZES, "padded", NUMPY)
    assert d3[0, 0, 1] == d3[0, 1, 0] == pytest.approx(d[0, 0, 1] + 0.5)


@pytest.mark.parametrize("layout", ["ragged", "padded"])
def test_not_positive_definite_sets_ok_false(backend, rng, layout):
    diag, up = _blocks(rng)
    diag[2] = -np.eye(SIZES[2])
    F = BlockTridiagonalCovariance(lambda nu, be: ([be.asarray(d) for d in diag], [be.asarray(x) for x in up]),
                                   SIZES, layout=layout).factor(backend.asarray(NU0), backend)
    assert not bool(F.ok)


def test_layout_is_validated():
    with pytest.raises(ValueError, match="layout"):
        BlockTridiagonalCovariance(lambda nu, be: ([], []), SIZES, layout="dense")
