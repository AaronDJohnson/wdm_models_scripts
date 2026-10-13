"""D-A4: diagonal, dense-Cholesky and block-tridiagonal factors against dense linear
algebra and, bit for bit, against the oracle formulas they port (NumPy).  The diagonal and
dense factors are the one family of ``domainia.covariance`` (S4 merge, E-C2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve, solve_triangular

import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan
from domainia import linalg as la
from domainia import testing as T
from domainia.covariance import CholFactor, DiagFactor, dense_factor
from domainia.domains import CellSet

NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
GRID = wdm.Grid(NF, NT, DT)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def eq45_blocks(cells, ac, ad):
    """The WT TriCovariance.blocks call pattern (Bessel coefficients per block)."""
    plan = eq45_plan(NF, NT, DT, cells.keep_mask(), P=24)
    Lf = L_of(plan.f)
    return eq45_blocks_cheb(plan, lambda k: eq45_coeffs_bessel(Lf[k], ac, ad, 24, r=1.0))


class WTBlockTri:
    """WT ``matrix.BlockTri`` (wdm_paper_2/toy_models/src/wdm_toys/matrix.py:522-563),
    verbatim: the oracle formula the ragged layout must reproduce bit for bit."""

    def __init__(self, blocks, upper):
        self.sizes = [b.shape[0] for b in blocks]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)])
        Ls, Xs = [], []
        prev = None
        for i, A in enumerate(blocks):
            A = A.copy()
            if i and prev is not None and upper[i - 1] is not None and A.size and prev.size:
                X = solve_triangular(prev, upper[i - 1], lower=True)
                A -= X.T @ X
                Xs.append(X)
            else:
                Xs.append(None)
            L = np.linalg.cholesky(A) if A.size else A
            Ls.append(L)
            prev = L
        self.Ls, self.Xs = Ls, Xs

    def logdet(self):
        return 2.0 * sum(float(np.sum(np.log(np.diag(L)))) for L in self.Ls if L.size)

    def solve(self, b):
        b = np.asarray(b, float)
        one = b.ndim == 1
        B = b[:, None] if one else b
        z = []
        for i, L in enumerate(self.Ls):
            r = B[self.off[i]:self.off[i + 1]].copy()
            if self.Xs[i] is not None:
                r -= self.Xs[i].T @ z[i - 1]
            z.append(solve_triangular(L, r, lower=True) if L.size else r)
        x = [None] * len(self.Ls)
        for i in range(len(self.Ls) - 1, -1, -1):
            r = z[i]
            if i + 1 < len(self.Ls) and self.Xs[i + 1] is not None:
                r = r - self.Xs[i + 1] @ x[i + 1]
            x[i] = solve_triangular(self.Ls[i], r, lower=True, trans="T") if self.Ls[i].size else r
        out = np.concatenate(x, axis=0)
        return out[:, 0] if one else out


def random_blocktri(rng, sizes):
    n = sum(sizes)
    band = np.repeat(np.arange(len(sizes)), sizes)
    A = rng.standard_normal((n, n))
    C = A @ A.T + n * np.eye(n)
    C = np.where(np.abs(band[:, None] - band[None, :]) <= 1, C, 0.0)
    C = 0.5 * (C + C.T)
    off = np.concatenate([[0], np.cumsum(sizes)])
    diag = [C[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
    up = [C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] for i in range(len(sizes) - 1)]
    return C, diag, up


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# -- diagonal and dense ------------------------------------------------------------------


def test_diag_factor_is_wt_solver_diag(rng):
    v = rng.uniform(0.5, 2.0, 40)
    b = rng.standard_normal((40, 3))
    f = DiagFactor.build(v, NUMPY)
    assert f.logdet == float(np.sum(np.log(v))) and bool(f.ok)          # Solver.logdet
    assert np.array_equal(f.solve(b), b / v[:, None])                     # Solver.solve, 2-D
    assert np.array_equal(f.solve(b[:, 0]), b[:, 0] / v)
    with np.errstate(invalid="ignore"):
        assert not bool(DiagFactor.build(np.r_[v, -1.0], NUMPY).ok)


def test_chol_scipy_is_wt_solver_dense(rng):
    A = rng.standard_normal((30, 30))
    C = A @ A.T + 30 * np.eye(30)
    b = rng.standard_normal((30, 2))
    f = CholFactor.build(C, impl="scipy", be=NUMPY)
    cf = cho_factor(C, lower=True, check_finite=False)
    assert f.logdet == 2.0 * float(np.sum(np.log(np.diag(cf[0]))))       # Solver.logdet
    assert np.array_equal(f.solve(b), cho_solve(cf, b, check_finite=False))
    assert np.array_equal(np.triu(f.L, 1), np.zeros((30, 30)))


def test_chol_numpy_is_gb_dense_quad_logdet(rng):
    """GB matrix._dense_quad_logdet and MatrixCache.state (matrix.py:928-938, 978-983)."""
    A = rng.standard_normal((25, 25))
    S = A @ A.T + 25 * np.eye(25) + 1e-9 * rng.standard_normal((25, 25))  # not symmetric
    r = rng.standard_normal(25)
    f = dense_factor(S, chol_impl="numpy", symmetrize=True, be=NUMPY)
    L = np.linalg.cholesky(0.5 * (S + S.T))
    v = solve_triangular(L, r, lower=True, check_finite=False)
    assert np.array_equal(f.L, L)
    assert f.logdet == 2.0 * np.sum(np.log(np.diagonal(L)))
    u = f.whiten(r)
    assert np.array_equal(u, v) and u @ u == v @ v
    y = solve_triangular(L, v, lower=True, trans="T", check_finite=False)
    assert np.array_equal(f.solve(r), y)                                # MatrixCache y = C^-1 d


def test_chol_failure_is_flagged():
    f = CholFactor.build(-np.eye(3), impl="numpy", be=NUMPY)
    assert not bool(f.ok) and np.isnan(f.logdet)
    f = CholFactor.build(-np.eye(3), impl="scipy", be=NUMPY)
    assert not bool(f.ok)


def test_one_ok_rule_finite_and_positive():
    """S4 merge decision: a factor is ok when its defining diagonal (``v``; ``diag L``) is
    finite and positive.  S2's linalg copy took ``all(v > 0)`` and so passed a ``+inf``
    variance (``ok`` True) where the covariance family said False."""
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not bool(DiagFactor.build(np.array([1.0, np.inf, 2.0]), NUMPY).ok)
        assert not bool(DiagFactor.build(np.array([1.0, np.nan, 2.0]), NUMPY).ok)
        assert not bool(DiagFactor.build(np.array([1.0, 0.0, 2.0]), NUMPY).ok)
        assert bool(DiagFactor.build(np.array([1.0, 3.0, 2.0]), NUMPY).ok)
        big = np.diag([1.0, np.inf, np.inf])                          # inf on the diagonal
        assert not bool(CholFactor.build(big, impl="numpy", be=NUMPY).ok)
        assert not bool(CholFactor.build(big, impl="scipy", be=NUMPY).ok)


# -- band layout ---------------------------------------------------------------------


class WGBandLayout:
    """WG ``_BandLayout`` (gpu_sinusoid/wdmtoysgpu/matrix.py:148-180) in NumPy."""

    def __init__(self, band_cells, cells):
        pos = -np.ones(int(cells.max()) + 1, int)
        pos[cells] = np.arange(cells.size)
        self.nb = len(band_cells)
        self.B = max(len(c) for c in band_cells)
        slot = np.full((self.nb, self.B), -1)
        for m, c in enumerate(band_cells):
            slot[m, :len(c)] = pos[c]
        self.valid = slot >= 0
        self.src = np.where(slot >= 0, slot, 0)
        flat = slot.ravel()
        back = np.empty(cells.size, int)
        back[flat[flat >= 0]] = np.flatnonzero(flat >= 0)
        self.back = back


def test_band_layout_is_the_wg_layout(rng):
    cells = CellSet.all_active(GRID).drop_columns(3, 1)
    lay = la.BandLayout.from_cells(cells)
    packed = cells.packed_positions()
    band_cells = [packed[cells.m_array == m] for m in range(NF + 1)]
    wg = WGBandLayout(band_cells, packed)
    assert (lay.nb, lay.B, lay.n) == (wg.nb, wg.B, cells.size)
    assert np.array_equal(lay.valid, wg.valid)
    assert np.array_equal(lay.src, wg.src)
    assert np.array_equal(lay.back, wg.back)
    b = rng.standard_normal((cells.size, 2))
    p = lay.pack(b)
    assert p.shape == (lay.nb, lay.B, 2) and np.all(p[~lay.valid] == 0.0)
    assert np.array_equal(lay.unpack(p), b)


def test_pack_blocks_puts_identity_on_the_dummies(rng):
    _, diag, up = random_blocktri(rng, [3, 5, 2])
    lay = la.BandLayout.from_sizes([3, 5, 2])
    D, U = lay.pack_blocks(diag, up)
    assert D.shape == (3, 5, 5) and U.shape == (2, 5, 5)
    assert np.array_equal(D[0, :3, :3], diag[0]) and np.array_equal(D[0, 3:, 3:], np.eye(2))
    assert np.array_equal(D[0, :3, 3:], np.zeros((3, 2)))
    assert np.array_equal(U[1, :5, :2], up[1]) and np.all(U[1, :, 2:] == 0.0)
    assert np.array_equal(lay.add_identity_dummies(np.zeros((3, 5, 5)))[2], np.diag([0., 0, 1, 1, 1]))


# -- block tridiagonal ------------------------------------------------------------------


@pytest.mark.parametrize("sizes", [[3, 5, 4, 0, 2], [6], [4, 4, 4, 4]])
def test_ragged_and_padded_match_dense(rng, sizes):
    C, diag, up = random_blocktri(rng, sizes)
    b = rng.standard_normal((C.shape[0], 3))
    want_logdet = np.linalg.slogdet(C)[1]
    want = np.linalg.solve(C, b)
    fr = la.blocktri_factor(diag, up, layout="ragged")
    assert isinstance(fr.logdet, float) and fr.ok is True
    assert abs(fr.logdet - want_logdet) < 1e-12 * abs(want_logdet)
    assert rel(la.blocktri_solve(fr, b), want) < 1e-12
    assert np.allclose(la.blocktri_solve(fr, b[:, 0]), want[:, 0], rtol=1e-12, atol=0)
    w = la.blocktri_whiten(fr, b)
    assert np.allclose(np.sum(w * w, axis=0), np.sum(b * want, axis=0), rtol=1e-12)
    lay = la.BandLayout.from_sizes(sizes)
    fp = la.blocktri_factor(diag, up, layout="padded", band_layout=lay)
    assert bool(fp.ok) and abs(fp.logdet - fr.logdet) <= 1e-12 * abs(fr.logdet)
    assert rel(la.blocktri_solve(fp, b), la.blocktri_solve(fr, b)) <= 1e-12
    assert rel(la.blocktri_whiten(fp, b), w) <= 1e-12
    zp = la.blocktri_whiten_packed(fp, lay.pack(b))
    assert np.all(zp[~lay.valid] == 0.0)                                  # dummies stay zero


def test_ragged_is_wt_blocktri_bit_for_bit(rng):
    """The ragged layout on the Eq. (45) blocks of the 8 x 16 grid (raw and cut) is WT
    BlockTri: every factor block, the log-determinant and the solves."""
    for cells in (CellSet.all_active(GRID), CellSet.all_active(GRID).drop_columns(3, 1)):
        diag, up = eq45_blocks(cells, -2.0, 1.2)
        wt = WTBlockTri(diag, up)
        f = la.blocktri_factor(diag, up, layout="ragged")
        assert all(np.array_equal(a, b) for a, b in zip(f.Ls, wt.Ls, strict=True))
        assert all((a is None and b is None) or np.array_equal(a, b)
                   for a, b in zip(f.Xs, wt.Xs, strict=True))
        assert f.logdet == wt.logdet()
        b = rng.standard_normal((cells.size, 4))
        assert np.array_equal(la.blocktri_solve(f, b), wt.solve(b))
        assert np.array_equal(la.blocktri_solve(f, b[:, 1]), wt.solve(b[:, 1]))


def test_ported_wt_block_tridiagonal_solver_matches_dense(rng):
    """WT tests/test_matrix.py::test_block_tridiagonal_solver_matches_dense."""
    cells = CellSet.all_active(GRID)
    diag, up = eq45_blocks(cells, -2.0, 1.2)
    bt = la.blocktri_factor(diag, up, layout="ragged")
    C = T.eq45_dense(cells, drift(-2.0, 1.2))
    b = rng.standard_normal((cells.size, 3))
    assert bt.logdet == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-9)
    assert np.allclose(la.blocktri_solve(bt, b), np.linalg.solve(C, b), rtol=1e-8, atol=1e-10)


def test_padded_matches_ragged_on_eq45_blocks(rng):
    for cells in (CellSet.all_active(GRID), CellSet.all_active(GRID).drop_columns(3, 1)):
        diag, up = eq45_blocks(cells, -1.7, 0.9)
        fr = la.blocktri_factor(diag, up, layout="ragged")
        fp = la.blocktri_factor(diag, up, layout="padded", band_layout=la.BandLayout.from_cells(cells))
        assert abs(fp.logdet - fr.logdet) <= 1e-12 * abs(fr.logdet)
        b = rng.standard_normal((cells.size, 2))
        assert rel(la.blocktri_solve(fp, b), la.blocktri_solve(fr, b)) <= 1e-12


def test_a_failed_block_is_flagged_not_raised(rng):
    C, diag, up = random_blocktri(rng, [3, 4, 3])
    diag = [d.copy() for d in diag]
    diag[1] = -diag[1]
    fr = la.blocktri_factor(diag, up, layout="ragged")
    assert fr.ok is False and np.isnan(fr.logdet) and len(fr.Ls) == 1   # stopped at block 1
    with pytest.raises(ValueError, match="positive definite"):
        la.blocktri_solve(fr, np.ones(10))
    fp = la.blocktri_factor(diag, up, layout="padded", band_layout=la.BandLayout.from_sizes([3, 4, 3]))
    assert not bool(fp.ok) and np.isnan(fp.logdet)


def test_layout_arguments_are_checked(rng):
    _, diag, up = random_blocktri(rng, [2, 2])
    with pytest.raises(ValueError, match="band_layout"):
        la.blocktri_factor(diag, up, layout="padded")
    with pytest.raises(ValueError, match="upper blocks"):
        la.blocktri_factor(diag, [], layout="ragged")
    with pytest.raises(ValueError, match="layout"):
        la.blocktri_factor(diag, up, layout="dense")
    assert la.blocktri_factor(diag, up, layout="ragged", be=NUMPY).ok is True
