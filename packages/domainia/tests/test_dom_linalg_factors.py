"""D-A5: Hermitian block, Schur and padded-marginal factors against dense linear algebra and,
bit for bit, against the oracle formulas they port (NumPy).  Since the S4 merge (E-C2) the
Schur and padded factors are the one family of ``domainia.covariance`` (``schur_factor``,
``padded_factor``); these oracle transcriptions -- WT ``Schur`` and ``PaddedMarginal``, GB
``EdgeSchurCache``, the WG diagonal ``_solve`` -- are the pins of their orders."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve, solve_triangular

import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan, exact_covariance
from domainia import linalg as la
from domainia import testing as T
from domainia.covariance import (BlockTriCovFactor, DiagFactor, HermitianCovariance, padded_factor,
                                 schur_factor)
from domainia.domains import CellSet

NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
GRID = wdm.Grid(NF, NT, DT)
AC, AD = -1.7, 0.9                     # WT test_matrix.py THETAS[0] noise indices


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def eq45_blocks(grid, cells, ac, ad):
    plan = eq45_plan(grid.Nf, grid.Nt, grid.dt, cells.keep_mask(), P=24)
    Lf = L_of(plan.f)
    r = plan.t_max / (N * DT)                                    # alpha continued over padding
    return eq45_blocks_cheb(plan, lambda k: eq45_coeffs_bessel(Lf[k], ac, ad, 24, r=r))


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# -- complex Hermitian (GB fullcache) ---------------------------------------------------


def gb_block_factor(M, b):
    """GB ``fullcache.block_factor`` (fullcache.py:82-95), NumPy, verbatim."""
    nb = M.shape[0] // b

    def blk(i, j):
        return M[i * b:(i + 1) * b, j * b:(j + 1) * b]

    Ls, Cs = [np.linalg.cholesky(blk(0, 0))], [np.zeros((b, b), dtype=M.dtype)]
    for i in range(1, nb):
        C = np.conj(solve_triangular(Ls[-1], blk(i - 1, i), lower=True, check_finite=False)).T
        Ls.append(np.linalg.cholesky(blk(i, i) - C @ np.conj(C).T))
        Cs.append(C)
    return np.stack(Ls), np.stack(Cs)


def gb_forward(Ls, Cs, h):
    """GB ``fullcache.forward``, block branch."""
    nb, b = Ls.shape[0], Ls.shape[1]
    u = [solve_triangular(Ls[0], h[:b], lower=True, check_finite=False)]
    for i in range(1, nb):
        u.append(solve_triangular(Ls[i], h[i * b:(i + 1) * b] - Cs[i] @ u[-1], lower=True,
                                  check_finite=False))
    return np.concatenate(u)


def gb_backward(Ls, Cs, u):
    """GB ``fullcache.backward``, block branch."""
    nb, b = Ls.shape[0], Ls.shape[1]
    y = [None] * nb
    y[-1] = solve_triangular(Ls[-1], u[(nb - 1) * b:], lower=True, trans="C", check_finite=False)
    for i in range(nb - 2, -1, -1):
        y[i] = solve_triangular(Ls[i], u[i * b:(i + 1) * b] - np.conj(Cs[i + 1]).T @ y[i + 1],
                                lower=True, trans="C", check_finite=False)
    return np.concatenate(y)


@pytest.mark.parametrize("block", [False, True])
def test_factor_reproduces_direct_inversion(block):
    """GB tests/test_fullcache.py::test_factor_reproduces_direct_inversion."""
    rng = np.random.default_rng(1)
    b, nb = 8, 6
    n = b * nb
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A @ A.conj().T / n + 3 * np.eye(n)
    if block:
        M = np.where(np.abs(np.arange(n)[:, None] // b - np.arange(n)[None, :] // b) <= 1, M, 0)
    z, h = (rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(2))
    f = HermitianCovariance(M, block=b if block else None).factor(np.zeros(1), NUMPY)
    u, t = f.whiten(z), f.whiten(h)
    y = f.backward(u)
    Mi = np.linalg.inv(M)
    assert abs(np.vdot(u, u).real - (z.conj() @ Mi @ z).real) < 1e-10
    assert abs(f.logdet - np.linalg.slogdet(M)[1]) < 1e-10
    assert np.max(np.abs(y - Mi @ z)) < 1e-12
    assert abs(np.vdot(t, t).real - (h.conj() @ Mi @ h).real) < 1e-10
    assert np.max(np.abs(f.solve(z) - Mi @ z)) < 1e-12


def test_cblock_is_gb_block_factor_bit_for_bit():
    rng = np.random.default_rng(4)
    b, nb = 6, 4
    n = b * nb
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A @ A.conj().T / n + 2 * np.eye(n)
    Ls, Cs = gb_block_factor(M, b)
    f = la.cblock_factor(M, b)
    assert np.array_equal(f.Ls, Ls) and np.array_equal(f.Cs, Cs) and bool(f.ok)
    assert f.logdet == 2.0 * np.sum(np.log(np.real(np.diagonal(Ls, axis1=-2, axis2=-1))))
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    u = gb_forward(Ls, Cs, z)
    assert np.array_equal(la.cblock_forward(f, z), u)
    assert np.array_equal(la.cblock_backward(f, u), gb_backward(Ls, Cs, u))


def test_cblock_failure_is_flagged():
    M = np.eye(8, dtype=complex)
    M[4:, 4:] *= -1.0
    f = la.cblock_factor(M, 4)
    assert not bool(f.ok) and np.isnan(f.logdet)
    with pytest.raises(ValueError, match="whole blocks"):
        la.cblock_factor(np.eye(9, dtype=complex), 4)


# -- Schur (edge-exact) ----------------------------------------------------------------


class WTSchur:
    """WT ``matrix.Schur`` (matrix.py:599-621), verbatim; ``T`` exposes solve/logdet."""

    def __init__(self, e, i, A, B, T):
        self.e, self.i, self.T = e, i, T
        self.TiBt = T.solve(B.T)
        Sc = A - B @ self.TiBt
        self.cf = cho_factor(0.5 * (Sc + Sc.T), lower=True, check_finite=False)
        self.B = B

    def solve(self, b):
        one = b.ndim == 1
        b = b[:, None] if one else b
        be, bi = b[self.e], b[self.i]
        ti = self.T.solve(bi)
        xe = cho_solve(self.cf, be - self.B @ ti, check_finite=False)
        xi = ti - self.TiBt @ xe
        x = np.empty_like(b)
        x[self.e], x[self.i] = xe, xi
        return x[:, 0] if one else x

    def logdet(self):
        return self.T.logdet() + 2.0 * float(np.sum(np.log(np.diag(self.cf[0]))))


class _Ragged:
    """The ragged factor seen through WT's Solver(blocktri=...) interface."""

    def __init__(self, f):
        self.f = f

    def solve(self, b):
        return la.blocktri_solve(self.f, b)

    def logdet(self):
        return self.f.logdet


def ragged(diag, up):
    """WT ``BlockTri`` as the covariance family's factor (cell order, ragged layout)."""
    bt = la.blocktri_factor(diag, up, layout="ragged")
    return BlockTriCovFactor(bt, bt.logdet, bt.ok, "ragged")


def consts(h):
    """A SchurFactor's constants: the edge / interior positions (a ragged or diagonal
    interior reads none)."""
    return {"e": h["e"], "i": h["i"], "interior": None}


@pytest.fixture(scope="module")
def hybrid():
    """The tri_exact hybrid on the 8 x 16 grid: exact rows on 3 leading + 1 trailing
    columns, Eq. (45) on the interior."""
    cells = CellSet.all_active(GRID)
    edge = cells.columns(3, 1)
    interior = cells.complement(edge)
    e, i = cells.index_of(edge), cells.index_of(interior)
    exact = exact_covariance(NF, NT, DT, drift(AC, AD))
    diag, up = eq45_blocks(GRID, interior, AC, AD)
    C = exact.copy()
    C[np.ix_(i, i)] = T.blocktri_dense(diag, up)
    return dict(e=e, i=i, exact=exact, diag=diag, up=up, C=C)


def test_schur_solve_form_matches_dense(hybrid, rng):
    h = hybrid
    base = ragged(h["diag"], h["up"])
    R = h["exact"][h["e"]]
    f = schur_factor(base, R[:, h["e"]], R[:, h["i"]], form="solve", chol_impl="scipy", be=NUMPY)
    assert bool(f.ok)
    assert f.logdet == pytest.approx(np.linalg.slogdet(h["C"])[1], abs=1e-9)
    b = rng.standard_normal((N, 2))
    assert rel(f.solve(b, C=consts(h)), np.linalg.solve(h["C"], b)) < 1e-9


def test_schur_solve_form_is_wt_schur_bit_for_bit(hybrid, rng):
    h = hybrid
    base = ragged(h["diag"], h["up"])
    R = h["exact"][h["e"]]
    A = R[:, h["e"]]
    wt = WTSchur(h["e"], h["i"], 0.5 * (A + A.T), R[:, h["i"]], _Ragged(base.bt))  # MatrixWdm.solver
    f = schur_factor(base, A, R[:, h["i"]], form="solve", chol_impl="scipy", be=NUMPY)
    assert np.array_equal(f.TiBt, wt.TiBt)
    assert np.array_equal(f.S.L, np.tril(wt.cf[0]))
    assert f.logdet == wt.logdet()
    b = rng.standard_normal((N, 3))
    assert np.array_equal(f.solve(b, C=consts(h)), wt.solve(b))
    assert np.array_equal(f.solve(b[:, 0], C=consts(h)), wt.solve(b[:, 0]))


def test_schur_scaled_form_is_gb_edge_schur_bit_for_bit(hybrid, rng):
    """GB ``EdgeSchurCache.state`` (matrix.py:1043-1060) on a diagonal interior."""
    h = hybrid
    E, I = h["e"], h["i"]
    R = h["exact"][E]
    vI = np.diag(h["exact"])[I] * rng.uniform(0.9, 1.1, I.size)
    d = rng.standard_normal(N)
    BD = R[:, I] / vI[None, :]
    S = 0.5 * (R[:, E] + R[:, E].T) - BD @ R[:, I].T
    Ls = np.linalg.cholesky(0.5 * (S + S.T))
    dI = d[I] / vI
    t = solve_triangular(Ls, d[E] - BD @ d[I], lower=True, check_finite=False)
    q0 = d[I] @ dI + t @ t
    logdet = np.sum(np.log(vI)) + 2.0 * np.sum(np.log(np.diagonal(Ls)))
    base = DiagFactor.build(vI, NUMPY)
    f = schur_factor(base, R[:, E], R[:, I], form="scaled", chol_impl="numpy", be=NUMPY)
    assert f.TiBt is None                                   # B D^-1 is formed on the fly, not stored
    assert np.array_equal(f.S.L, Ls) and f.logdet == logdet
    tt = f.whiten(d, C=consts(h))[I.size:]                  # [L_T^-1 d_I ; L_S^-1 (d_E - BD d_I)]
    assert np.array_equal(tt, t)
    assert d[I] @ base.solve(d[I]) + tt @ tt == q0
    C = h["exact"].copy()
    C[np.ix_(I, I)] = np.diag(vI)
    assert f.logdet == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-9)
    assert abs(q0 - d @ np.linalg.solve(C, d)) < 1e-9 * abs(q0)


def test_schur_scaled_solve_is_wg_diagonal_solve(hybrid, rng):
    """WG ``MatrixGPU._solve`` diagonal branch (matrix.py:614-625), in NumPy."""
    h = hybrid
    E, I = h["e"], h["i"]
    R = h["exact"][E]
    vI = np.diag(h["exact"])[I]
    f = schur_factor(DiagFactor.build(vI, NUMPY), R[:, E], R[:, I], form="scaled", chol_impl="scipy", be=NUMPY)
    b = rng.standard_normal((N, 2))
    Bm = R[:, I]
    Sc = 0.5 * (R[:, E] + R[:, E].T) - (Bm / vI[None, :]) @ Bm.T           # WG _factor, diagonal
    assert np.array_equal(f.S.L, np.tril(cho_factor(0.5 * (Sc + Sc.T), lower=True, check_finite=False)[0]))
    ti = b[I] / vI[:, None]
    xe = cho_solve((f.S.L, True), b[E] - Bm @ ti, check_finite=False)
    xi = ti - (xe.T @ Bm).T / vI[:, None]
    want = np.zeros_like(b)
    want[E], want[I] = xe, xi
    assert np.array_equal(f.solve(b, C=consts(h)), want)


def test_schur_failures(hybrid):
    h = hybrid
    R = h["exact"][h["e"]]
    Ree, Rei = R[:, h["e"]], R[:, h["i"]]
    bad = ragged([-d for d in h["diag"]], h["up"])
    f = schur_factor(bad, Ree, Rei, form="solve", chol_impl="scipy", be=NUMPY)
    assert not bool(f.ok) and f.S is None
    with pytest.raises(ValueError, match="positive definite"):
        f.solve(np.ones(N), C=consts(h))
    base = ragged(h["diag"], h["up"])
    g = schur_factor(base, 1e-6 * Ree, Rei, form="solve", chol_impl="scipy", be=NUMPY)
    assert not bool(g.ok) and np.isnan(g.logdet)                         # S not PD: -inf row
    with pytest.raises(TypeError, match="DiagFactor"):
        schur_factor(base, Ree, Rei, form="scaled", chol_impl="numpy", be=NUMPY)
    # the arguments are checked before the failed-base shortcut (fail closed)
    with pytest.raises(TypeError, match="DiagFactor"):
        schur_factor(bad, Ree, Rei, form="scaled", chol_impl="numpy", be=NUMPY)
    with pytest.raises(ValueError, match="form must be"):
        schur_factor(bad, Ree, Rei, form="dense", chol_impl="scipy", be=NUMPY)
    with pytest.raises(ValueError, match="chol_impl must be"):
        schur_factor(bad, Ree, Rei, form="solve", chol_impl="lapack", be=NUMPY)


# -- padded marginal --------------------------------------------------------------------

PAD = 2
EXT = wdm.Grid(NF, NT + PAD, DT)
NP = NF * (NT + PAD)


@pytest.fixture(scope="module")
def padded():
    ext = CellSet.all_active(EXT)
    W = T.dense_W(ext)                                         # (NP, NP), W W^T = dt I
    diag, up = eq45_blocks(EXT, ext, AC, AD)
    Lam = T.blocktri_dense(diag, up)
    C_dd = (W.T @ Lam @ W)[:N, :N] / DT ** 2
    return dict(W=W, diag=diag, up=up, Lam=Lam, C_dd=C_dd, Wp=W[:, N:])


def _to_time(W, z):
    return (W.T @ z)[:N]                                        # dt W^-1 z, data samples


def _pad_consts(p):
    """A PaddedFactor's constants (the ragged / diagonal ``Lambda`` reads none)."""
    m, n = wdm.grid.packed_cells(NF, NT + PAD)
    return {"Wp": p["Wp"], "m": m, "n_idx": n, "base": None}


def test_padded_marginal_matches_the_dense_data_covariance(padded, rng):
    p = padded
    lam = ragged(p["diag"], p["up"])
    f = padded_factor(lam, p["Wp"], grid=EXT, n=N, chol_impl="scipy", be=NUMPY)
    assert bool(f.ok)
    assert abs(f.logdet - np.linalg.slogdet(p["C_dd"])[1]) < 1e-9 * abs(f.logdet)
    x = rng.standard_normal(N)
    w = p["W"][:, :N] @ x                                       # cells of the zero-padded x
    assert np.array_equal(f.lift(x, C=_pad_consts(p)), wdm.forward_time_many(
        np.r_[x, np.zeros(NP - N)][None], NF, NT + PAD, DT)[0][wdm.active_mask(NF, NT + PAD)])
    got = _to_time(p["W"], f.solve_cells(w, C=_pad_consts(p)))
    assert rel(got, np.linalg.solve(p["C_dd"], x)) < 1e-9
    t = lam.solve(w)
    cb = f.reduce(t, C=_pad_consts(p))
    assert abs((w @ t - cb @ cb) - x @ got) < 1e-10 * abs(x @ got)        # WG forward form


def test_padded_factor_is_wt_padded_marginal_bit_for_bit(padded, rng):
    """WT ``PaddedMarginal`` (matrix.py:631-651) with the ragged factor as ``Lam``."""
    p = padded
    lam = ragged(p["diag"], p["up"])
    Wp = p["Wp"]
    LiWp = la.blocktri_solve(lam.bt, Wp)
    Qpp = Wp.T @ LiWp
    cf = cho_factor(0.5 * (Qpp + Qpp.T), lower=True, check_finite=False)
    logdet = lam.logdet - NP * np.log(DT) + 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    f = padded_factor(lam, Wp, grid=EXT, n=N, chol_impl="scipy", be=NUMPY)
    assert np.array_equal(f.LiWp, LiWp) and np.array_equal(f.Q.L, np.tril(cf[0]))
    assert f.logdet == logdet
    w = rng.standard_normal((NP, 2))
    Liw = la.blocktri_solve(lam.bt, w)
    want = Liw - LiWp @ cho_solve(cf, Wp.T @ Liw, check_finite=False)
    assert np.array_equal(f.solve_cells(w, C=_pad_consts(p)), want)


def test_padded_with_a_diagonal_base_and_failures(padded, rng):
    p = padded
    v = np.diag(p["Lam"]).copy()
    f = padded_factor(DiagFactor.build(v, NUMPY), p["Wp"], grid=EXT, n=N, chol_impl="scipy", be=NUMPY)
    C_dd = (p["W"].T @ np.diag(v) @ p["W"])[:N, :N] / DT ** 2
    assert abs(f.logdet - np.linalg.slogdet(C_dd)[1]) < 1e-9 * abs(f.logdet)
    bad = ragged([-d for d in p["diag"]], p["up"])
    g = padded_factor(bad, p["Wp"], grid=EXT, n=N, chol_impl="scipy", be=NUMPY)
    assert not bool(g.ok) and g.Q is None
    with pytest.raises(ValueError, match="positive definite"):
        g.solve_cells(np.ones(NP), C=_pad_consts(p))
    with pytest.raises(ValueError, match="chol_impl must be"):
        padded_factor(bad, p["Wp"], grid=EXT, n=N, chol_impl="lapack", be=NUMPY)
    with pytest.raises(ValueError, match="form must be"):
        padded_factor(bad, p["Wp"], grid=EXT, n=N, chol_impl="scipy", be=NUMPY, form="dense")


def test_the_duplicate_family_is_gone():
    """S4 deferral (a): one implementation per factor (spec section 0 rule 3).  S2's linalg
    copies of the diagonal, dense, Schur and padded factors are removed; the factor objects
    are ``domainia.covariance``'s."""
    import domainia.covariance as cov

    for name in ("DiagFactor", "CholFactor", "SchurFactor", "PaddedFactor", "diag_factor", "diag_solve",
                 "chol_factor", "chol_whiten", "chol_backward", "chol_solve", "schur_factor", "schur_solve",
                 "schur_edge_whiten", "padded_factor", "padded_solve", "padded_reduce", "factor_solve"):
        assert not hasattr(la, name), name
    assert set(la.__all__) == {"BandLayout", "BlockTriFactor", "CBlockFactor", "PCGResult", "blocktri_factor",
                               "blocktri_solve", "blocktri_solve_packed", "blocktri_whiten",
                               "blocktri_whiten_packed", "cblock_backward", "cblock_factor", "cblock_forward",
                               "device_constants", "pcg"}
    assert cov.DiagFactor is cov.base.DiagFactor and cov.SchurFactor is cov.edges.SchurFactor
