"""D-A5: the wdm_dev EV linear-algebra tests retired from ``wdm`` in C8 (part-1 plan note 7),
re-homed on ``domainia.linalg`` (originals: packages/wdm/tests/vendored/test_evolutionary.py,
which still runs them against the frozen ``wdm_dev_ev`` oracle)."""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from wdm.grid import active_mask
from wdm.matrix import active_W
from domainia import linalg as la

NF, NT, DT = 8, 16, 0.5
N = NF * NT


@pytest.fixture
def rng():
    return np.random.default_rng(20261010)


def _spd(n, rng, jitter=1.0):
    A = rng.standard_normal((n, n))
    return A @ A.T / n + jitter * np.eye(n)


def _blocktri(sizes, rng):
    off = np.concatenate([[0], np.cumsum(sizes)])
    C = np.zeros((off[-1], off[-1]))
    for i, s in enumerate(sizes):
        C[off[i]:off[i + 1], off[i]:off[i + 1]] = _spd(s, rng, jitter=4.0)
    for i in range(len(sizes) - 1):
        B = 0.3 * rng.standard_normal((sizes[i], sizes[i + 1]))
        C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = B
        C[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = B.T
    blocks = [C[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
    upper = [C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] for i in range(len(sizes) - 1)]
    return C, blocks, upper


def test_block_tridiagonal_solver(rng):
    C, blocks, upper = _blocktri([5, 3, 6, 4], rng)
    bt = la.blocktri_factor(blocks, upper, layout="ragged")
    b = rng.standard_normal((C.shape[0], 3))
    np.testing.assert_allclose(la.blocktri_solve(bt, b), np.linalg.solve(C, b), atol=1e-10)
    np.testing.assert_allclose(la.blocktri_solve(bt, b[:, 0]), np.linalg.solve(C, b[:, 0]), atol=1e-10)
    assert bt.logdet == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-10)


def test_solver_flavours_agree_with_numpy(rng):
    C, blocks, upper = _blocktri([4, 4, 4], rng)
    d = np.diag(C).copy()
    b = rng.standard_normal(C.shape[0])
    for f, ref in ((la.chol_factor(C, impl="scipy"), C),
                   (la.blocktri_factor(blocks, upper, layout="ragged"), C),
                   (la.diag_factor(d), np.diag(d))):
        np.testing.assert_allclose(la.factor_solve(f, b), np.linalg.solve(ref, b), atol=1e-10)
        assert float(f.logdet) == pytest.approx(np.linalg.slogdet(ref)[1], abs=1e-10)


def test_schur_edge_exact_matches_the_dense_matrix(rng):
    n = 14
    C = _spd(n, rng)
    e = np.array([0, 1, 12, 13])
    i = np.setdiff1d(np.arange(n), e)
    T = la.chol_factor(C[np.ix_(i, i)], impl="scipy")
    sch = la.schur_factor(T, C[np.ix_(e, e)], C[np.ix_(e, i)], e=e, i=i, form="solve",
                          impl="scipy")
    b = rng.standard_normal((n, 2))
    np.testing.assert_allclose(la.schur_solve(sch, b), np.linalg.solve(C, b), atol=1e-10)
    assert float(sch.logdet) == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-10)


def test_padded_marginal_is_the_data_block_of_the_extended_covariance(rng):
    pad = 4
    Ntp = NT + pad
    Np = NF * Ntp
    mask = active_mask(NF, Ntp)
    Wa = active_W(NF, Ntp, DT)
    Lam = _spd(Np, rng)
    unit = np.zeros((Np - N, Np))
    unit[np.arange(Np - N), np.arange(N, Np)] = 1.0
    Wp = wdm.forward_time_many(unit, NF, Ntp, DT)[:, mask].T
    pm = la.padded_factor(la.chol_factor(Lam, impl="scipy"), Wp, n_ext=Np, dt=DT, impl="scipy")
    C_ext = Wa.T @ Lam @ Wa / DT**2
    C_dd = C_ext[:N, :N]
    x = rng.standard_normal(N)
    ext = np.zeros((1, Np))
    ext[0, :N] = x
    z = la.padded_solve(pm, wdm.forward_time_many(ext, NF, Ntp, DT)[:, mask].T)
    full = np.zeros((1, NF + 1, Ntp))
    full[:, mask] = z.T
    got = DT * wdm.inverse_time_many(full, NF, Ntp, DT)[0, :N]
    np.testing.assert_allclose(got, np.linalg.solve(C_dd, x), rtol=1e-8, atol=1e-10)
    assert float(pm.logdet) == pytest.approx(np.linalg.slogdet(C_dd)[1], abs=1e-8)
