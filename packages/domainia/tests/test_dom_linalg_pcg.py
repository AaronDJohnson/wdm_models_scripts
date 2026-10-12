"""D-A6: preconditioned CG -- one frozen-row body (WG) whose single-row run is the WT
early-exit loop bit for bit, with the WT true-residual check (NumPy)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import linalg as la

N = 256
_rng = np.random.default_rng(5)
DVEC = np.exp(0.3 * _rng.standard_normal(N))
SPEC = np.exp(2.0 * np.cos(np.linspace(0, np.pi, N // 2 + 1)))


def A1(v):
    """SPD test operator D F^H S F D (one series)."""
    return DVEC * np.fft.irfft(np.fft.rfft(DVEC * v) * SPEC, n=N)


def M1(v):
    """Its stationary preconditioner (inexact: ignores D)."""
    return np.fft.irfft(np.fft.rfft(v) / SPEC, n=N)


def A2(V):
    return DVEC * np.fft.irfft(np.fft.rfft(DVEC * V, axis=-1) * SPEC, n=N, axis=-1)


def M2(V):
    return np.fft.irfft(np.fft.rfft(V, axis=-1) / SPEC, n=N, axis=-1)


def wt_pcg(A, b, M, tol=1e-7, maxit=200):
    """WT ``fastcov.pcg`` (src/wdm_toys/fastcov.py:121-143) verbatim, ``x0=None``."""
    bn = float(np.linalg.norm(b))
    if bn == 0.0:
        return np.zeros_like(b), 0
    x = np.zeros_like(b)
    r = b.copy()
    z = M(r)
    p = z.copy()
    rz = float(r @ z)
    for it in range(1, maxit + 1):
        Ap = A(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) <= tol * bn:
            return x, it
        z = M(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(f"PCG did not converge in {maxit} iterations")


def test_one_row_is_the_wt_loop_bit_for_bit():
    b = np.random.default_rng(1).standard_normal(N)
    for tol in (1e-7, 1e-10):
        x, it = wt_pcg(A1, b, M1, tol=tol)
        res = la.pcg(A2, M2, b[None], tol=tol, maxit=200)
        assert int(res.iterations) == it and it > 20
        assert np.array_equal(res.x[0], x)
        assert bool(res.converged[0])


def test_frozen_rows_reproduce_each_rows_own_run():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((4, N)) * np.array([[1.0], [1e3], [0.0], [1e-3]])
    B[1] += 50.0 * np.cos(2 * np.pi * 7 * np.arange(N) / N)
    res = la.pcg(A2, M2, B, tol=1e-9, maxit=300)
    its = []
    for r in range(4):
        x, it = wt_pcg(A1, B[r], M1, tol=1e-9, maxit=300)
        assert np.array_equal(res.x[r], x)                               # frozen after convergence
        its.append(it)
    assert int(res.iterations) == max(its) and its[2] == 0               # zero row: x = 0, no work
    assert np.all(res.converged)


def test_solution_matches_dense():
    Adense = A2(np.eye(N))
    B = np.random.default_rng(3).standard_normal((2, N))
    res = la.pcg(A2, M2, B, tol=1e-12, maxit=400)
    want = np.linalg.solve(Adense, B.T).T
    assert np.max(np.abs(res.x - want)) < 1e-9 * np.max(np.abs(want))


def test_true_residual_check():
    b = np.random.default_rng(4).standard_normal((1, N))
    with pytest.raises(RuntimeError, match="true residual"):
        la.pcg(A2, M2, b, tol=1e-7, maxit=200, true_rtol=0.0)
    res = la.pcg(A2, M2, b, tol=1e-7, maxit=200, true_rtol=0.0, on_fail="nan")
    assert not bool(res.converged[0]) and np.all(np.isnan(res.x[0]))


def test_iteration_limit():
    B = np.random.default_rng(5).standard_normal((2, N))
    with pytest.raises(RuntimeError, match="did not converge in 3 iterations"):
        la.pcg(A2, M2, B, tol=1e-10, maxit=3)
    res = la.pcg(A2, M2, B, tol=1e-10, maxit=3, on_fail="nan")
    assert int(res.iterations) == 3 and not np.any(res.converged)


def test_an_indefinite_row_fails_alone():
    """WG semantics: a row with p.Ap <= 0 is frozen as failed; the others finish."""
    B = np.random.default_rng(6).standard_normal((3, N))

    def A(V):
        out = A2(V)
        return out * np.array([[1.0], [-1.0], [1.0]])                    # row 1: negative definite

    with pytest.raises(RuntimeError, match="positive definiteness"):
        la.pcg(A, M2, B, tol=1e-9, maxit=300)
    res = la.pcg(A, M2, B, tol=1e-9, maxit=300, on_fail="nan")
    assert list(map(bool, res.converged)) == [True, False, True]
    assert np.all(np.isnan(res.x[1])) and np.all(np.isfinite(res.x[[0, 2]]))
    x0, _ = wt_pcg(A1, B[0], M1, tol=1e-9, maxit=300)
    assert np.array_equal(res.x[0], x0)


def test_arguments_are_checked():
    with pytest.raises(ValueError, match="on_fail"):
        la.pcg(A2, M2, np.ones((1, N)), tol=1e-7, maxit=10, on_fail="ignore")
    with pytest.raises(ValueError, match=r"\(R, N\)"):
        la.pcg(A2, M2, np.ones(N), tol=1e-7, maxit=10)
