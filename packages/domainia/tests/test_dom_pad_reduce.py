"""S4 deferral (b): the padded marginal's WG-shaped noise state (``form="reduce"``).

``ZeroPadded(strategy="time_marginal", form="reduce")`` under ``FlatMarginal(form="reduce")``
stores what WG's ``MatrixGPU._factor`` pad branch stored -- ``Lambda``'s factor, ``cQ``,
``Gu`` and ``cH`` -- and evaluates ``_evaluate``'s pad branch: ``t = Lambda^-1 w``,
``cb = cQ^-1 W_p^T t``, ``quad = w.t - cb.cb - |cH^-1 (W_u^T t - Gu^T cb)|^2``
(``gpu_sinusoid/wdmtoysgpu/matrix.py:578-590, 648-662``).  Checked bit for bit against a
NumPy transcription of that branch (diagonal ``Lambda``), against WT's time-domain
``solve`` form (both bases), for the stored shapes, and jitted on JAX.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from scipy.linalg import solve_triangular
from wdm.backend import NUMPY

from domainia.covariance import DiagonalCovariance, Eq45Covariance, ZeroPadded
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD
from domainia.variance import MidpointVariance

NF, NT, DT, PAD = 8, 16, 1 / 16, 4
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
EXT = wdm.Grid(NF, NT + PAD, DT)
EXT_CELLS = CellSet.all_active(EXT)
NUIS = FourierBandNuisance(N, DT, 1.0)


def _base(kind):
    if kind == "tri":
        return Eq45Covariance(PSD, EXT_CELLS, builder="gram", t_max=EXT.N * DT)
    return DiagonalCovariance(MidpointVariance(PSD, EXT_CELLS), EXT.N)


def _marginal(kind, form, **kw):
    zp = ZeroPadded(_base(kind), strategy="time_marginal", grid=EXT, pad_columns=PAD,
                    form="reduce" if form == "reduce" else "solve", **kw)
    return FlatMarginal(zp, NUIS.discarded_basis().T, form=form)


def test_reduce_is_the_wg_pad_branch_bit_for_bit(rng):
    """WG ``_factor`` / ``_evaluate`` pad branch, diagonal ``Lambda``, ``jnp.linalg.cholesky``
    as ``np.linalg.cholesky`` (``chol_impl="numpy"``)."""
    cov = _marginal("diag", "reduce", chol_impl="numpy")
    C = cov.constants(NUMPY)
    F = cov.factor(NU, NUMPY)
    y = rng.standard_normal(N)
    v = np.asarray(MidpointVariance(PSD, EXT_CELLS)(NU, be=NUMPY))
    Wp, Wu = C["cov"]["Wp"], C["Wu"]
    # WG _factor, pad branch
    Q = Wp.T @ (Wp / v[:, None])
    cQ = np.linalg.cholesky(0.5 * (Q + Q.T))
    logdet = np.sum(np.log(v)) - float(EXT.N * np.log(EXT.dt)) + 2.0 * np.sum(np.log(np.diagonal(cQ)))
    LiWu = Wu / v[:, None]
    Gu = solve_triangular(cQ, Wp.T @ LiWu, lower=True, check_finite=False)
    H = Wu.T @ LiWu - Gu.T @ Gu
    cH = np.linalg.cholesky(0.5 * (H + H.T))
    logdet = logdet + 2.0 * np.sum(np.log(np.diagonal(cH)))
    # WG _evaluate, pad branch (y: the zero-padded residual's transform on every extended cell)
    ext = np.zeros(EXT.N)
    ext[:N] = y
    w = wdm.forward_time(ext, NF, NT + PAD, DT)[wdm.active_mask(NF, NT + PAD)]
    t = w / v
    cb = solve_triangular(cQ, Wp.T @ t, lower=True, check_finite=False)
    quad = np.dot(w, t) - np.dot(cb, cb)
    cu = solve_triangular(cH, Wu.T @ t - Gu.T @ cb, lower=True, check_finite=False)
    quad = quad - np.dot(cu, cu)
    assert np.array_equal(F.base.Q.L, cQ) and np.array_equal(F.Gu, Gu) and np.array_equal(F.H.L, cH)
    assert float(F.logdet) == logdet
    assert float(F.quad(y)) == quad


@pytest.mark.parametrize("kind", ["diag", "tri"])
def test_reduce_matches_the_wt_solve_form(kind, rng):
    y = rng.standard_normal(N)
    wt = _marginal(kind, "solve").factor(NU, NUMPY)
    wg = _marginal(kind, "reduce").factor(NU, NUMPY)
    assert abs(float(wg.logdet) - float(wt.logdet)) <= 1e-12 * abs(float(wt.logdet))
    assert abs(float(wg.quad(y)) - float(wt.quad(y))) <= 1e-9 * abs(float(wt.quad(y)))


def test_the_reduce_state_is_wg_shaped():
    """Lambda's factor, cQ (n_pad^2), Gu (n_pad x n_u), cH (n_u^2): no n' x n_pad array."""
    cov = _marginal("diag", "reduce")
    F = cov.factor(NU, NUMPY)
    n_pad, n_u, n_ext = PAD * NF, cov.U.shape[1], EXT.N
    assert F.base.LiWp is None and F.Zu is None
    assert F.base.Lam.v.shape == (n_ext,)
    assert F.base.Q.L.shape == (n_pad, n_pad) and F.Gu.shape == (n_pad, n_u) and F.H.L.shape == (n_u, n_u)
    assert cov.constants(NUMPY)["Wu"].shape == (n_ext, n_u)


def test_reduce_arguments_are_checked(rng):
    with pytest.raises(ValueError, match="form must be"):
        ZeroPadded(_base("diag"), strategy="time_marginal", grid=EXT, pad_columns=PAD, form="whiten")
    with pytest.raises(ValueError, match="time_marginal only"):
        ZeroPadded(_base("diag"), strategy="cell_congruence", form="reduce")
    solve_form = ZeroPadded(_base("diag"), strategy="time_marginal", grid=EXT, pad_columns=PAD)
    with pytest.raises(ValueError, match="form='reduce' marginalises"):
        FlatMarginal(solve_form, NUIS.discarded_basis().T, form="reduce")
    F = ZeroPadded(_base("diag"), strategy="time_marginal", grid=EXT, pad_columns=PAD,
                   form="reduce").factor(NU, NUMPY)
    with pytest.raises(ValueError, match="needs form='solve'"):
        F.solve(rng.standard_normal(N))
    x = rng.standard_normal(N)                       # the factor alone: w.t - |cb|^2
    wt = solve_form.factor(NU, NUMPY)
    assert abs(float(F.quad(x)) - float(wt.quad(x))) <= 1e-9 * abs(float(wt.quad(x)))


@pytest.mark.jax
@pytest.mark.parametrize("kind", ["diag", "tri"])
def test_reduce_on_jax(jax_be, kind, rng):
    import jax
    import jax.numpy as jnp

    cov = _marginal(kind, "reduce")
    y = rng.standard_normal(N)
    Fn = cov.factor(NU, NUMPY)
    Fj = cov.factor(jnp.asarray(NU), jax_be)
    assert isinstance(Fj.Gu, jax.Array) and Fj.base.LiWp is None
    Cj = cov.constants(jax_be)
    q = jax.jit(lambda F, C, y: F.quad(y, C=C, be=jax_be))(Fj, Cj, jnp.asarray(y))
    assert abs(float(Fj.logdet) - float(Fn.logdet)) <= 1e-12 * abs(float(Fn.logdet))
    assert abs(float(q) - float(Fn.quad(y))) <= 1e-11 * abs(float(Fn.quad(y)))
