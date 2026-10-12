"""D-C6: FourierBandNuisance, nuisance_basis and FlatMarginal (the 4 Hz analysis cut).

* 16256 retained / 128 discarded coordinates on the sinusoid grid; the discarded basis is
  orthonormal and orthogonal to ``project``.
* nuisance_basis keeps a full-rank image as given and replaces a deficient one by an
  orthonormal range basis.
* FlatMarginal is the dense flat-prior marginal (WT test_matrix.py ``oracle``) in both
  operation orders, on both backends; ok is the conjunction.
* G6: the pushed nuisance ``u`` (raw, cut) and the sinusoid band/mid/tri raw/cut
  (quad_H, logdet_H) and lnL at the truth (WT marginal_lnlike recombination).
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm

import domtest_inputs as T
from domainia.covariance import BlockTridiagonalCovariance, DenseCovariance, DiagonalCovariance, Eq45Covariance
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance, MarginalFactor, nuisance_basis
from domainia.spectra import LogLinearPSD

LOG_2PI = float(np.log(2.0 * np.pi))
NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)


def oracle(y: np.ndarray, C: np.ndarray, u: np.ndarray | None) -> tuple[float, float, int]:
    """Dense flat-prior marginal (WT test_matrix.py): (quad, logdet, n_eff)."""
    Ci = np.linalg.inv(C)
    quad, logdet, n = y @ Ci @ y, np.linalg.slogdet(C)[1], len(y)
    if u is not None:
        H, b = u.T @ Ci @ u, u.T @ Ci @ y
        quad -= b @ np.linalg.solve(H, b)
        logdet += np.linalg.slogdet(H)[1]
        n -= u.shape[1]
    return quad, logdet, n


def test_counts_on_the_sinusoid_grid():
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, 4.0)
    assert (nb.n_keep, nb.n_drop) == (16256, 128)
    assert not nb.keep_rfft[0] and not nb.keep_rfft[-1]
    B = nb.discarded_basis()
    assert B.shape == (128, T.SIN_N)
    assert np.allclose(B @ B.T, np.eye(128), atol=1e-12)
    x = np.random.default_rng(0).standard_normal(T.SIN_N)
    assert np.max(np.abs(B @ nb.project(x))) < 1e-10
    assert np.array_equal(nb.project(x.tolist()), nb.project(x))   # array-likes, as WT's np.asarray
    with pytest.raises(ValueError, match="positive"):
        FourierBandNuisance(T.SIN_N, T.SIN_DT, 0.0)


def test_nuisance_basis_branches(rng):
    u = rng.standard_normal((20, 4))
    assert nuisance_basis(u) is u
    deficient = np.hstack([u, u[:, :1]])
    V = nuisance_basis(deficient)
    assert V.shape == (20, 4) and np.allclose(V.T @ V, np.eye(4), atol=1e-12)
    assert np.allclose(V @ (V.T @ deficient), deficient, atol=1e-12)


@pytest.mark.parametrize("kind", ["diag", "tri", "dense"])
@pytest.mark.parametrize("form", ["solve", "whiten"])
def test_flat_marginal_is_the_dense_oracle(backend, rng, kind, form):
    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    nb = FourierBandNuisance(N, DT, 1.0)
    U = nb.push_to_cells(cells)
    if kind == "diag":
        cov = DiagonalCovariance(rng.uniform(0.5, 2.0, N), N)
    elif kind == "tri":                       # each backend's historical Eq. 45 strategy
        jax_leg = backend.name == "jax"
        cov = Eq45Covariance(PSD, cells, builder="gram" if jax_leg else "bessel",
                             layout="padded" if jax_leg else "ragged")
    else:
        A = rng.standard_normal((N, N))
        cov = DenseCovariance(A @ A.T / N + np.eye(N), N, chol_impl="scipy")
    fm = FlatMarginal(cov, U, form=form)
    F = fm.factor(backend.asarray(NU), backend)
    assert isinstance(F, MarginalFactor) and F.n_u == nb.n_drop and bool(F.ok)
    y = rng.standard_normal(N)
    q, ld, n_eff = oracle(y, cov.dense(NU), U)
    assert float(F.quad(backend.asarray(y), be=backend)) == pytest.approx(q, rel=1e-9)
    assert float(F.logdet) == pytest.approx(ld, rel=1e-11)
    assert N - F.n_u == n_eff


def test_ok_is_the_conjunction(backend):
    U = np.eye(4)[:, :1]
    with np.errstate(invalid="ignore"):
        F = FlatMarginal(DiagonalCovariance(np.array([1.0, 2.0, 3.0, -1.0]), 4), U).factor(
            backend.asarray(np.zeros(1)), backend)
    assert not bool(F.ok)
    # a ragged block that fails stops the base (WT LinAlgError): the marginal fails, no exception
    bad = BlockTridiagonalCovariance(lambda nu, be: ([be.asarray(-np.eye(2)), be.asarray(np.eye(2))],
                                                     [be.asarray(np.zeros((2, 2)))]), (2, 2))
    F = FlatMarginal(bad, U).factor(backend.asarray(np.zeros(1)), backend)
    assert not bool(F.ok) and np.isnan(float(F.logdet))
    with pytest.raises(ValueError, match="failed"):
        F.quad(backend.asarray(np.ones(4)))
    with pytest.raises(ValueError, match="U must be"):
        FlatMarginal(DiagonalCovariance(np.ones(4), 4), np.ones((3, 1)))
    with pytest.raises(ValueError, match="at least one column"):
        FlatMarginal(DiagonalCovariance(np.ones(4), 4), np.ones((4, 0)))


def test_g6_pushed_nuisance():
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, T.SIN_FMIN_CUT)
    g6 = T.sinusoid_mid()["g6"]["maps"]
    cells = CellSet.all_active(wdm.Grid(T.SIN_NF, T.SIN_NT, T.SIN_DT))
    T.assert_summary(nb.push_to_cells(cells), g6["band_raw"]["u"])
    U_cut = nb.push_to_cells(cells.drop_columns(6, 5))
    assert nuisance_basis(U_cut) is U_cut                          # full rank: kept as given
    T.assert_summary(U_cut, g6["band_cut"]["u"])
    T.assert_summary(nb.discarded_basis().T, g6["band_pad"]["u"])


@pytest.mark.parametrize("model", ["band_raw", "mid_raw", "band_cut", "mid_cut", "tri_raw", "tri_cut"])
def test_g6_marginal_parts_and_lnl(model):
    s = T.sinusoid_mid()
    ref = s["g6"]["parts"][model]
    base, edge = model.split("_")
    cells = CellSet.all_active(wdm.Grid(T.SIN_NF, T.SIN_NT, T.SIN_DT))
    if edge == "cut":
        cells = cells.drop_columns(6, 5)
    from domainia.covariance import keep_mask

    keep = keep_mask(cells)
    y = T.sinusoid_cells_residual()[keep]
    if base == "tri":
        cov = Eq45Covariance(LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T), cells, builder="bessel")
    else:
        v = s["arrays"][f"{base}_variance_raw"][keep]
        cov = DiagonalCovariance(v, v.size)
    nb = FourierBandNuisance(T.SIN_N, T.SIN_DT, T.SIN_FMIN_CUT)
    F = FlatMarginal(cov, nb.push_to_cells(cells)).factor(T.SIN_NU)
    p = F.parts(y)
    for key in ("quad_C", "quad_H", "logdet_C", "logdet_H"):
        assert float(p[key]) == pytest.approx(T.num(ref[key]), rel=1e-12, abs=0), key
    # WT marginal_lnlike recombination, A_n = 400, log-Jacobian 1/2 N ln dt (full N, also cut)
    A_n = T.num(ref["A_n"])
    n_eff = len(cells.m) - F.n_u
    log_jac = 0.5 * T.SIN_N * float(np.log(T.SIN_DT))
    lnl = -0.5 * (float(F.quad(y)) / A_n + float(F.logdet) + n_eff * np.log(A_n) + n_eff * LOG_2PI) + log_jac
    assert n_eff == ref["n_eff"] and log_jac == T.num(ref["log_jac"])
    assert abs(lnl - T.num(ref["lnL"])) <= 1e-9                    # the approved near gate


@pytest.mark.jax
@pytest.mark.parametrize("form", ["solve", "whiten"])
def test_jax_flat_marginal_jitted(jax_be, rng, form):
    import jax.numpy as jnp

    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    U = FourierBandNuisance(N, DT, 1.0).push_to_cells(cells)
    fm = FlatMarginal(Eq45Covariance(PSD, cells, builder="gram", layout="padded"), U, form=form)
    assert fm.jittable
    y = rng.standard_normal(N)
    Fn, Fj = fm.factor(NU), fm.factor(jnp.asarray(NU))
    assert type(Fj.logdet).__module__.startswith("jax")
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
    assert float(Fj.quad(jnp.asarray(y))) == pytest.approx(float(Fn.quad(y)), rel=1e-11)
