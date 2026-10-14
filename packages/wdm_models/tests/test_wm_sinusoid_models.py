"""E-A9: every sinusoid model against dense linear algebra on a small grid.

The WT likelihood-matrix tests (``wdm_paper_2/toy_models/tests/test_matrix.py``) that S2
deferred to the model layer, ported to the release: ``test_every_wdm_model_matches_its_dense_oracle``
(the 12 WDM models; 8 x 16 grid, ``f_min = 0.5``, edges and cut (3, 1), with and without the
analysis cut), ``test_raw_diagonals_are_the_existing_model`` (the existing model is now
``wdm``'s Paper I evolutionary Whittle likelihood) and
``test_tri_exact_is_the_exact_model_for_stationary_noise`` (the WT ``WdmFullCovariance`` is
now the dense time-domain Gaussian).  Every oracle is built independently of the domainia
pieces the models use: ``wdm.matrix.active_W`` rows, the ``wdm.noise`` variance grids, the
direct-sum Eq. (45) and the dense ``sigma_tilde``.
"""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from domainia.covariance import ExactTimeCovariance
from domainia.domains import CellSet
from domainia.likelihood import FourierWhittle, GaussianLikelihood
from domainia.nuisance import FourierBandNuisance, nuisance_basis
from domainia.testing import dense_W, eq45_dense, wdm_congruence
from wdm_models.common.params import MODELS, split_model
from wdm_models.options import ModelOptions, resolve
from wdm_models.sinusoid import config as C
from wdm_models.sinusoid import models as M
from wdm_models.sinusoid.psd import as_loglinear
from wdm_models.sinusoid.signal import Sinusoid

CFG = C.SinusoidConfig(Nf=8, Nt=16, dt=1 / 16, A_h=1.0, f_0=3.5, A_n=1.0, f_min=0.5)
CUT = 1.0
WDM_MODELS = tuple(m for m in MODELS if not m.startswith("whittle") and m != "exact")
_exact = ExactTimeCovariance(as_loglinear(CFG), CFG.N, CFG.dt, form="dense")
DATA = (_exact.draw((CFG.alpha_c, CFG.alpha_d), np.random.default_rng(3), CFG.A_n)
        + Sinusoid.for_config(CFG)(CFG.truth("band_raw")[:3]))
THETAS = [np.array([CFG.A_h, CFG.f_0, CFG.phi_0, 0.1, -1.7, 0.9]),
          np.array([0.8 * CFG.A_h, CFG.f_0 + 0.01, 0.4, -0.2, -2.3, -1.4])]


def geometry(fmin: float) -> M.Geometry:
    """The production 8 + 7 edges would cover this 16-column grid: (3, 1), as WT's test."""
    return M.Geometry(cut=(3, 1), edge_exact=(3, 1), pad_columns=16, analysis_fmin=fmin)


def oracle(y, C_, u):
    """Dense ``ln N(y; 0, C)`` marginalised over the columns of ``u`` (flat prior; WT verbatim)."""
    Ci = np.linalg.inv(C_)
    quad = y @ Ci @ y
    logdet = np.linalg.slogdet(C_)[1]
    n = len(y)
    if u is not None:
        H = u.T @ Ci @ u
        b = u.T @ Ci @ y
        quad -= b @ np.linalg.solve(H, b)
        logdet += np.linalg.slogdet(H)[1]
        n -= u.shape[1]
    return -0.5 * (quad + logdet + n * np.log(2 * np.pi))


def variance_grid(base: str, grid: wdm.Grid, nu) -> np.ndarray:
    """``(Nf + 1, Nt)`` Eq. 36 (band) / Eq. 39 (mid) from ``wdm.noise`` (A_n = 1)."""
    Sn = as_loglinear(CFG).bind(nu)
    if base == "band":
        return wdm.noise.evolutionary_band_variance(grid.Nf, grid.Nt, grid.dt, Sn)
    return wdm.noise.evolutionary_variance(grid.Nf, grid.Nt, grid.dt,
                                           lambda f, t: Sn(f[:, None], t[None, :]))


def approx(base: str, cells: CellSet, nu, t_max=None) -> np.ndarray:
    if base == "tri":
        return eq45_dense(cells, as_loglinear(CFG).bind(nu), t_max=t_max)
    return np.diag(variance_grid(base, cells.grid, nu)[cells.m_array, cells.n_array])


def dense_case(model: str, fmin: float, theta: np.ndarray) -> float:
    """``model``'s lnL assembled densely from independent pieces."""
    base, edge = split_model(model)
    g, geo = CFG.grid(), geometry(fmin)
    nu, A_n = (theta[4], theta[5]), 10.0 ** theta[3]
    r = DATA - Sinusoid.for_config(CFG)(theta[:3])
    B = FourierBandNuisance(CFG.N, CFG.dt, fmin).discarded_basis() if fmin else None
    if edge == "pad":
        ext = wdm.Grid(g.Nf, g.Nt + geo.pad_columns, g.dt)
        cells = CellSet.all_active(ext)
        Wd = dense_W(cells)[:, :CFG.N]
        Lam = approx(base, cells, nu, t_max=ext.Nt * wdm.delta_T(g.Nf, g.dt))
        Cdd = Wd.T @ Lam @ Wd / g.dt ** 2
        return oracle(r, A_n * Cdd, None if B is None else B.T)
    cells = CellSet.all_active(g)
    if edge == "cut":
        cells = cells.drop_columns(*geo.cut)
    W = dense_W(cells)
    u = None if B is None else nuisance_basis(W @ B.T)
    if edge == "exact":
        e = cells.columns(*geo.edge_exact)
        i = cells.index_of(cells.complement(e))
        Lam = wdm_congruence(_exact.sigma_tilde(nu), cells)
        Lam = 0.5 * (Lam + Lam.T)
        Lam[np.ix_(i, i)] = approx(base, cells.complement(e), nu)
    else:
        Lam = approx(base, cells, nu)
    return oracle(W @ r, A_n * Lam, u) + 0.5 * CFG.N * np.log(CFG.dt)


@pytest.mark.parametrize("fmin", [0.0, CUT])
@pytest.mark.parametrize("model", WDM_MODELS)
def test_every_wdm_model_matches_its_dense_oracle(model, fmin):
    lik = M.likelihood(model, CFG, DATA, geometry=geometry(fmin))
    assert isinstance(lik, GaussianLikelihood) and lik.param_names == (
        "A_h", "f_0", "phi_0", "log10_A_n", "alpha_c", "alpha_d")
    got = lik(np.stack(THETAS))
    for th, val in zip(THETAS, got, strict=True):
        assert val == pytest.approx(dense_case(model, fmin, th), abs=1e-7), model


@pytest.mark.parametrize("base", ["band", "mid"])
def test_raw_diagonals_are_the_paper_i_evolutionary_whittle(base):
    """Without the analysis cut, band/mid_raw are ``wdm``'s Paper I diagonal likelihood
    (``inner_product.loglike_evolutionary_whittle`` on the Eq. 36 / Eq. 39 grid, every active
    cell) plus the declared Jacobian ``N/2 ln dt``."""
    lik = M.likelihood(f"{base}_raw", CFG, DATA, geometry=geometry(0.0))
    g = CFG.grid()
    for th in THETAS:
        var = 10.0 ** th[3] * variance_grid(base, g, (th[4], th[5]))
        w_d = wdm.forward_time(DATA, g.Nf, g.Nt, g.dt)
        w_h = wdm.forward_time(Sinusoid.for_config(CFG)(th[:3]), g.Nf, g.Nt, g.dt)
        want = wdm.inner_product.loglike_evolutionary_whittle(w_d, w_h, var, g.Nf) + 0.5 * CFG.N * np.log(CFG.dt)
        assert float(lik(th[None, :])[0]) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("fmin", [0.0, CUT])
def test_tri_exact_is_the_exact_model_for_stationary_noise(fmin):
    """At ``alpha_d = 0`` Eq. 45 is exact, so tri_exact differs from the dense time-domain
    Gaussian (marginalised over the same Fourier modes) only by a constant."""
    lik = M.likelihood("tri_exact", CFG, DATA, geometry=geometry(fmin))
    a = [np.r_[th[:5], 0.0] for th in THETAS]
    B = FourierBandNuisance(CFG.N, CFG.dt, fmin).discarded_basis().T if fmin else None

    def exact(th):
        r = DATA - Sinusoid.for_config(CFG)(th[:3])
        return oracle(r, 10.0 ** th[3] * _exact.sigma_tilde((th[4], 0.0)), B)

    got = lik(np.stack(a))
    assert got[0] - got[1] == pytest.approx(exact(a[0]) - exact(a[1]), abs=1e-7)


def test_the_whittle_forms_and_their_layout():
    for model in ("whittle_raw", "whittle_cut"):
        lik = M.likelihood(model, CFG, DATA, geometry=geometry(CUT))
        assert isinstance(lik, FourierWhittle)
        assert lik.param_names == ("A_h", "f_0", "phi_0", "log10_A_n", "alpha")
    assert M.likelihood("whittle_cut", CFG, DATA, geometry=geometry(CUT)).crop == (24, 120)
    with pytest.raises(ValueError, match="analysis cut"):
        M.likelihood("whittle_raw", CFG, DATA, geometry=geometry(0.0))


def test_the_options_reach_the_domainia_pieces():
    """Per-job overrides change the algorithm, not the model (small-grid agreement)."""
    th = np.stack(THETAS)
    geo = geometry(CUT)
    base = M.likelihood("tri_raw", CFG, DATA, geometry=geo)(th)
    for opts in (ModelOptions(tri_build="gram"), ModelOptions(marginal_form="whiten"),
                 ModelOptions(layout="padded")):
        r = resolve("sinusoid", "tri_raw", "numpy", opts)
        assert np.max(np.abs(M.likelihood("tri_raw", CFG, DATA, geometry=geo, resolved=r)(th) - base)) <= 1e-8
    fixed = M.likelihood("band_exact", CFG, DATA, geometry=geo)(th)
    r = resolve("sinusoid", "band_exact", "numpy", ModelOptions(rows_order="adaptive"))
    assert np.max(np.abs(M.likelihood("band_exact", CFG, DATA, geometry=geo, resolved=r)(th) - fixed)) <= 1e-8
    lik = M.likelihood("tri_raw", CFG, DATA, geometry=geo,
                       resolved=resolve("sinusoid", "tri_raw", "numpy", ModelOptions(marginal_form="whiten")))
    assert lik.covs[0].form == "whiten"
    assert M.likelihood("tri_raw", CFG, DATA, geometry=geo).covs[0].form == "solve"


def test_the_builder_refuses_bad_input():
    with pytest.raises(ValueError, match="unknown model"):
        M.likelihood("band_win", CFG, DATA, geometry=geometry(CUT))
    with pytest.raises(ValueError, match="exact"):
        M.likelihood("exact", CFG, DATA, geometry=geometry(CUT))
    with pytest.raises(ValueError, match="samples"):
        M.likelihood("band_raw", CFG, DATA[:-1], geometry=geometry(CUT))
