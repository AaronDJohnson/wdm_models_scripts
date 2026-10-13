"""WT's two "existing model" tests (``wdm_paper_2/toy_models/tests/test_matrix.py``) against
the WT classes themselves, transcribed -- the complement of E-A9's ports:

* ``test_raw_diagonals_are_the_existing_model`` (WT lines 76-82): band_raw / mid_raw equal
  WT's pre-matrix diagonal likelihood ``likelihood.WdmDiagonal`` (variance "diagonal" /
  "midpoint") with AND without the analysis cut.  E-A9's
  ``test_raw_diagonals_are_the_paper_i_evolutionary_whittle`` covers fmin = 0 only (the
  Paper I Whittle has no Fourier marginal); the fmin = 1 case -- the flat marginal over the
  discarded modes on a diagonal -- is pinned here;
* ``test_tri_exact_is_the_exact_model_for_stationary_noise`` (WT lines 207-214): at
  alpha_d = 0 Eq. (45) is exact, so tri_exact's lnL differences equal those of WT
  ``likelihood.WdmFullCovariance``.  E-A9 runs the same test with a domainia oracle
  (``ExactTimeCovariance.sigma_tilde`` + ``FourierBandNuisance``); this one shares no code
  with domainia (``wdm.evolutionary.time_domain_covariance`` and WT's cos/sin projection).

The release has no ``WdmDiagonal`` or ``WdmFullCovariance`` (spec section 7.5: dropped,
dense oracles only), so each is its ``lnlike`` transcribed line for line over the release
``wdm`` functions (``wdm_toys/likelihood.py:432-468`` and ``:510-543``; the discarded
Fourier modes as ``band.wdm_discarded``, ``band.py:57-68``, through ``wdm.forward_many``,
not ``TimeToWDM``).  Same small configuration, thetas and tolerances as WT (abs 1e-9 and
1e-7); the edges are E-A9's ``Geometry(cut=(3, 1), edge_exact=(3, 1))`` (WT monkeypatched
its module constants: the production 8/7 would cover the 16-column grid).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

import wdm
from wdm.evolutionary import draw_noise, time_domain_covariance
from wdm.grid import active_mask
from wdm.noise import evolutionary_band_variance, evolutionary_variance
from wdm_models.sinusoid.config import SinusoidConfig
from wdm_models.sinusoid.models import Geometry, likelihood

#: WT ``ToyConfig(Nf=8, Nt=16, dt=1/16, f_0=3.5, f_min=0.5)`` (A_h = A_n = 1, phi_0 = 0.6,
#: alpha_c = -2, alpha_d = 1.2, f_ref = 1): E-A9's ``CFG``.
CFG = SinusoidConfig(Nf=8, Nt=16, dt=1 / 16, A_h=1.0, f_0=3.5, A_n=1.0, f_min=0.5)
CUT = 1.0                                          # WT's second analysis_fmin
THETAS = (np.array([CFG.A_h, CFG.f_0, CFG.phi_0, 0.1, -1.7, 0.9]),
          np.array([0.8 * CFG.A_h, CFG.f_0 + 0.01, 0.4, -0.2, -2.3, -1.4]))
LOG_2PI = float(np.log(2.0 * np.pi))


def geometry(fmin: float) -> Geometry:
    return Geometry(cut=(3, 1), edge_exact=(3, 1), pad_columns=16, analysis_fmin=fmin)


def psd(ac: float, ad: float):
    """WT ``noise.psd`` at A_n = 1: ``(max(f, f_min)/f_ref)^(ac + ad (t/T - 1/2))``, T = N dt."""
    T = CFG.N * CFG.dt
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T - 0.5))
                               * np.log(np.maximum(np.asarray(f), CFG.f_min) / CFG.f_ref))


def _signal(theta) -> np.ndarray:
    return theta[0] * np.cos(2 * np.pi * theta[1] * (np.arange(CFG.N) * CFG.dt) + theta[2])


#: WT ``DATA = noise.draw(CFG, default_rng(3)) + signal`` in the release's ``wdm`` (the
#: realisation differs from WT's; both checks hold for any data).
DATA = (CFG.A_n ** 0.5 * draw_noise(CFG.N, CFG.dt, psd(CFG.alpha_c, CFG.alpha_d), np.random.default_rng(3))
        + _signal(np.array([CFG.A_h, CFG.f_0, CFG.phi_0])))


def keep_rfft(fmin: float) -> np.ndarray:
    """WT ``FourierBand.keep_rfft`` (``band.py:20-27``)."""
    f = np.fft.rfftfreq(CFG.N, d=CFG.dt)
    keep = (f >= fmin) & (f < 0.5 / CFG.dt)
    keep[[0, -1]] = False
    return keep


def discarded_spectra(fmin: float) -> np.ndarray:
    """WT ``FourierBand.discarded_spectra`` (``band.py:31-42``): ``(n_drop, N/2 + 1)``."""
    keep = keep_rfft(fmin)
    spectra = np.zeros((CFG.N - 2 * int(keep.sum()), CFG.N // 2 + 1), dtype=complex)
    row = 0
    for k in np.flatnonzero(~keep):
        if k in (0, CFG.N // 2):
            spectra[row, k] = np.sqrt(CFG.N)
            row += 1
        else:
            spectra[row, k] = np.sqrt(CFG.N / 2)
            spectra[row + 1, k] = -1j * np.sqrt(CFG.N / 2)
            row += 2
    return spectra


def variance_grid(variance: str, A_n: float, ac: float, ad: float) -> np.ndarray:
    """WT ``noise.diagonal_variance_grid`` / ``midpoint_variance_grid`` as ``wdm.noise``."""
    S = psd(ac, ad)
    if variance == "diagonal":
        return A_n * evolutionary_band_variance(CFG.Nf, CFG.Nt, CFG.dt, S)
    return A_n * evolutionary_variance(CFG.Nf, CFG.Nt, CFG.dt, lambda f, t: S(f[:, None], t[None, :]))


def wdm_diagonal_lnlike(variance: str, fmin: float, theta) -> float:
    """WT ``WdmDiagonal.lnlike`` (``edge_cut = 0``), transcribed."""
    r = DATA - _signal(theta)
    log10_A_n, alpha_c, alpha_d = theta[3:]
    mask = active_mask(CFG.Nf, CFG.Nt)
    w = wdm.forward_time(r, CFG.Nf, CFG.Nt, CFG.dt)
    if fmin:
        transformed = wdm.forward_many(discarded_spectra(fmin), CFG.Nf, CFG.Nt, CFG.dt)
        u = transformed[:, mask].T
        rows = np.flatnonzero(np.any(u != 0, axis=1))
        u = u[rows]
        v = variance_grid(variance, 1.0, alpha_c, alpha_d)[mask]
        H = u.T @ (u / v[rows, None])
        cf = cho_factor(H, lower=True, check_finite=False)
        logdet = float(np.log(v).sum()) - CFG.N * np.log(CFG.dt) + 2.0 * float(np.log(np.diag(cf[0])).sum())
        wp = w[mask].copy()
        z = u.T @ (wp[rows] / v[rows])
        a = cho_solve(cf, z, check_finite=False)
        wp[rows] -= u @ a
        A_n = 10.0 ** log10_A_n
        n_keep = 2 * int(keep_rfft(fmin).sum())
        quad = float(np.sum(wp * wp / v)) / A_n
        logdet += n_keep * np.log(A_n)
        return -0.5 * (quad + logdet + n_keep * LOG_2PI)
    var = variance_grid(variance, 10.0 ** log10_A_n, alpha_c, alpha_d)
    chi2 = float(np.sum((w[mask] ** 2) / var[mask]))
    logdet = float(np.sum(np.log(var[mask])))
    return -0.5 * (chi2 + logdet + int(mask.sum()) * LOG_2PI) + 0.5 * CFG.N * float(np.log(CFG.dt))


def wdm_full_lnlike(fmin: float, theta) -> float:
    """WT ``WdmFullCovariance.lnlike`` (dense time domain, A_n outside), transcribed."""
    r = DATA - _signal(theta)
    log10_A_n, alpha_c, alpha_d = theta[3:]
    cov = time_domain_covariance(CFG.N, CFG.dt, psd(alpha_c, alpha_d))
    n = CFG.N
    if fmin:
        k = np.flatnonzero(keep_rfft(fmin))
        phase = 2 * np.pi * k[:, None] * np.arange(CFG.N)[None, :] / CFG.N
        basis = np.concatenate((np.cos(phase), np.sin(phase))) * np.sqrt(2 / CFG.N)
        cov = basis @ cov @ basis.T
        r = basis @ r
        n = 2 * k.size
    cf = cho_factor(cov, lower=True, check_finite=False)
    logdet_tilde = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    A_n = 10.0 ** log10_A_n
    quad = float(r @ cho_solve(cf, r, check_finite=False)) / A_n
    return -0.5 * (quad + logdet_tilde + n * float(np.log(A_n)) + n * LOG_2PI)


@pytest.mark.parametrize("fmin", [0.0, CUT])
@pytest.mark.parametrize("base,variance", [("band", "diagonal"), ("mid", "midpoint")])
def test_raw_diagonals_are_the_existing_model(base, variance, fmin):
    new = likelihood(f"{base}_raw", CFG, DATA, geometry=geometry(fmin))
    for th in THETAS:
        assert float(new(th[None, :])[0]) == pytest.approx(wdm_diagonal_lnlike(variance, fmin, th), abs=1e-9)


@pytest.mark.parametrize("fmin", [0.0, CUT])
def test_tri_exact_is_the_exact_model_for_stationary_noise(fmin):
    tri = likelihood("tri_exact", CFG, DATA, geometry=geometry(fmin))
    a = [np.r_[th[:5], 0.0] for th in THETAS]
    d_new = float(tri(a[0][None, :])[0]) - float(tri(a[1][None, :])[0])
    d_old = wdm_full_lnlike(fmin, a[0]) - wdm_full_lnlike(fmin, a[1])
    assert d_new == pytest.approx(d_old, abs=1e-7)
