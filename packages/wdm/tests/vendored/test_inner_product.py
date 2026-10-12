"""Tests for wdm.inner_product."""

from __future__ import annotations

import numpy as np

from wdm.inner_product import (
    band_averaged,
    evolutionary_whittle,
    gaussian,
    gaussian_from_func,
    loglike_band_averaged,
    loglike_evolutionary_whittle,
    loglike_gaussian,
    loglike_whittle,
    whittle,
)
from wdm.noise import (
    band_averaged_variance,
    evolutionary_variance,
    whittle_variance,
)
from wdm.transform import forward_time


def _freq_inner(x, h, Sn_array, dt, N):
    """Reference: standard frequency-domain noise-weighted inner product."""
    X = np.fft.rfft(x)
    H = np.fft.rfft(h)
    integrand = np.conj(X) * H / Sn_array
    return float(4.0 * dt / N * np.real(np.sum(integrand[1 : N // 2])))


def test_gaussian_exact_for_flat_psd(medium_grid, flat_psd):
    """Gaussian inner product equals frequency-domain answer to machine precision."""
    g = medium_grid
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    h = rng.standard_normal(g.N)
    w_x = forward_time(x, g.Nf, g.Nt, g.dt)
    w_h = forward_time(h, g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)

    ip_g = gaussian(w_x, w_h, g.Nf, g.Nt, g.dt, Sn_arr)
    ip_f = _freq_inner(x, h, Sn_arr, g.dt, g.N)
    assert np.isclose(ip_g, ip_f, rtol=1e-10)


def test_gaussian_exact_for_lisa_psd(medium_grid, lisa_psd):
    """Gaussian is exact regardless of how steeply the PSD varies."""
    g = medium_grid
    rng = np.random.default_rng(0)
    h = rng.standard_normal(g.N)
    w_h = forward_time(h, g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = np.where(np.isfinite(lisa_psd(freqs)), lisa_psd(freqs), 1e30)

    ip_g = gaussian(w_h, w_h, g.Nf, g.Nt, g.dt, Sn_arr)
    ip_f = _freq_inner(h, h, Sn_arr, g.dt, g.N)
    assert np.isclose(ip_g, ip_f, rtol=1e-10)


def test_gaussian_from_func_matches_array_version(simple_grid, flat_psd):
    """gaussian_from_func gives the same answer as gaussian with a precomputed array."""
    g = simple_grid
    rng = np.random.default_rng(0)
    h = rng.standard_normal(g.N)
    w = forward_time(h, g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)

    ip_a = gaussian(w, w, g.Nf, g.Nt, g.dt, Sn_arr)
    ip_b = gaussian_from_func(w, w, g.Nf, g.Nt, g.dt, flat_psd)
    assert np.isclose(ip_a, ip_b)


def test_whittle_symmetry(simple_grid, flat_psd):
    """<a|b> == <b|a> for the Whittle inner product."""
    g = simple_grid
    rng = np.random.default_rng(0)
    a = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    b = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert np.isclose(whittle(a, b, var, g.Nf), whittle(b, a, var, g.Nf))


def test_whittle_positivity(simple_grid, flat_psd, gaussian_pulse):
    """<a|a> > 0 for nonzero a."""
    g = simple_grid
    w = forward_time(gaussian_pulse, g.Nf, g.Nt, g.dt)
    var = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert whittle(w, w, var, g.Nf) > 0


def test_whittle_linearity(simple_grid, flat_psd):
    """<alpha*a + beta*b | c> = alpha <a|c> + beta <b|c>."""
    g = simple_grid
    rng = np.random.default_rng(1)
    a = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    b = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    c = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    alpha, beta = 1.7, -0.4
    lhs = whittle(alpha * a + beta * b, c, var, g.Nf)
    rhs = alpha * whittle(a, c, var, g.Nf) + beta * whittle(b, c, var, g.Nf)
    assert np.isclose(lhs, rhs)


def test_band_averaged_matches_whittle_for_flat_psd(simple_grid, flat_psd):
    """For constant S_n, band-averaged and Whittle inner products agree."""
    g = simple_grid
    rng = np.random.default_rng(0)
    w = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var_w = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    var_b = band_averaged_variance(g.Nf, g.Nt, g.dt, flat_psd)
    ip_w = whittle(w, w, var_w, g.Nf)
    ip_b = band_averaged(w, w, var_b, g.Nf)
    assert np.isclose(ip_w, ip_b, rtol=1e-12)


def test_evolutionary_reduces_to_whittle_for_stationary(simple_grid, flat_psd):
    """For a time-independent PSD, evolutionary == Whittle."""
    g = simple_grid

    def Sn_ft(f, t):
        f = np.asarray(f, dtype=float)
        t = np.asarray(t, dtype=float)
        return np.broadcast_to(flat_psd(f)[:, None], (f.size, t.size)).copy()

    rng = np.random.default_rng(0)
    w = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var_evo = evolutionary_variance(g.Nf, g.Nt, g.dt, Sn_ft)
    var_w = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert np.isclose(
        evolutionary_whittle(w, w, var_evo, g.Nf), whittle(w, w, var_w, g.Nf)
    )


def test_loglike_quadratic_part_zero_for_perfect_match(
    simple_grid, flat_psd, gaussian_pulse
):
    """The quadratic part of the log-likelihood vanishes when h == d.

    The full log-likelihood still includes the constant
    $-\\tfrac{1}{2}\\ln\\det(2\\pi C)$ piece, so the absolute value
    is nonzero; we test the quadratic-only form here.
    """
    g = simple_grid
    w = forward_time(gaussian_pulse, g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)
    assert loglike_gaussian(w, w, g.Nf, g.Nt, g.dt, Sn_arr, include_logdet=False) == 0.0


def test_loglike_negative_for_mismatch(simple_grid, flat_psd):
    """Log-likelihood is negative for nonzero residual."""
    g = simple_grid
    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)
    assert loglike_gaussian(w_d, w_h, g.Nf, g.Nt, g.dt, Sn_arr) < 0.0


def test_loglike_whittle_quadratic_part(simple_grid, flat_psd):
    """loglike_whittle without logdet equals -0.5 * inner product."""
    g = simple_grid
    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var = wdm_variance(simple_grid, flat_psd)
    quad = whittle(w_d - w_h, w_d - w_h, var, g.Nf)
    ll = loglike_whittle(w_d, w_h, var, g.Nf, include_logdet=False)
    assert np.isclose(ll, -0.5 * quad)


def test_loglike_whittle_includes_logdet(simple_grid, flat_psd):
    """include_logdet=True adds -0.5 sum ln(2 pi sigma^2)."""
    g = simple_grid
    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var = wdm_variance(simple_grid, flat_psd)
    ll_no = loglike_whittle(w_d, w_h, var, g.Nf, include_logdet=False)
    ll_yes = loglike_whittle(w_d, w_h, var, g.Nf, include_logdet=True)
    expected_logdet = sum(
        float(np.sum(np.log(2.0 * np.pi * var[m]))) for m in range(1, g.Nf)
    )
    assert np.isclose(ll_yes, ll_no - 0.5 * expected_logdet)


def test_loglike_band_averaged_alias(simple_grid, flat_psd):
    """loglike_band_averaged delegates to loglike_whittle."""
    g = simple_grid
    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var = wdm_variance(simple_grid, flat_psd)
    a = loglike_band_averaged(w_d, w_h, var, g.Nf)
    b = loglike_whittle(w_d, w_h, var, g.Nf)
    assert np.isclose(a, b)


def test_loglike_evolutionary_reduces_to_whittle(simple_grid, flat_psd):
    """For stationary noise, evolutionary log-likelihood == Whittle log-likelihood."""
    g = simple_grid

    def Sn_ft(f, t):
        f = np.asarray(f, dtype=float)
        t = np.asarray(t, dtype=float)
        return np.broadcast_to(flat_psd(f)[:, None], (f.size, t.size)).copy()

    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    var_evo = wdm_variance_evo(simple_grid, Sn_ft)
    var_w = wdm_variance(simple_grid, flat_psd)
    a = loglike_evolutionary_whittle(w_d, w_h, var_evo, g.Nf)
    b = loglike_whittle(w_d, w_h, var_w, g.Nf)
    assert np.isclose(a, b)


def test_loglike_gaussian_includes_logdet(simple_grid, flat_psd):
    """The logdet term equals ln det(2 pi C_wdm) for the *actual* covariance.

    Checked against ``build_C_wdm_active``, an independent construction, rather
    than by restating the implementation's own formula.
    """
    from wdm.matrix import build_C_wdm_active

    g = simple_grid
    rng = np.random.default_rng(0)
    w_d = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    w_h = forward_time(rng.standard_normal(g.N), g.Nf, g.Nt, g.dt)
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)
    ll_no = loglike_gaussian(w_d, w_h, g.Nf, g.Nt, g.dt, Sn_arr, include_logdet=False)
    ll_yes = loglike_gaussian(w_d, w_h, g.Nf, g.Nt, g.dt, Sn_arr, include_logdet=True)

    C = build_C_wdm_active(g.Nf, g.Nt, g.dt, flat_psd)
    expected_logdet = np.linalg.slogdet(2.0 * np.pi * C)[1]
    assert np.isclose(ll_yes, ll_no - 0.5 * expected_logdet, rtol=1e-9)


def test_loglike_whittle_zero_for_perfect_match(simple_grid, flat_psd, gaussian_pulse):
    """Quadratic part vanishes when h == d (logdet term remains)."""
    g = simple_grid
    w = forward_time(gaussian_pulse, g.Nf, g.Nt, g.dt)
    var = wdm_variance(simple_grid, flat_psd)
    quad_only = loglike_whittle(w, w, var, g.Nf, include_logdet=False)
    assert quad_only == 0.0


# Helpers (defined below to avoid touching module-level imports)
def wdm_variance(grid, Sn_func):
    from wdm.noise import whittle_variance

    return whittle_variance(grid.Nf, grid.Nt, grid.dt, Sn_func)


def wdm_variance_evo(grid, Sn_ft_func):
    from wdm.noise import evolutionary_variance

    return evolutionary_variance(grid.Nf, grid.Nt, grid.dt, Sn_ft_func)
