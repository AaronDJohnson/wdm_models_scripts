"""Tests for wdm.noise."""

from __future__ import annotations

import numpy as np

from wdm.noise import (
    band_averaged_variance,
    empirical_variance,
    evaluate_psd_on_band_grid,
    evolutionary_variance,
    whittle_variance,
    whittle_variance_from_array,
)


def test_whittle_variance_shape(simple_grid, flat_psd):
    """Variance grid has shape (Nf+1, Nt) and is positive on interior."""
    g = simple_grid
    var = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert var.shape == (g.Nf + 1, g.Nt)
    assert np.all(var[1 : g.Nf] > 0)


def test_whittle_variance_uniform_over_n(simple_grid, lisa_psd):
    """For stationary noise, variance is independent of time index n."""
    g = simple_grid
    var = whittle_variance(g.Nf, g.Nt, g.dt, lisa_psd)
    for m in range(1, g.Nf):
        # Relative: LISA-scale variances are ~1e-20, so an absolute tolerance
        # here would pass against an all-zeros return.
        assert np.max(np.abs(var[m] - var[m, 0])) <= 1e-12 * abs(var[m, 0])


def test_whittle_variance_value_flat_psd(simple_grid, flat_psd):
    """For flat S_n, var = S_n/2 everywhere."""
    g = simple_grid
    var = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    expected = 1e-3 / 2
    # Inspect interior bands; edges may also equal this since PSD is constant.
    assert np.allclose(var[1 : g.Nf], expected)


def test_whittle_from_array(simple_grid):
    """whittle_variance_from_array agrees with whittle_variance."""
    g = simple_grid

    def Sn(f):
        return 0.001 * (1 + (f / g.f_nyq) ** 2)

    Sn_band = evaluate_psd_on_band_grid(g.Nf, g.dt, Sn)
    var_a = whittle_variance_from_array(g.Nt, Sn_band)
    var_b = whittle_variance(g.Nf, g.Nt, g.dt, Sn)
    assert np.allclose(var_a, var_b)


def test_band_averaged_matches_whittle_for_flat_psd(simple_grid, flat_psd):
    """For constant S_n, band-averaged variance equals Whittle variance."""
    g = simple_grid
    var_w = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    var_b = band_averaged_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert np.allclose(var_w[1 : g.Nf], var_b[1 : g.Nf], rtol=1e-12)


def test_band_averaged_differs_for_steep_psd(simple_grid):
    """For a steep PSD, band-averaged variance differs from Whittle."""
    g = simple_grid

    def steep(f):
        f = np.asarray(f, dtype=float)
        return 1e-3 * (1.0 + 1e3 / (f + g.dF) ** 2)

    var_w = whittle_variance(g.Nf, g.Nt, g.dt, steep)
    var_b = band_averaged_variance(g.Nf, g.Nt, g.dt, steep)
    # Should differ at low m where the PSD is steep.
    assert not np.allclose(var_w[1, 0], var_b[1, 0], rtol=0.01)


def test_evolutionary_variance_shape(simple_grid):
    """evolutionary_variance returns (Nf+1, Nt)."""
    g = simple_grid

    def Sn(f, t):
        f = np.asarray(f, dtype=float)
        t = np.asarray(t, dtype=float)
        # Slowly varying in time; no f dependence for simplicity.
        return 1e-3 * (1.0 + 0.1 * t[None, :] / g.T) * np.ones_like(f[:, None])

    var = evolutionary_variance(g.Nf, g.Nt, g.dt, Sn)
    assert var.shape == (g.Nf + 1, g.Nt)
    # Time-dependence should be visible.
    assert not np.allclose(var[5, 0], var[5, -1])


def test_evolutionary_reduces_to_whittle_for_stationary(simple_grid, flat_psd):
    """If Sn(f, t) = Sn(f), evolutionary variance equals Whittle variance."""
    g = simple_grid

    def Sn_ft(f, t):
        f = np.asarray(f, dtype=float)
        t = np.asarray(t, dtype=float)
        return np.broadcast_to(flat_psd(f)[:, None], (f.size, t.size)).copy()

    var_evo = evolutionary_variance(g.Nf, g.Nt, g.dt, Sn_ft)
    var_w = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    assert np.allclose(var_evo, var_w)


def test_empirical_variance_matches_whittle_flat_psd(simple_grid, flat_psd):
    """MC estimate of variance matches Whittle prediction for flat PSD."""
    g = simple_grid
    var_mc = empirical_variance(g.Nf, g.Nt, g.dt, flat_psd, n_realizations=2000, seed=0)
    var_pred = whittle_variance(g.Nf, g.Nt, g.dt, flat_psd)
    # Compare interior bands; MC noise scales as 1/sqrt(n_realizations).
    rel_err = np.abs(var_mc[1 : g.Nf] - var_pred[1 : g.Nf]) / var_pred[1 : g.Nf]
    assert np.median(rel_err) < 0.05


def test_evaluate_psd_on_band_grid(simple_grid, lisa_psd):
    """evaluate_psd_on_band_grid returns (Nf+1,) array of S_n(f_m)."""
    g = simple_grid
    Sn_band = evaluate_psd_on_band_grid(g.Nf, g.dt, lisa_psd)
    assert Sn_band.shape == (g.Nf + 1,)
    # Should match direct evaluation.
    from wdm.grid import band_frequencies

    expected = np.where(
        np.isfinite(lisa_psd(band_frequencies(g.Nf, g.dt))),
        lisa_psd(band_frequencies(g.Nf, g.dt)),
        np.inf,
    )
    finite = np.isfinite(expected)
    assert (
        np.max(np.abs(Sn_band[finite] - expected[finite]) / np.abs(expected[finite]))
        < 1e-12
    )
