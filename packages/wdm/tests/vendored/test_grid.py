"""Tests for wdm.grid."""

from __future__ import annotations

import numpy as np
import pytest

from wdm.grid import (
    active_mask,
    band_frequencies,
    band_indices,
    carrier_phase,
    count_active,
    delta_F,
    delta_T,
    n_samples,
    time_pixel_centers,
    validate_grid,
)


@pytest.mark.parametrize("Nf,Nt", [(4, 8), (8, 16), (16, 32), (32, 64)])
def test_count_active_equals_N(Nf, Nt):
    """Total active WDM pixels must equal N for any valid grid."""
    assert count_active(Nf, Nt) == Nf * Nt


def test_active_mask_shape_and_dtype(simple_grid):
    """active_mask returns shape (Nf+1, Nt) of bool."""
    g = simple_grid
    mask = active_mask(g.Nf, g.Nt)
    assert mask.shape == (g.Nf + 1, g.Nt)
    assert mask.dtype == bool


def test_active_mask_interior_all_true(simple_grid):
    """Interior bands (m=1..Nf-1) are fully active."""
    g = simple_grid
    mask = active_mask(g.Nf, g.Nt)
    for m in range(1, g.Nf):
        assert np.all(mask[m])


def test_active_mask_dc_even_only(simple_grid):
    """DC band (m=0) has only even n active."""
    g = simple_grid
    mask = active_mask(g.Nf, g.Nt)
    assert np.all(mask[0] == (np.arange(g.Nt) % 2 == 0))


def test_active_mask_nyquist_parity(simple_grid):
    """Nyquist band (m=Nf) is active where (Nf + n) is even."""
    g = simple_grid
    mask = active_mask(g.Nf, g.Nt)
    assert np.all(mask[g.Nf] == ((g.Nf + np.arange(g.Nt)) % 2 == 0))


def test_band_indices_unique_modN(simple_grid):
    """For each band, the Nt indices are distinct."""
    g = simple_grid
    for m in range(g.Nf + 1):
        idx = band_indices(m, g.Nf, g.Nt)
        assert idx.shape == (g.Nt,)
        assert len(set(idx.tolist())) == g.Nt


def test_band_frequencies_endpoints(simple_grid):
    """freqs[0] == 0 and freqs[Nf] == Nyquist."""
    g = simple_grid
    freqs = band_frequencies(g.Nf, g.dt)
    assert freqs.shape == (g.Nf + 1,)
    assert freqs[0] == 0.0
    assert np.isclose(freqs[g.Nf], g.f_nyq)


def test_time_pixel_centers_shape(simple_grid):
    """time_pixel_centers returns Nt values, increasing by dT."""
    g = simple_grid
    t = time_pixel_centers(g.Nf, g.Nt, g.dt)
    assert t.shape == (g.Nt,)
    diffs = np.diff(t)
    assert np.allclose(diffs, g.dT)


def test_carrier_phase_even_m(simple_grid):
    """For even m, carrier_phase is all ones."""
    g = simple_grid
    assert np.all(carrier_phase(0, g.Nt) == 1.0)
    assert np.all(carrier_phase(2, g.Nt) == 1.0)


def test_carrier_phase_odd_m(simple_grid):
    """For odd m, carrier_phase alternates [+1, -1, +1, -1, ...]."""
    g = simple_grid
    expected = np.array([1.0 if n % 2 == 0 else -1.0 for n in range(g.Nt)])
    assert np.all(carrier_phase(1, g.Nt) == expected)
    assert np.all(carrier_phase(3, g.Nt) == expected)


def test_delta_relations(simple_grid):
    """dT * dF = 1/2 (critical sampling)."""
    g = simple_grid
    assert np.isclose(delta_T(g.Nf, g.dt) * delta_F(g.Nf, g.dt), 0.5)


def test_n_samples(simple_grid):
    g = simple_grid
    assert n_samples(g.Nf, g.Nt) == g.Nf * g.Nt


@pytest.mark.parametrize("bad", [-2, 0, 3, 7, 1])
def test_validate_grid_rejects_odd_or_nonpositive(bad):
    """Reject odd or non-positive Nf/Nt."""
    with pytest.raises(ValueError):
        validate_grid(bad, 16)
    with pytest.raises(ValueError):
        validate_grid(16, bad)


def test_validate_grid_accepts_valid():
    """Even positive ints pass validate_grid."""
    validate_grid(8, 16)
    validate_grid(2, 2)
