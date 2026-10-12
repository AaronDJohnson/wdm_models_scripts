"""C5: the b5973b7 fixes and the ``cells`` convention of wdm.inner_product."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from wdm.grid import active_mask
from wdm.inner_product import (
    loglike_whittle,
    loglike_whittle_many,
    whittle,
    whittle_many,
)
from wdm.noise import whittle_variance
from wdm.transform import forward_time, forward_time_many

NF, NT, DT = 8, 16, 1.0


def _flat(f):
    return 1e-3 * np.ones_like(np.asarray(f, dtype=float))


def _lisa(f):
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Sn = (9e-30 / (2 * np.pi * f) ** 4 + 2.25e-22) * (1.0 + (f / 19.09e-3) ** 2)
    return np.where(np.isfinite(Sn), Sn, np.inf)


def _stack(B, seed=0):
    rng = np.random.default_rng(seed)
    W = forward_time_many(rng.standard_normal((B, NF * NT)), NF, NT, DT)
    w_d = forward_time(rng.standard_normal(NF * NT), NF, NT, DT)
    return w_d, W


@pytest.mark.parametrize("cells", ["active", "interior"])
def test_batched_per_template_variance_equals_the_loop(cells):
    """B = 12 > Nf: the vendored var[1:Nf] sliced the batch axis here."""
    B = 12
    w_d, W = _stack(B)
    var = whittle_variance(NF, NT, DT, _flat)
    stack = var[None] * np.linspace(0.5, 8.0, B)[:, None, None]
    for logdet in (True, False):
        many = loglike_whittle_many(w_d, W, stack, NF, include_logdet=logdet, cells=cells)
        loop = np.array([loglike_whittle(w_d, W[i], stack[i], NF, include_logdet=logdet,
                                         cells=cells) for i in range(B)])
        assert np.array_equal(many, loop)


@pytest.mark.parametrize("cells", ["active", "interior"])
def test_batch_of_one_keeps_the_logdet(cells):
    """B = 1: the vendored slice was empty, so include_logdet=True did nothing."""
    w_d, W = _stack(1)
    var = whittle_variance(NF, NT, DT, _flat)
    stack = var[None].copy()
    yes = loglike_whittle_many(w_d, W, stack, NF, include_logdet=True, cells=cells)
    no = loglike_whittle_many(w_d, W, stack, NF, include_logdet=False, cells=cells)
    assert yes[0] != no[0]
    assert np.array_equal(yes, [loglike_whittle(w_d, W[0], var, NF, cells=cells)])


def test_active_cells_are_exactly_the_active_mask():
    rng = np.random.default_rng(1)
    w = forward_time(rng.standard_normal(NF * NT), NF, NT, DT)
    var = whittle_variance(NF, NT, DT, _flat)
    mask = active_mask(NF, NT)
    expect = np.sum(w[mask] ** 2 / var[mask])
    got = whittle(w, w, var, NF)
    assert np.isclose(got, expect, rtol=1e-14, atol=0)
    junk = var.copy()
    junk[~mask] = -1.0                      # inactive pixels are not cells
    assert whittle(w, w, junk, NF) == got   # same code path: bit-identical
    ll = loglike_whittle(w, np.zeros_like(w), var, NF)
    ref = -0.5 * expect - 0.5 * np.sum(np.log(2 * np.pi * var[mask]))
    assert np.isclose(ll, ref, rtol=1e-14, atol=0)
    interior = np.sum(w[1:NF] ** 2 / var[1:NF])
    assert np.isclose(whittle(w, w, var, NF, cells="interior"), interior, rtol=1e-14, atol=0)


def test_active_cells_reject_a_psd_that_vanishes_at_dc():
    rng = np.random.default_rng(2)
    w = forward_time(rng.standard_normal(NF * NT), NF, NT, DT)
    var = whittle_variance(NF, NT, DT, _lisa)          # S(0) = inf -> 0 power at DC
    with pytest.raises(ValueError, match="cells='interior'"):
        whittle(w, w, var, NF)
    assert whittle(w, w, var, NF, cells="interior") > 0
    with pytest.raises(ValueError, match="cells must be"):
        whittle(w, w, var, NF, cells="edges")


def test_float32_strain_scale_does_not_underflow():
    """lisa/wdm b5973b7: w_a * w_b underflowed float32 at amplitude 1e-24."""
    Nf, Nt, dt = 32, 64, 10.0
    for amplitude in (1e-22, 1e-23, 1e-24):
        rng = np.random.default_rng(0)
        w = forward_time(amplitude * rng.standard_normal(Nf * Nt), Nf, Nt, dt)
        w32 = w.astype(np.float32)
        stack32 = np.stack([(c * w32).astype(np.float32) for c in (1.0, 0.5, 0.25)])
        var64 = whittle_variance(Nf, Nt, dt, _lisa)
        truth = loglike_whittle_many(w32.astype(np.float64), stack32.astype(np.float64), var64,
                                     Nf, include_logdet=False, cells="interior")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = loglike_whittle_many(w32, stack32, var64, Nf, include_logdet=False,
                                       cells="interior")
            quad_many = whittle_many(stack32, stack32, var64, Nf, cells="interior")
        assert np.allclose(got, truth, rtol=1e-6, atol=0.0)
        assert np.all(quad_many > 0.0) and len(np.unique(quad_many)) == 3
        assert got[0] == 0.0 and got[1] < 0.0 and got[2] < got[1]


def test_delta_F_rejects_unrepresentable_widths():
    from wdm.grid import delta_F

    assert delta_F(8, 1.0) == 0.0625
    with pytest.raises(ValueError, match="not representable"):
        delta_F(8, 5e-324)
    with pytest.raises(ValueError, match="Nf must be even"):
        delta_F(7, 1.0)


def test_rfft_packing_rejects_odd_N():
    from wdm.matrix import real_to_rfft, rfft_to_real

    x = np.random.default_rng(0).standard_normal(32)
    assert np.allclose(real_to_rfft(rfft_to_real(np.fft.rfft(x), 32), 32), np.fft.rfft(x))
    with pytest.raises(ValueError, match="N must be even"):
        rfft_to_real(np.fft.rfft(x[:5]), 5)
    with pytest.raises(ValueError, match="N must be even"):
        real_to_rfft(np.zeros(5), 5)
