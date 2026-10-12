"""Tests for wdm.transform."""

from __future__ import annotations

import numpy as np
import pytest

from wdm.grid import active_mask
from wdm.transform import forward, forward_time, inverse, inverse_time


def test_roundtrip_normalized(simple_grid, white_noise):
    """forward -> inverse round-trip is exact."""
    g = simple_grid
    w = forward_time(white_noise, g.Nf, g.Nt, g.dt)
    x_rec = inverse_time(w, g.Nf, g.Nt, g.dt)
    assert np.allclose(x_rec, white_noise, atol=1e-12)


def test_parseval_normalized(simple_grid, white_noise):
    """Parseval: sum(w^2) == dt * sum(x^2) when normalized."""
    g = simple_grid
    w = forward_time(white_noise, g.Nf, g.Nt, g.dt)
    parseval = np.sum(w**2) / (g.dt * np.sum(white_noise**2))
    assert np.isclose(parseval, 1.0, atol=1e-12)


def test_output_shape_dtype(simple_grid):
    """Forward returns shape (Nf+1, Nt) of real floats."""
    g = simple_grid
    x = np.random.default_rng(0).standard_normal(g.N)
    w = forward_time(x, g.Nf, g.Nt, g.dt)
    assert w.shape == (g.Nf + 1, g.Nt)
    assert w.dtype.kind == "f"


def test_inactive_pixels_zero(simple_grid, white_noise):
    """Coefficients at inactive (masked-False) positions vanish to machine precision."""
    g = simple_grid
    w = forward_time(white_noise, g.Nf, g.Nt, g.dt)
    mask = active_mask(g.Nf, g.Nt)
    # Inactive pixels come from algebraic cancellations and are at the
    # floating-point noise floor, not bit-exactly zero.
    scale = np.max(np.abs(w[mask]))
    assert np.max(np.abs(w[~mask])) < 1e-12 * scale


def test_monochromatic_lands_in_band(medium_grid):
    """A pure cosine at frequency m*dF lands in band m."""
    g = medium_grid
    m_signal = 5
    t = np.arange(g.N) * g.dt
    x = np.cos(2 * np.pi * m_signal * g.dF * t)
    w = forward_time(x, g.Nf, g.Nt, g.dt)
    band_powers = np.sum(w**2, axis=1)
    assert np.argmax(band_powers) == m_signal


def test_linearity(simple_grid):
    """forward(a*x + b*y) = a*forward(x) + b*forward(y)."""
    g = simple_grid
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    y = rng.standard_normal(g.N)
    a, b = 1.7, -0.4
    w_combined = forward_time(a * x + b * y, g.Nf, g.Nt, g.dt)
    w_separate = a * forward_time(x, g.Nf, g.Nt, g.dt) + b * forward_time(
        y, g.Nf, g.Nt, g.dt
    )
    assert np.allclose(w_combined, w_separate, atol=1e-12)


def test_inverse_returns_complex_rfft(simple_grid):
    """inverse returns shape (N//2+1,) of complex."""
    g = simple_grid
    w = np.zeros((g.Nf + 1, g.Nt))
    w[5, 3] = 1.0
    out = inverse(w, g.Nf, g.Nt, g.dt)
    assert out.shape == (g.N // 2 + 1,)
    assert out.dtype.kind == "c"


def test_forward_accepts_rfft_directly(simple_grid):
    """forward(rfft(x)) equals forward_time(x)."""
    g = simple_grid
    x = np.random.default_rng(0).standard_normal(g.N)
    w_a = forward(np.fft.rfft(x), g.Nf, g.Nt, g.dt)
    w_b = forward_time(x, g.Nf, g.Nt, g.dt)
    assert np.allclose(w_a, w_b, atol=1e-15)


def _forward_reference(x_rfft, Nf, Nt, dt):
    """Naive per-band transcription of Algorithm 1, for cross-checking.

    Deliberately unoptimised: one band at a time, Hermitian slab lookups done
    element by element, no batching and no parity tricks. It shares no code
    with the shipped implementation beyond the window, so it is a genuine
    oracle for the batched fast path rather than a restatement of it.
    """
    from wdm.filter import phi_centered

    N = Nf * Nt
    phi_c = phi_centered(Nf, Nt, dt)
    kappa = 2.0 * np.sqrt(np.pi) / Nf
    out = np.zeros((Nf + 1, Nt))

    for m in range(Nf + 1):
        slab = np.empty(Nt, dtype=complex)
        for k in range(Nt):
            ell = k + (m - 1) * (Nt // 2)
            if ell < 0:
                slab[k] = np.conj(x_rfft[-ell])
            elif ell > N // 2:
                slab[k] = np.conj(x_rfft[N - ell])
            else:
                slab[k] = x_rfft[ell]
        y = np.fft.ifft(slab * phi_c)
        for n in range(Nt):
            c_conj = 1.0 if (m + n) % 2 == 0 else -1j
            sign = (-1.0) ** (((m + 1) * n) % 2)
            out[m, n] = kappa * sign * np.real(c_conj * y[n])

    out[0] /= np.sqrt(2)
    out[Nf] /= np.sqrt(2)
    return out


@pytest.mark.parametrize("dt", [0.5, 1.0, 3.0])
def test_forward_matches_a_naive_reference(simple_grid, white_noise, dt):
    """The batched fast path must agree with a straightforward transcription."""
    g = simple_grid
    x_rfft = np.fft.rfft(white_noise)
    fast = forward(x_rfft, g.Nf, g.Nt, dt)
    naive = _forward_reference(x_rfft, g.Nf, g.Nt, dt)
    assert np.max(np.abs(fast - naive)) < 1e-14 * np.max(np.abs(naive))
