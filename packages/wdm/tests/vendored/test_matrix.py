"""Tests for wdm.matrix."""

from __future__ import annotations

import numpy as np

from wdm.matrix import (
    active_W,
    build_C_time,
    build_C_wdm,
    build_C_wdm_active,
    build_W,
    real_to_rfft,
    rfft_to_real,
)
from wdm.transform import forward_time


def test_W_shape(simple_grid):
    """build_W returns shape ((Nf+1)*Nt, N)."""
    g = simple_grid
    W = build_W(g.Nf, g.Nt, g.dt)
    assert W.shape == ((g.Nf + 1) * g.Nt, g.N)


def test_W_matches_FFT_transform(simple_grid):
    """W @ x == forward(rfft(x)) to machine precision."""
    g = simple_grid
    W = build_W(g.Nf, g.Nt, g.dt)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    w_matrix = (W @ x).reshape(g.Nf + 1, g.Nt)
    w_fft = forward_time(x, g.Nf, g.Nt, g.dt)
    assert np.allclose(w_matrix, w_fft, atol=1e-13)


def test_active_W_orthogonal(simple_grid):
    """W_active @ W_active.T = I (orthogonal rows)."""
    g = simple_grid
    Wa = active_W(g.Nf, g.Nt, g.dt)
    assert Wa.shape == (g.N, g.N)
    err = np.max(np.abs(Wa @ Wa.T - np.eye(g.N)))
    assert err < 1e-12


def test_active_W_complete(simple_grid):
    """W_active.T @ W_active = I (orthogonal columns / completeness)."""
    g = simple_grid
    Wa = active_W(g.Nf, g.Nt, g.dt)
    err = np.max(np.abs(Wa.T @ Wa - np.eye(g.N)))
    assert err < 1e-12


def test_active_W_inverse_via_transpose(simple_grid):
    """W_active^T W_active x == x for any x."""
    g = simple_grid
    Wa = active_W(g.Nf, g.Nt, g.dt)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    assert np.allclose(Wa.T @ Wa @ x, x, atol=1e-12)


def test_C_time_symmetric(simple_grid, lisa_psd):
    """C_time is symmetric."""
    g = simple_grid
    C = build_C_time(g.N, g.dt, lisa_psd)
    assert np.max(np.abs(C - C.T)) < 1e-12 * np.max(np.abs(C))


def test_C_time_toeplitz(simple_grid, flat_psd):
    """C_time has Toeplitz (constant-along-diagonal) structure for stationary noise."""
    g = simple_grid
    C = build_C_time(g.N, g.dt, flat_psd)
    # Top row determines the matrix (modulo symmetry); check a few diagonals.
    for offset in [0, 1, 2, g.N // 3]:
        diag = np.diag(C, k=offset)
        # Allow tiny numerical jitter from irfft.
        assert np.max(np.abs(diag - diag[0])) < 1e-12 * np.max(np.abs(C))


def test_C_wdm_block_tridiagonal(simple_grid, lisa_psd):
    """C_wdm is block-tridiagonal in m: |m - m'| >= 2 blocks are tiny."""
    g = simple_grid
    C = build_C_wdm(g.Nf, g.Nt, g.dt, lisa_psd)
    Nb = g.Nf + 1
    # Compute block-Frobenius norms.
    diag_norm = np.linalg.norm(C[: g.Nt, : g.Nt])
    for m in range(Nb):
        for mp in range(Nb):
            if abs(m - mp) >= 2:
                block = C[m * g.Nt : (m + 1) * g.Nt, mp * g.Nt : (mp + 1) * g.Nt]
                # Distant blocks should be << diag_norm.
                assert np.linalg.norm(block) < 1e-3 * diag_norm


def test_C_wdm_diag_matches_whittle(simple_grid, flat_psd):
    """The active diagonal of C_wdm equals the Whittle variance for flat PSD."""
    g = simple_grid
    C = build_C_wdm(g.Nf, g.Nt, g.dt, flat_psd)
    diag = np.diag(C).reshape(g.Nf + 1, g.Nt)
    # Average over the active interior pixels; should equal S_n / 2.
    expected = 1e-3 / 2
    interior = diag[1 : g.Nf]
    assert np.isclose(np.mean(interior), expected, rtol=0.01)


def test_C_wdm_active_shape(simple_grid, lisa_psd):
    """C_wdm_active has shape (N, N)."""
    g = simple_grid
    C = build_C_wdm_active(g.Nf, g.Nt, g.dt, lisa_psd)
    assert C.shape == (g.N, g.N)


def test_rfft_real_roundtrip(simple_grid):
    """rfft_to_real and real_to_rfft are mutual inverses."""
    g = simple_grid
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    x_rfft = np.fft.rfft(x)
    x_real = rfft_to_real(x_rfft, g.N)
    assert x_real.shape == (g.N,)
    x_rfft_back = real_to_rfft(x_real, g.N)
    assert np.allclose(x_rfft_back, x_rfft, atol=1e-15)
