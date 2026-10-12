"""Tests for wdm.stationary (fast slab-IFFT covariance from frequency PSD)."""

from __future__ import annotations

import numpy as np
import pytest

from wdm.matrix import active_W, build_C_time, build_W
from wdm.stationary import (
    assemble_lambda,
    assemble_lambda_full_active,
    build_lambda_blocks,
    build_lambda_blocks_full_active,
    build_lambda_blocks_packed,
    is_diagonal,
    unpack_diag_block,
)


def _rel(a, b, scale):
    """Max |a - b| relative to the scale of the data being compared.

    The absolute tolerances these tests used to carry (atol=1e-12 against
    blocks whose largest entry is ~1e-20 for a LISA PSD) would have passed
    against an all-zeros return.
    """
    return float(np.max(np.abs(a - b))) / scale


def _dense_lambda(Nf, Nt, dt, Sn_func):
    W = build_W(Nf, Nt, dt)
    C_time = build_C_time(Nf * Nt, dt, Sn_func)
    return W @ C_time @ W.T


def test_blocks_match_dense_reference(simple_grid, lisa_psd):
    """Fast slab-IFFT interior blocks match the dense W C W^T reference."""
    g = simple_grid
    diag, offd = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    Lambda = _dense_lambda(g.Nf, g.Nt, g.dt, lisa_psd)
    scale = float(np.max(np.abs(Lambda)))
    assert scale > 0
    for k, m in enumerate(range(1, g.Nf)):
        ref = Lambda[m * g.Nt : (m + 1) * g.Nt, m * g.Nt : (m + 1) * g.Nt]
        assert _rel(diag[k], ref, scale) < 1e-12
    for k, m in enumerate(range(1, g.Nf - 1)):
        ref = Lambda[m * g.Nt : (m + 1) * g.Nt, (m + 1) * g.Nt : (m + 2) * g.Nt]
        assert _rel(offd[k], ref, scale) < 1e-12


def test_packed_unpack_roundtrip(simple_grid, lisa_psd):
    """Packed diagonal blocks unpack to the full symmetric blocks."""
    g = simple_grid
    diag_full, _ = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    diag_packed, _ = build_lambda_blocks_packed(g.Nf, g.Nt, g.dt, lisa_psd)
    scale = max(float(np.max(np.abs(b))) for b in diag_full)
    assert scale > 0
    for full, packed in zip(diag_full, diag_packed, strict=False):
        assert _rel(unpack_diag_block(packed, g.Nt), full, scale) < 1e-12


def test_psd_array_input_one_sided(simple_grid, lisa_psd):
    """1D PSD on the rfftfreq grid produces the same blocks as the callable."""
    g = simple_grid
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = lisa_psd(freqs)
    diag_a, offd_a = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    diag_b, offd_b = build_lambda_blocks(g.Nf, g.Nt, g.dt, Sn_arr)
    scale = max(float(np.max(np.abs(b))) for b in diag_a)
    assert scale > 0
    for a, b in zip(diag_a, diag_b, strict=False):
        assert _rel(a, b, scale) < 1e-12
    for a, b in zip(offd_a, offd_b, strict=False):
        assert _rel(a, b, scale) < 1e-12


def test_psd_array_input_two_sided(simple_grid, lisa_psd):
    """1D PSD on the fftfreq grid (length N) produces the same blocks."""
    g = simple_grid
    freqs = np.fft.fftfreq(g.N, d=g.dt)
    Sn_arr = lisa_psd(np.abs(freqs))
    diag_a, offd_a = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    diag_b, offd_b = build_lambda_blocks(g.Nf, g.Nt, g.dt, Sn_arr)
    scale = max(float(np.max(np.abs(b))) for b in diag_a)
    assert scale > 0
    for a, b in zip(diag_a, diag_b, strict=False):
        assert _rel(a, b, scale) < 1e-12
    for a, b in zip(offd_a, offd_b, strict=False):
        assert _rel(a, b, scale) < 1e-12


def test_psd_matrix_input_diagonal(simple_grid, lisa_psd):
    """A diagonal 2D PSD matrix (Sn on diagonal) is accepted."""
    g = simple_grid
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = lisa_psd(freqs)
    # New convention: 2D form has S_n values on the diagonal (no pre-scaling).
    psd_mat = np.diag(Sn_arr)

    diag_callable, _ = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    diag_matrix, _ = build_lambda_blocks(g.Nf, g.Nt, g.dt, psd_mat)
    scale = max(float(np.max(np.abs(b))) for b in diag_callable)
    assert scale > 0
    for a, b in zip(diag_callable, diag_matrix, strict=False):
        assert _rel(a, b, scale) < 1e-12


def test_nonstationary_matrix_raises(simple_grid, lisa_psd):
    """A non-diagonal 2D PSD matrix is rejected."""
    g = simple_grid
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = lisa_psd(freqs)
    psd_mat = np.diag(Sn_arr)
    # Inject a single off-diagonal entry well above the rtol threshold.
    psd_mat[0, 1] = 0.5 * Sn_arr.max()
    with pytest.raises(ValueError, match="not diagonal"):
        build_lambda_blocks(g.Nf, g.Nt, g.dt, psd_mat)


def test_nonstationary_matrix_within_tolerance_accepted(simple_grid, flat_psd):
    """Tiny off-diagonal noise (numerical jitter) is tolerated by rtol."""
    g = simple_grid
    freqs = np.fft.rfftfreq(g.N, d=g.dt)
    Sn_arr = flat_psd(freqs)
    psd_mat = np.diag(Sn_arr)
    # Off-diagonal jitter at ~1e-14 of the diagonal should be tolerated
    # at the default rtol=1e-10.
    rng = np.random.default_rng(0)
    psd_mat += 1e-14 * Sn_arr.max() * rng.standard_normal(psd_mat.shape)
    psd_mat = 0.5 * (psd_mat + psd_mat.T)
    # Should not raise.
    build_lambda_blocks(g.Nf, g.Nt, g.dt, psd_mat, rtol=1e-10)


def test_invalid_1d_length_raises(simple_grid):
    """A 1D PSD with length ≠ N or N//2+1 raises."""
    g = simple_grid
    bad = np.ones(g.N + 3)
    with pytest.raises(ValueError, match="1D PSD array"):
        build_lambda_blocks(g.Nf, g.Nt, g.dt, bad)


def test_invalid_2d_shape_raises(simple_grid):
    """A 2D matrix with wrong shape raises."""
    g = simple_grid
    bad = np.eye(g.N + 1)
    with pytest.raises(ValueError, match="2D PSD matrix must have shape"):
        build_lambda_blocks(g.Nf, g.Nt, g.dt, bad)


def test_assemble_block_tridiagonal(simple_grid, lisa_psd):
    """assemble_lambda places blocks in the right rows/columns."""
    g = simple_grid
    diag, offd = build_lambda_blocks(g.Nf, g.Nt, g.dt, lisa_psd)
    full = assemble_lambda(diag, offd, g.Nf, g.Nt)
    assert full.shape == ((g.Nf + 1) * g.Nt, (g.Nf + 1) * g.Nt)
    # Edge-band rows are all zero (interior-only fast algorithm).
    assert np.all(full[: g.Nt, :] == 0)
    assert np.all(full[g.Nf * g.Nt :, :] == 0)
    # Symmetry across the off-diagonal.
    Lambda = _dense_lambda(g.Nf, g.Nt, g.dt, lisa_psd)
    interior = slice(g.Nt, g.Nf * g.Nt)
    scale = float(np.max(np.abs(Lambda)))
    assert _rel(full[interior, interior], Lambda[interior, interior], scale) < 1e-12


def test_full_active_blocks_match_dense_active(simple_grid, lisa_psd):
    """Full-active slab-IFFT (interior + edges) matches W_a @ C @ W_a^T at machine precision."""
    g = simple_grid
    diag, offd, n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
    W_a = active_W(g.Nf, g.Nt, g.dt)
    C_time = build_C_time(g.N, g.dt, lisa_psd)
    Lambda_a = W_a @ C_time @ W_a.T

    # Active-W ordering: m=0 (Nt/2 rows), m=1..Nf-1 (Nt each), m=Nf (Nt/2).
    offsets = [0]
    for m in range(g.Nf + 1):
        offsets.append(offsets[-1] + len(n_active[m]))

    for m in range(g.Nf + 1):
        ref = Lambda_a[offsets[m] : offsets[m + 1], offsets[m] : offsets[m + 1]]
        rel = np.max(np.abs(diag[m] - ref)) / max(np.max(np.abs(ref)), 1.0)
        assert rel < 1e-10, f"diag[m={m}] rel err = {rel:.3e}"

    for m in range(g.Nf):
        ref = Lambda_a[offsets[m] : offsets[m + 1], offsets[m + 1] : offsets[m + 2]]
        rel = np.max(np.abs(offd[m] - ref)) / max(np.max(np.abs(ref)), 1.0)
        assert rel < 1e-10, f"offd[m={m}] rel err = {rel:.3e}"


def test_full_active_assembly_matches_dense_W_C_W(simple_grid, lisa_psd):
    """Assembled full-active Λ matches the dense W @ C_time @ W^T at machine precision."""
    g = simple_grid
    diag, offd, n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
    Lambda_fast = assemble_lambda_full_active(diag, offd, n_active, g.Nf, g.Nt)

    W = build_W(g.Nf, g.Nt, g.dt)
    C_time = build_C_time(g.N, g.dt, lisa_psd)
    Lambda_dense = W @ C_time @ W.T

    rel = np.max(np.abs(Lambda_fast - Lambda_dense)) / np.max(np.abs(Lambda_dense))
    assert rel < 1e-10, f"|Λ_fast − W C W^T| / |Λ| = {rel:.3e}"


def test_full_active_block_shapes(simple_grid, lisa_psd):
    """Edge blocks are (Nt/2, Nt/2); interior blocks are (Nt, Nt); offd shapes match."""
    g = simple_grid
    diag, offd, _n_active = build_lambda_blocks_full_active(g.Nf, g.Nt, g.dt, lisa_psd)
    assert len(diag) == g.Nf + 1
    assert len(offd) == g.Nf
    assert diag[0].shape == (g.Nt // 2, g.Nt // 2)
    assert diag[g.Nf].shape == (g.Nt // 2, g.Nt // 2)
    for m in range(1, g.Nf):
        assert diag[m].shape == (g.Nt, g.Nt)
    assert offd[0].shape == (g.Nt // 2, g.Nt)
    assert offd[g.Nf - 1].shape == (g.Nt, g.Nt // 2)
    for m in range(1, g.Nf - 1):
        assert offd[m].shape == (g.Nt, g.Nt)


def test_is_diagonal():
    """is_diagonal flags off-diagonal entries above rtol * max(|diag|)."""
    A = np.diag([1.0, 2.0, 3.0])
    assert is_diagonal(A)
    B = A.copy()
    B[0, 1] = 1e-5
    assert not is_diagonal(B, rtol=1e-10)
    assert is_diagonal(B, rtol=1e-3)
