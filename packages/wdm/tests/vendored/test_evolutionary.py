"""Tests for :mod:`wdm.evolutionary`, each against an explicit dense reference."""

from __future__ import annotations

import numpy as np
import pytest

from wdm import evolutionary as ev
from wdm.grid import active_mask
from wdm.matrix import active_W
from wdm.transform import forward_time

NF, NT, DT = 8, 16, 0.5
N = NF * NT


@pytest.fixture
def rng():
    return np.random.default_rng(20261010)


def _spd(n, rng, jitter=1.0):
    A = rng.standard_normal((n, n))
    return A @ A.T / n + jitter * np.eye(n)


def _blocktri(sizes, rng):
    """A random SPD matrix that is block tri-diagonal, with its blocks."""
    off = np.concatenate([[0], np.cumsum(sizes)])
    C = np.zeros((off[-1], off[-1]))
    for i, s in enumerate(sizes):
        C[off[i]:off[i + 1], off[i]:off[i + 1]] = _spd(s, rng, jitter=4.0)
    for i in range(len(sizes) - 1):
        B = 0.3 * rng.standard_normal((sizes[i], sizes[i + 1]))
        C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = B
        C[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = B.T
    blocks = [C[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
    upper = [C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] for i in range(len(sizes) - 1)]
    return C, blocks, upper


def test_endpoint_weights_are_the_trapezoid_rule():
    c = ev.endpoint_weights(N // 2 + 1)
    assert c[0] == c[-1] == 0.5
    assert np.all(c[1:-1] == 1.0)


def test_packed_grid_matches_the_dense_transform(rng):
    g = ev.PackedGrid(NF, NT, DT)
    Wa = active_W(NF, NT, DT)
    assert g.size == N
    np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
    x = rng.standard_normal((3, N))
    np.testing.assert_allclose(g.forward_many(x), x @ Wa.T, atol=1e-12)
    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT)[g.mask], atol=0)
    # round trip: W^{-1} = W^T / dt
    np.testing.assert_allclose(g.inverse_many(g.forward_many(x)), x, atol=1e-12)


def test_atoms_are_the_rows_of_the_transform():
    g = ev.PackedGrid(NF, NT, DT)
    Wa = active_W(NF, NT, DT)
    for p in range(g.size):
        np.testing.assert_allclose(g.atom(p), Wa[p], atol=1e-12)


def test_apply_mask_and_energy_fraction_match_the_dense_mask(rng):
    g = ev.PackedGrid(NF, NT, DT)
    Wa = active_W(NF, NT, DT)
    w2 = rng.uniform(0.0, 1.0, N)
    Mm = Wa @ np.diag(w2) @ Wa.T / DT
    Y = rng.standard_normal((2, N))
    np.testing.assert_allclose(g.apply_mask(Y, w2), Y @ Mm.T, atol=1e-12)
    np.testing.assert_allclose(g.energy_fraction(w2), np.diag(Mm), atol=1e-12)


def test_block_tridiagonal_solver(rng):
    C, blocks, upper = _blocktri([5, 3, 6, 4], rng)
    bt = ev.BlockTri(blocks, upper)
    b = rng.standard_normal((C.shape[0], 3))
    np.testing.assert_allclose(bt.solve(b), np.linalg.solve(C, b), atol=1e-10)
    np.testing.assert_allclose(bt.solve(b[:, 0]), np.linalg.solve(C, b[:, 0]), atol=1e-10)
    assert bt.logdet() == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-10)


def test_solver_flavours_agree_with_numpy(rng):
    C, blocks, upper = _blocktri([4, 4, 4], rng)
    d = np.diag(C).copy()
    b = rng.standard_normal(C.shape[0])
    for s, ref in ((ev.Solver(dense=C), C), (ev.Solver(blocktri=ev.BlockTri(blocks, upper)), C),
                   (ev.Solver(diag=d), np.diag(d))):
        np.testing.assert_allclose(s.solve(b), np.linalg.solve(ref, b), atol=1e-10)
        assert s.logdet() == pytest.approx(np.linalg.slogdet(ref)[1], abs=1e-10)


def test_schur_edge_exact_matches_the_dense_matrix(rng):
    n = 14
    C = _spd(n, rng)
    e = np.array([0, 1, 12, 13])
    i = np.setdiff1d(np.arange(n), e)
    T = ev.Solver(dense=C[np.ix_(i, i)])
    sch = ev.Schur(e, i, C[np.ix_(e, e)], C[np.ix_(e, i)], T)
    b = rng.standard_normal((n, 2))
    np.testing.assert_allclose(sch.solve(b), np.linalg.solve(C, b), atol=1e-10)
    assert sch.logdet() == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-10)


def test_padded_marginal_is_the_data_block_of_the_extended_covariance(rng):
    pad = 4
    v = ev.make_variant(NF, NT, DT, "pad", pad_columns=pad)
    g = v.grid
    Wa = active_W(NF, NT + pad, DT)
    Lam = _spd(g.size, rng)
    unit = np.zeros((g.N - N, g.N))
    unit[np.arange(g.N - N), np.arange(N, g.N)] = 1.0
    Wp = g.forward_many(unit).T
    pm = ev.PaddedMarginal(g, ev.Solver(dense=Lam), Wp, N)
    C_ext = Wa.T @ Lam @ Wa / DT**2           # time-domain covariance of the extended record
    C_dd = C_ext[:N, :N]
    x = rng.standard_normal(N)
    np.testing.assert_allclose(pm.solve(x), np.linalg.solve(C_dd, x), rtol=1e-8, atol=1e-10)
    assert pm.logdet() == pytest.approx(np.linalg.slogdet(C_dd)[1], abs=1e-8)


def test_marginal_lnlike_is_the_flat_prior_limit(rng):
    n, k = 12, 3
    C = _spd(n, rng)
    u = rng.standard_normal((n, k))
    y = rng.standard_normal(n)
    got = ev.marginal_lnlike(y, u, ev.Solver(dense=C), 1.0, n, 0.0)
    # flat prior = limit of a wide Gaussian prior: ln N(y; 0, C + s^2 u u^T) + (k/2) ln(2 pi s^2)
    s2 = 1e8
    Cp = C + s2 * u @ u.T
    lnN = -0.5 * (y @ np.linalg.solve(Cp, y) + np.linalg.slogdet(Cp)[1] + n * np.log(2 * np.pi))
    assert got == pytest.approx(lnN + 0.5 * k * np.log(2 * np.pi * s2), abs=1e-5)
    # with no nuisance it is the plain Gaussian, and A_n scales the covariance
    plain = ev.marginal_lnlike(y, None, ev.Solver(dense=C), 2.0, n, 0.0)
    ref = -0.5 * (y @ np.linalg.solve(2 * C, y) + np.linalg.slogdet(2 * C)[1] + n * np.log(2 * np.pi))
    assert plain == pytest.approx(ref, abs=1e-10)


def test_nuisance_basis_keeps_full_rank_and_reduces_degenerate(rng):
    u = rng.standard_normal((10, 3))
    assert ev.nuisance_basis(u) is u
    ud = np.column_stack([u, u[:, 0] + u[:, 1]])
    b = ev.nuisance_basis(ud)
    assert b.shape == (10, 3)
    np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-12)


def test_fourier_band_discards_an_orthonormal_low_frequency_set(rng):
    fmin = 0.2
    band = ev.FourierBand(NF, NT, DT, fmin)
    B = band.discarded_basis()
    f = np.fft.rfftfreq(N, d=DT)
    assert band.n_keep + band.n_drop == N
    assert band.n_keep == 2 * int(np.sum((f >= fmin) & (f > 0) & (f < 0.5 / DT)))
    np.testing.assert_allclose(B @ B.T, np.eye(band.n_drop), atol=1e-12)
    x = rng.standard_normal(N)
    np.testing.assert_allclose(B @ band.project(x), 0.0, atol=1e-12)
    rows, u = ev.wdm_discarded(band)
    full = np.stack([forward_time(b, NF, NT, DT)[active_mask(NF, NT)] for b in B]).T
    np.testing.assert_allclose(u, full[rows], atol=1e-12)
    assert np.max(np.abs(full[np.setdiff1d(np.arange(N), rows)])) < 1e-14   # rows it drops are empty


def test_fourier_band_zero_cut_is_the_identity(rng):
    band = ev.FourierBand(NF, NT, DT, 0.0)
    assert band.n_drop == 0
    x = rng.standard_normal(N)
    np.testing.assert_array_equal(band.project(x), x)


def test_variants(rng):
    raw = ev.make_variant(NF, NT, DT, "raw")
    assert raw.keep.all() and raw.w is None
    cut = ev.make_variant(NF, NT, DT, "cut", cut=(2, 1))
    assert set(np.unique(cut.grid.n[~cut.keep])) == {0, 1, NT - 1}
    pad = ev.make_variant(NF, NT, DT, "pad", pad_columns=4)
    assert pad.grid.Nt == NT + 4 and np.all(pad.grid.n[pad.keep] < NT)
    assert pad.w[:N].all() and not pad.w[N:].any()
    x = rng.standard_normal(N)
    np.testing.assert_allclose(cut.map(x), raw.map(x)[cut.keep], atol=0)
    with pytest.raises(ValueError):
        ev.make_variant(NF, NT, DT, "cut")
    with pytest.raises(ValueError):
        ev.make_variant(NF, NT, DT, "pad")
    with pytest.raises(ValueError):
        ev.make_variant(NF, NT, DT, "win")


# ---------------------------------------------------------------------------
# General S(f, t): band average, exact covariance, Eq. (45), wrap rule
# ---------------------------------------------------------------------------

from wdm.matrix import build_C_time, build_C_wdm_active  # noqa: E402
from wdm.noise import band_averaged_variance  # noqa: E402

T_REC = N * DT
F_NYQ = 0.5 / DT


def _S_stat(f):
    """A coloured stationary PSD, steep at both ends."""
    u = np.asarray(f, float) / F_NYQ
    return 1.0 + 0.3 / (u + 0.05) ** 2 + 5.0 * u**4


def _S_evo(f, t):
    """Slowly modulated, elementwise in (f, t)."""
    return _S_stat(f) * (1.0 + 0.4 * np.sin(2 * np.pi * np.asarray(t, float) / T_REC))


def test_band_average_reduces_to_the_stationary_band_average():
    got = ev.band_averaged_evolutionary_variance(NF, NT, DT, lambda f, t: _S_stat(f) + 0 * t)
    ref = band_averaged_variance(NF, NT, DT, _S_stat)
    np.testing.assert_allclose(got, ref, rtol=1e-13)


def test_band_average_samples_the_cell_centres():
    sep = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)
    stat = band_averaged_variance(NF, NT, DT, _S_stat)
    tn = np.arange(NT) * NF * DT
    np.testing.assert_allclose(sep, stat * (1 + 0.4 * np.sin(2 * np.pi * tn / T_REC))[None, :], rtol=1e-13)
    sub = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
    np.testing.assert_array_equal(sub, sep[[0, 3, NF]])


def test_time_domain_factor_is_the_square_root_of_the_paper_covariance():
    G = ev.time_domain_factor(N, DT, _S_evo)
    t = np.arange(N) * DT
    f = np.fft.rfftfreq(N, d=DT)
    eta = ev.endpoint_weights(f.size)
    St = _S_evo(f[None, :], t[:, None])                                   # (N, K)
    ref = (np.sqrt(St)[:, None, :] * np.sqrt(St)[None, :, :] * eta
           * np.cos(2 * np.pi * f * (t[:, None, None] - t[None, :, None]))).sum(-1) / (N * DT)
    np.testing.assert_allclose(G @ G.T, ref, atol=1e-12 * np.abs(ref).max())
    np.testing.assert_allclose(ev.time_domain_covariance(N, DT, _S_evo), ref, atol=1e-12 * np.abs(ref).max())


def test_stationary_limit_of_the_exact_covariance():
    Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
    np.testing.assert_allclose(ev.time_domain_covariance(N, DT, Sst), build_C_time(N, DT, _S_stat),
                               atol=1e-12)
    Lam = ev.exact_covariance(NF, NT, DT, Sst)
    np.testing.assert_allclose(Lam, build_C_wdm_active(NF, NT, DT, _S_stat), atol=1e-12)


def test_exact_covariance_is_the_transformed_time_covariance():
    Wa = active_W(NF, NT, DT)
    ref = Wa @ ev.time_domain_covariance(N, DT, _S_evo) @ Wa.T
    np.testing.assert_allclose(ev.exact_covariance(NF, NT, DT, _S_evo), ref, atol=1e-12 * np.abs(ref).max())


def test_draw_noise_is_reproducible():
    a = ev.draw_noise(N, DT, _S_evo, np.random.default_rng(5))
    b = ev.draw_noise(N, DT, _S_evo, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (N,)


def test_tri_covariance_is_exact_for_stationary_noise():
    g = ev.PackedGrid(NF, NT, DT)
    tri = ev.TriCovariance(g)
    Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
    ref = build_C_wdm_active(NF, NT, DT, _S_stat)
    np.testing.assert_allclose(tri.dense(Sst), ref, atol=1e-12 * np.abs(ref).max())
    diag, up = tri.blocks(Sst)
    C = np.zeros_like(ref)
    for m, c in enumerate(tri.band_cells):
        C[np.ix_(tri.pos[c], tri.pos[c])] = diag[m]
        if m < NF:
            c2 = tri.band_cells[m + 1]
            C[np.ix_(tri.pos[c], tri.pos[c2])] = up[m]
            C[np.ix_(tri.pos[c2], tri.pos[c])] = up[m].T
    np.testing.assert_allclose(C, ref, atol=1e-11 * np.abs(ref).max())


def test_tri_covariance_chebyshev_blocks_match_the_direct_sum():
    g = ev.PackedGrid(NF, NT, DT)
    tri = ev.TriCovariance(g)
    D = tri.dense(_S_evo)
    diag, up = tri.blocks(_S_evo)
    for m, c in enumerate(tri.band_cells):
        np.testing.assert_allclose(diag[m], D[np.ix_(tri.pos[c], tri.pos[c])], atol=1e-12 * np.abs(D).max())
        if m < NF:
            np.testing.assert_allclose(up[m], D[np.ix_(tri.pos[c], tri.pos[tri.band_cells[m + 1]])],
                                       atol=1e-12 * np.abs(D).max())
    # it solves through BlockTri like any block tri-diagonal covariance
    bt = ev.BlockTri(diag, up)
    assert np.isfinite(bt.logdet())


def test_tri_covariance_is_close_to_exact_for_slow_variation():
    """Eq. (45) is leading order in the timescale ratio; for a once-per-record
    modulation it is accurate on the diagonal away from the wrapped edges."""
    g = ev.PackedGrid(NF, NT, DT)
    D = ev.TriCovariance(g).dense(_S_evo)
    E = ev.exact_covariance(NF, NT, DT, _S_evo)
    inner = (g.n >= 3) & (g.n <= NT - 4) & (g.m > 0) & (g.m < NF)
    rel = np.abs(np.diag(D)[inner] / np.diag(E)[inner] - 1)
    assert rel.max() < 0.05


def test_wrap_fractions_match_the_atoms_directly():
    g = ev.PackedGrid(NF, NT, DT)
    frac = ev.wrap_fractions(NF, NT, DT)
    for p in (0, 3, 17, g.size // 2, g.size - 1):
        m, n = int(g.m[p]), int(g.n[p])
        e = g.atom(p) ** 2
        c = n * NF
        pos = c + (np.arange(N) - c + N // 2) % N - N // 2                # unwrapped positions
        assert frac[m, n] == pytest.approx(e[(pos < 0) | (pos >= N)].sum() / e.sum(), abs=1e-14)
    assert np.all(np.isnan(frac[~active_mask(NF, NT)]))


def test_wrap_cut_columns_grow_as_the_tolerance_tightens():
    loose = ev.wrap_cut_columns(NF, NT, DT, tol=1e-2)
    tight = ev.wrap_cut_columns(NF, NT, DT, tol=1e-4)
    assert tight[0] >= loose[0] and tight[1] >= loose[1]
    frac = np.nanmax(ev.wrap_fractions(NF, NT, DT), axis=0)
    lead, trail = ev.wrap_cut_columns(NF, NT, DT, tol=1e-3)
    assert np.all(frac[lead:NT - trail] <= 1e-3)
    if lead:
        assert frac[lead - 1] > 1e-3
    if trail:
        assert frac[NT - trail] > 1e-3
