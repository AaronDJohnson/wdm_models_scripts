"""D-B2: TimeToWDM and WDMToTime -- dense oracles, adjoints, G6 (bitwise) and JAX parity."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from dom_db_inputs import (SIN_CUT, SIN_DT, SIN_NF, SIN_NT, SIN_PAD_COLUMNS, DenseCov,
                           jax_or_skip, sha256_array, sinusoid_data, sinusoid_g6)
from wdm.matrix import active_W

from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.maps import TimeToWDM, WDMToTime

SMALL = [(8, 16, 0.5), (16, 32, 1.7)]


def _maps(Nf, Nt, dt, lead=0, trail=0):
    g = wdm.Grid(Nf, Nt, dt)
    cells = CellSet.all_active(g)
    if lead or trail:
        cells = cells.drop_columns(lead, trail)
    return g, TimeDomain(g.N, dt), TimeToWDM(TimeDomain(g.N, dt), WDMDomain(cells))


@pytest.mark.parametrize("grid", SMALL)
def test_rows_are_active_W_and_apply_is_the_transform(grid):
    g, _, M = _maps(*grid)
    Wa = active_W(*grid)
    assert M.shape == (g.N, g.N) and M.kind == "real"
    assert np.max(np.abs(M.dense() - Wa)) <= 1e-12
    x = np.random.default_rng(0).standard_normal((3, g.N))
    assert np.max(np.abs(M.apply(x) - x @ Wa.T)) <= 1e-12
    assert np.array_equal(M.apply(x[1]), M.apply(x)[1])          # 1-D == batched row


def test_atoms_have_energy_dt_and_give_the_cells():
    """Ported from WT tests/test_matrix.py:58-64 on its grid (Nf, Nt, dt) = (8, 16, 1/16)."""
    g, _, M = _maps(8, 16, 1 / 16)
    x = np.random.default_rng(3).standard_normal(g.N)
    w = M.apply(x)
    for p in (0, 5, g.N // 2, g.N - 1):
        atom = M.rows(p)[0]
        assert atom @ x == pytest.approx(w[p], abs=1e-12)
        assert atom @ atom == pytest.approx(g.dt, rel=1e-12)


@pytest.mark.parametrize("grid", SMALL)
@pytest.mark.parametrize("cut", [(0, 0), (2, 1)])
def test_adjoint_is_W_transpose_on_any_cell_subset(grid, cut):
    g, _, M = _maps(*grid, *cut)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, g.N))
    y = rng.standard_normal((2, M.shape[0]))
    lhs, rhs = np.sum(M.apply(x) * y), np.sum(x * M.adjoint(y))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)
    assert np.max(np.abs(M.adjoint(y) - y @ M.dense())) <= 1e-12


@pytest.mark.parametrize("grid", SMALL)
def test_push_equals_W_sigma_WT(grid):
    g, _, M = _maps(*grid, 2, 1)
    rng = np.random.default_rng(2)
    A = rng.standard_normal((g.N, g.N))
    S = A @ A.T / g.N
    D = M.dense()
    ref = D @ S @ D.T
    P = M.push(DenseCov(S))
    assert np.max(np.abs(P.dense() - ref)) <= 1e-12 * np.max(np.abs(ref))
    assert np.max(np.abs(P.rows([3, 7]) - ref[[3, 7]])) <= 1e-12 * np.max(np.abs(ref))


def test_push_of_an_exact_covariance_matches_wdm_exact_covariance():
    """wdm.evolutionary.exact_covariance (dense W Sigma W^T) is the S1 oracle."""
    Nf, Nt, dt = 8, 16, 0.5
    g, _, M = _maps(Nf, Nt, dt)
    Sn = lambda f, t: (np.maximum(f, 0.05) / 1.0) ** (-1.0 + 0.8 * (t / (g.N * dt) - 0.5))  # noqa: E731
    Sigma = wdm.evolutionary.time_domain_covariance(g.N, dt, Sn)
    ref = wdm.evolutionary.exact_covariance(Nf, Nt, dt, Sn)
    got = M.push(DenseCov(Sigma)).dense()
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("grid", SMALL)
def test_wdm_to_time_is_the_inverse_on_complete_sets(grid):
    g, td, M = _maps(*grid)
    Mi = WDMToTime(M.dst, td)
    x = np.random.default_rng(3).standard_normal((2, g.N))
    assert np.max(np.abs(Mi.apply(M.apply(x)) - x)) <= 1e-12
    assert np.max(np.abs(Mi.dense() - active_W(*grid).T / grid[2])) <= 1e-12
    y = np.random.default_rng(4).standard_normal((2, g.N))
    assert abs(np.sum(Mi.apply(y) * x) - np.sum(y * Mi.adjoint(x))) <= 1e-12 * np.sum(np.abs(x))
    _, _, Mc = _maps(*grid, 2, 1)
    with pytest.raises(ValueError, match="complete cell sets"):
        WDMToTime(Mc.dst, td)


def test_mismatched_grid_is_refused():
    g = wdm.Grid(8, 16, 0.5)
    with pytest.raises(ValueError, match="does not match"):
        TimeToWDM(TimeDomain(g.N, 1.0), WDMDomain(CellSet.all_active(g)))


def _sinusoid_maps():
    g = wdm.Grid(SIN_NF, SIN_NT, SIN_DT)
    td = TimeDomain(g.N, SIN_DT)
    cells = CellSet.all_active(g)
    return td, TimeToWDM(td, WDMDomain(cells)), TimeToWDM(td, WDMDomain(cells.drop_columns(*SIN_CUT)))


def test_g6_sinusoid_cells_are_bitwise():
    """y_data of band_raw/band_exact (16384 cells) and band_cut (15681): WT Variant.map."""
    meta, _ = sinusoid_g6()
    d = sinusoid_data()
    _, raw, cut = _sinusoid_maps()
    y = raw.apply(d)
    assert y.shape == (16384,)
    assert sha256_array(y) == meta["maps"]["band_raw"]["y_data"]["sha256"]
    assert sha256_array(y) == meta["maps"]["band_exact"]["y_data"]["sha256"]
    yc = cut.apply(d)
    assert yc.shape == (15681,)
    assert sha256_array(yc) == meta["maps"]["band_cut"]["y_data"]["sha256"]


def test_g6_sinusoid_padding_impulses_Wp_are_bitwise():
    """``Wp`` (17408 x 1024): the padded grid's transforms of the 1024 padding samples
    (WT MatrixWdm, matrix.py:782-784) -- batched forward_time_many + band-order packing."""
    meta, _ = sinusoid_g6()
    Ntp = SIN_NT + SIN_PAD_COLUMNS
    g = wdm.Grid(SIN_NF, Ntp, SIN_DT)
    ext = TimeToWDM(TimeDomain(g.N, SIN_DT), WDMDomain(CellSet.all_active(g)))
    n_pad = g.N - SIN_NF * SIN_NT
    unit = np.zeros((n_pad, g.N))
    unit[np.arange(n_pad), SIN_NF * SIN_NT + np.arange(n_pad)] = 1.0
    Wp = np.ascontiguousarray(ext.apply(unit).T)
    assert Wp.shape == (17408, 1024)
    assert sha256_array(Wp) == meta["maps"]["Wp"]["sha256"]


@pytest.mark.jax
def test_jax_parity_under_jit_with_the_map_as_argument():
    jax, JB = jax_or_skip()
    td, raw, cut = _sinusoid_maps()
    d = sinusoid_data()
    for M in (raw, cut):
        f = jax.jit(lambda m, x: m.apply(x))
        out = f(M.on(JB), JB.asarray(d))
        assert isinstance(out, jax.Array)
        ref = M.apply(d)
        assert np.max(np.abs(np.asarray(out) - ref)) <= 1e-13 * np.max(np.abs(ref))
    y = raw.apply(d)
    back = jax.jit(lambda m, v: m.adjoint(v))(raw.on(JB), JB.asarray(y))
    assert isinstance(back, jax.Array)
    assert np.max(np.abs(np.asarray(back) - raw.adjoint(y))) <= 1e-13 * np.max(np.abs(d))
    Mi = WDMToTime(raw.dst, td)
    inv = jax.jit(lambda m, v: m.apply(v))(Mi.on(JB), JB.asarray(y))
    assert np.max(np.abs(np.asarray(inv) - d)) <= 1e-12 * np.max(np.abs(d))
