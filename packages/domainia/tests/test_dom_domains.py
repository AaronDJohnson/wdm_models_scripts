"""D-A2: domains and cell sets (spec section 4.1; counts from spec sections 4.8-4.9)."""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain

SIN = wdm.Grid(64, 256, 2.0**-10)


def test_time_domain():
    d = TimeDomain(16384, 2.0**-10)
    assert (d.N, d.dt, d.t0, d.size) == (16384, 2.0**-10, 0.0, 16384)
    assert np.array_equal(d.times(), np.arange(16384) * 2.0**-10)        # WT cfg.times()
    assert d == TimeDomain(16384, 2.0**-10) and hash(d) == hash(TimeDomain(16384, 2.0**-10))
    with pytest.raises(ValueError):
        TimeDomain(0, 1.0)
    with pytest.raises(ValueError):
        TimeDomain(8, -1.0)
    with pytest.raises(TypeError):
        TimeDomain(8.0, 1.0)


def test_fourier_domain():
    d = FourierDomain(64, 0.5, 3, 10, "sigma0")
    assert np.array_equal(d.bins(), np.arange(3, 13))
    assert np.array_equal(d.frequencies(), np.arange(3, 13) / 32.0)
    assert d.size == 10
    FourierDomain(64, 0.5, 23, 10, "rfft")                              # last bin is 32 = N/2
    with pytest.raises(ValueError, match="exceed"):
        FourierDomain(64, 0.5, 24, 10, "rfft")
    with pytest.raises(ValueError, match="units"):
        FourierDomain(64, 0.5, 0, 10, "hz")


def test_all_active_is_the_packed_order():
    cells = CellSet.all_active(wdm.Grid(8, 16, 1.0))
    m, n = wdm.grid.packed_cells(8, 16)
    assert np.array_equal(cells.m_array, m) and np.array_equal(cells.n_array, n)
    assert np.array_equal(cells.packed_positions(), np.arange(128))
    assert cells.keep_mask().all()


def test_sinusoid_counts():
    """Spec section 4.8: 16384 cells, cut (6, 5) keeps 15681, edge-exact (8, 7) has 959
    rows over a 15425-cell interior."""
    cells = CellSet.all_active(SIN)
    assert cells.size == 16384
    cut = cells.drop_columns(6, 5)
    assert cut.size == 15681
    edge = cells.columns(8, 7)
    interior = cells.complement(edge)
    assert (edge.size, interior.size) == (959, 15425)
    assert set(edge.n) == set(wdm.grid.edge_columns(256, 8, 7))
    assert cells.complement(interior) == edge
    pad = CellSet.all_active(wdm.Grid(64, 272, 2.0**-10))
    assert pad.size == 17408


def test_gb_counts_from_the_prepared_grids():
    """Spec section 4.9: 3 bands x NT cells per channel; cut (5, 4) and 27 edge rows."""
    for N, NT, bands in ((1576800, 240, (65, 66, 67)), (3153600, 438, (71, 72, 73)),
                         (6307200, 900, (69, 70, 71))):
        grid = wdm.Grid(N // NT, NT, 5.0)
        cells = CellSet.bands(grid, bands)
        assert cells.size == 3 * NT
        assert cells.drop_columns(5, 4).size == 3 * NT - 27
        assert cells.columns(5, 4).size == 27
        assert cells.band_sizes() == tuple((b, NT) for b in bands)
        assert np.array_equal(cells.m_array, np.repeat(bands, NT))      # GB Variant.b order
        assert np.array_equal(cells.n_array, np.tile(np.arange(NT), 3))


def test_positions_and_masks_agree():
    grid = wdm.Grid(8, 16, 0.5)
    cells = CellSet.all_active(grid)
    sub = cells.drop_columns(3, 1)
    pos = sub.packed_positions()
    m, n = wdm.grid.packed_cells(8, 16)
    assert np.array_equal(m[pos], sub.m_array) and np.array_equal(n[pos], sub.n_array)
    assert np.array_equal(np.flatnonzero(sub.keep_mask()), pos)
    assert CellSet.from_mask(grid, sub.keep_mask()) == sub
    assert np.array_equal(cells.index_of(sub), pos)
    with pytest.raises(ValueError, match="subset"):
        sub.index_of(cells)


def test_dc_and_nyquist_bands_have_half_the_cells():
    cells = CellSet.bands(wdm.Grid(8, 16, 1.0), (0, 1, 8))
    assert cells.band_sizes() == ((0, 8), (1, 16), (8, 8))
    assert set(cells.n_array[cells.m_array == 0]) == set(range(0, 16, 2))


def test_validation():
    g = wdm.Grid(8, 16, 1.0)
    with pytest.raises(ValueError, match="not active"):
        CellSet(g, (0,), (1,))                                          # DC, odd n
    with pytest.raises(ValueError, match="band order"):
        CellSet(g, (2, 1), (0, 0))
    with pytest.raises(ValueError, match="band order"):
        CellSet(g, (1, 1), (3, 3))                                      # repeat
    with pytest.raises(ValueError, match="outside"):
        CellSet(g, (9,), (0,))
    with pytest.raises(ValueError, match="ascending"):
        CellSet.bands(g, (3, 2))
    with pytest.raises(ValueError, match="exceeds"):
        CellSet.all_active(g).drop_columns(10, 7)
    # non-integral, non-finite or boolean indices are refused, never truncated (2.9999999 -> 2)
    for m_bad, n_bad in (((2.9999999,), (0,)), ((1,), (0.5,)), ((np.nan,), (0,)), ((True,), (0,))):
        with pytest.raises(ValueError, match="integer"):
            CellSet(g, m_bad, n_bad)
    assert CellSet(g, np.array([1.0, 2.0]), np.array([3.0, 3.0])) == CellSet(g, (1, 2), (3, 3))


def test_identity_is_value_based_and_cheap():
    a = CellSet.all_active(SIN).drop_columns(6, 5)
    b = CellSet.all_active(SIN).drop_columns(6, 5)
    assert a is not b and a == b and hash(a) == hash(b)
    assert {a: 1}[b] == 1
    assert a != CellSet.all_active(SIN)
    assert "size=15681" in repr(a)
    m_in, n_in = np.array([1, 1, 2, 2]), np.array([0, 1, 0, 1])
    c = CellSet(wdm.Grid(8, 16, 1.0), m_in, n_in)
    assert m_in.flags.writeable and n_in.flags.writeable              # caller's arrays untouched
    assert not c.m_array.flags.writeable and c.m == (1, 1, 2, 2)


def test_wdm_domain_and_channels():
    cells = CellSet.all_active(wdm.Grid(8, 16, 1.0))
    d = WDMDomain(cells)
    assert (d.size, d.scale) == (128, 1.0)
    assert WDMDomain(cells, 2.0) != d
    with pytest.raises(ValueError):
        WDMDomain(cells, 0.0)
    ch = ChannelAxis(("A", "E"))
    assert ch.size == 2 and ChannelAxis().size == 0
    with pytest.raises(ValueError, match="unique"):
        ChannelAxis(("A", "A"))
