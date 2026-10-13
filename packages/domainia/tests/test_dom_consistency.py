"""Domain and cell consistency at the composition points (post-review fix G-F1).

Equal-sized but different domains or cell sets used to pass every shape check and give a
wrong lnL silently (the review's GB band shift: -1428.33 against the golden -1428.29; the
root-lane regression is ``tests/golden/test_s2_g6_gb.py``).  Now:

* ``GaussianLikelihood`` requires ``data.domain == map.src`` and, per channel,
  ``map.dst.cells == cov.cells`` wherever those attributes exist (``DiagonalCovariance``
  takes its cells from a ``domainia.variance`` variance; ``FlatMarginal`` forwards them);
* ``PushedRows`` requires ``cells=`` to equal the map's ``dst.cells`` when both are given;
* ``SchurEdge`` requires an interior with cells to cover ``rows.cells.complement(edge)``.

Toys without the attributes (raw arrays, ``dom_toys`` maps, callable variances) are not
checked (``test_dom_likelihood.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm

from domainia.covariance import DiagonalCovariance, Eq45Covariance, ExactTimeCovariance, PushedRows, SchurEdge
from domainia.data import TimeSeries
from domainia.density import RealGaussian
from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.likelihood import GaussianLikelihood, Params, StaticSignal
from domainia.maps import TimeToWDM
from domainia.nuisance import FlatMarginal
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
GRID = wdm.Grid(NF, NT, DT)
TIME = TimeDomain(N, DT)
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(GRID)
CUT = CELLS.drop_columns(2, 1)          # 105 cells
OTHER = CELLS.drop_columns(0, 3)        # 105 different cells
EDGE = CELLS.columns(3, 1)              # 32 cells; the interior has 96
PARAMS = Params(("a", "log10_A_n", "alpha_c", "alpha_d"), (0,), (2, 3))
THETA = np.array([[0.0, 0.3, -1.7, 0.9]])


def _lik(cov, *, data_domain=TIME, map_time=TIME, cells=CUT, **kw):
    x = np.random.default_rng(0).standard_normal(N)
    return GaussianLikelihood(TimeSeries(x, data_domain), TimeToWDM(map_time, WDMDomain(cells)), cov,
                              RealGaussian(0.0, "log10_A_n"), signal=StaticSignal(lambda s: np.full(N, s[0])),
                              params=PARAMS, **kw)


def test_equal_sized_cell_sets_differ():
    assert CUT.size == OTHER.size == 105 and CUT != OTHER


@pytest.mark.parametrize("make", [
    lambda c: DiagonalCovariance(BandAveragedVariance(PSD, c), c.size),
    lambda c: DiagonalCovariance(MidpointVariance(PSD, c), c.size),
    lambda c: Eq45Covariance(PSD, c, builder="bessel"),
], ids=["band", "mid", "tri"])
def test_likelihood_refuses_covariance_cells_other_than_the_map_cells(make):
    good = _lik(make(CUT))
    assert np.isfinite(good(THETA)[0])
    assert make(CUT).cells == CUT
    with pytest.raises(ValueError, match="cells"):
        _lik(make(OTHER))
    U = np.random.default_rng(1).standard_normal((CUT.size, 2))
    assert FlatMarginal(make(OTHER), U).cells == OTHER                 # forwarded through the marginal
    with pytest.raises(ValueError, match="cells"):
        _lik(make(OTHER), nuisance=U)
    assert np.isfinite(_lik(make(CUT), nuisance=U)(THETA)[0])


def test_likelihood_refuses_data_on_another_domain():
    shifted = TimeDomain(N, DT, t0=5.0)                                # same N and dt, other record
    cov = DiagonalCovariance(BandAveragedVariance(PSD, CUT), CUT.size)
    with pytest.raises(ValueError, match="domain"):
        _lik(cov, data_domain=shifted)
    with pytest.raises(ValueError, match="domain"):
        _lik(cov, map_time=shifted)
    assert np.isfinite(_lik(cov, data_domain=shifted, map_time=shifted)(THETA)[0])


def test_callable_and_array_variances_carry_no_cells():
    assert DiagonalCovariance(np.ones(3), 3).cells is None
    assert DiagonalCovariance(lambda nu, be: be.asarray(np.ones(3)), 3).cells is None
    assert np.isfinite(_lik(DiagonalCovariance(np.ones(CUT.size), CUT.size))(THETA)[0])


def test_pushed_rows_refuses_cells_other_than_the_map_cells():
    ex = ExactTimeCovariance(PSD, N, DT)
    tmap = TimeToWDM(TIME, WDMDomain(CELLS))
    assert PushedRows(ex, tmap, EDGE, cells=CELLS).cells == CELLS     # agreeing: fine
    with pytest.raises(ValueError, match="cells"):
        PushedRows(ex, TimeToWDM(TIME, WDMDomain(CUT)), CUT.columns(3, 1), cells=OTHER)


def test_schur_edge_refuses_an_interior_on_other_cells():
    rows = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS)
    interior = CELLS.complement(EDGE)
    wrong = CELLS.drop_columns(1, 3)                                   # 96 cells, as the interior
    assert wrong.size == interior.size and wrong != interior
    SchurEdge(Eq45Covariance(PSD, interior, builder="bessel"), rows, EDGE)
    with pytest.raises(ValueError, match="cells"):
        SchurEdge(Eq45Covariance(PSD, wrong, builder="bessel"), rows, EDGE)
    with pytest.raises(ValueError, match="cells"):
        SchurEdge(DiagonalCovariance(BandAveragedVariance(PSD, wrong), wrong.size), rows, EDGE)
    SchurEdge(DiagonalCovariance(np.ones(interior.size), interior.size), rows, EDGE)   # no cells: size only
