"""D-C12: section D-C's exact covariances pushed through section D-B's maps (spec §4.2,
"covariance translations"): ``map.push(cov.at(nu)).rows(e)`` is ``PushedRows``' rows bit
for bit, in time (``TimeToWDM``, the sinusoid exact-edge rows) and in Fourier bins
(``FourierToWDM(restrict=True)``, the GB exact-edge rows)."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import exact_covariance

import domtest_inputs as T
from domainia.covariance import ExactFourierCovariance, ExactTimeCovariance, PushedRows, positions
from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.maps import FourierToWDM, TimeToWDM
from domainia.spectra import LogLinearPSD

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(wdm.Grid(NF, NT, DT))
EDGE = CELLS.columns(3, 1)


def test_time_push_of_the_exact_covariance_is_pushed_rows():
    exact = ExactTimeCovariance(PSD, N, DT)
    tmap = TimeToWDM(TimeDomain(N, DT), WDMDomain(CELLS))
    rows = PushedRows(exact, tmap, EDGE)
    want = np.asarray(rows.evaluate(rows.host_constants(), NU, NUMPY))
    e = positions(EDGE, CELLS)
    got = np.asarray(tmap.push(exact.at(NU)).rows(e))
    assert np.array_equal(got, want)
    dense = exact_covariance(NF, NT, DT, lambda f, t: PSD(f, t, NU))
    assert np.max(np.abs(got - dense[e])) < 1e-9 * np.max(np.abs(dense))
    x = np.random.default_rng(1).standard_normal(N)
    assert np.array_equal(exact.at(NU).matvec(x), exact.at(NU).matvec(x[None, :])[0])


def test_fourier_push_of_mprime_is_pushed_rows():
    c = T.gb_case("gb_3mo")
    edge = c.cells.columns(5, 4)
    fmap = FourierToWDM(c.window, WDMDomain(c.cells), restrict=True)
    for ch in range(2):
        ex = ExactFourierCovariance(c.psd, c.window, channel=ch)
        rows = PushedRows(ex, fmap, edge)
        want = np.asarray(rows.evaluate(rows.host_constants(), c.mu[ch], NUMPY))
        got = np.asarray(fmap.push(ex.at(c.mu[ch])).rows(rows.e))
        assert got.shape == (27, len(c.cells.m)) and np.array_equal(got, want), ch


@pytest.mark.jax
def test_jax_push_matches_numpy(jax_be):
    exact = ExactTimeCovariance(PSD, N, DT)
    tmap = TimeToWDM(TimeDomain(N, DT), WDMDomain(CELLS))
    e = positions(EDGE, CELLS)
    ref = np.asarray(tmap.push(exact.at(NU)).rows(e))
    got = tmap.on(jax_be).push(exact.at(NU)).rows(e, jax_be)
    assert type(got).__module__.startswith("jax")
    assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-12 * np.max(np.abs(ref))
