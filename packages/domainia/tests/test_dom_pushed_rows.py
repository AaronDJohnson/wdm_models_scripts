"""D-C7: PushedRows -- exact covariance rows pushed to WDM cells.

* Time (sinusoid): the rows are the exact WDM covariance's rows (dense oracle,
  ``wdm.evolutionary.exact_covariance``); the strategies fixed / adaptive / phi_cache /
  staged agree; the adaptive order is a deterministic function of alpha_d with tail
  bound < 1e-14.  G6: 10 picked rows of the 959 at the truth (all 959 under ``slow``).
* Fourier (GB): 1/2 Re(U_E M' U^H) on the window bins; G6 exact-row fingerprints; the
  periodic 12-month record has none (G6 ``V_exact_rows`` is empty).
* JAX: the WT order mapped over atoms and the staged Phi path, against NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import exact_covariance

import domtest_inputs as T
from domainia.covariance import (CHEB_TOL, ExactFourierCovariance, ExactTimeCovariance, PushedRows, cheb_order,
                                 cheb_tail, positions)
from domainia.domains import CellSet
from domainia.spectra import LogLinearPSD

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
NU = np.array([-1.7, 0.9])
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(wdm.Grid(NF, NT, DT))
EDGE = CELLS.columns(3, 1)


def _rows(**kw) -> np.ndarray:
    pr = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, **kw)
    return np.asarray(pr.evaluate(pr.constants(NUMPY), NU, NUMPY))


def test_time_rows_are_the_exact_covariance_rows():
    exact = exact_covariance(NF, NT, DT, lambda f, t: PSD(f, t, NU))
    rows = _rows()
    e = positions(EDGE, CELLS)
    assert rows.shape == (e.size, N)
    assert np.max(np.abs(rows - exact[e])) < 1e-9 * np.max(np.abs(exact))


def test_time_strategies_agree():
    ref = _rows()
    for kw in ({"order": "adaptive"}, {"phi_cache": True, "chunk": 8},
               {"order": "adaptive", "phi_cache": True, "staged": True, "chunk": 5}):
        assert np.max(np.abs(_rows(**kw) - ref)) <= 1e-13 * np.max(np.abs(ref)), kw


def test_adaptive_order_is_deterministic_and_bounded():
    L_max = 8.0
    orders = [cheb_order(ad, L_max) for ad in (0.0, 0.3, -0.3, 1.2, 2.0, 3.9)]
    assert orders == [cheb_order(ad, L_max) for ad in (0.0, 0.3, -0.3, 1.2, 2.0, 3.9)]
    assert orders[1] == orders[2] and all(P % 2 == 0 and 8 <= P <= 24 for P in orders)
    assert orders == sorted(orders)
    for ad in (0.3, 1.2, 2.0):
        P = cheb_order(ad, L_max)
        x = 0.25 * abs(ad) * L_max
        assert cheb_tail(P, x) < CHEB_TOL
        if P > 8:
            assert cheb_tail(P - 2, x) >= CHEB_TOL * 0.5


def test_rows_must_be_destination_cells():
    with pytest.raises(ValueError, match="destination"):
        PushedRows(ExactTimeCovariance(PSD, N, DT), None, CellSet.all_active(wdm.Grid(NF, 32, DT)).columns(1, 0),
                   cells=CELLS)
    with pytest.raises(ValueError, match="grid"):
        PushedRows(ExactTimeCovariance(PSD, 2 * N, DT), None, EDGE, cells=CELLS)
    with pytest.raises(TypeError):
        PushedRows(object(), None, EDGE, cells=CELLS)
    # the time-only guards fail at construction / call, not inside the transforms
    ex = ExactTimeCovariance(PSD, N, DT)
    with pytest.raises(ValueError, match="map"):
        PushedRows(ex, None, EDGE)
    with pytest.raises(ValueError, match="form='operator'"):
        PushedRows(ExactTimeCovariance(PSD, N, DT, form="dense"), None, EDGE, cells=CELLS)
    with pytest.raises(ValueError, match="phi_cache=True"):
        PushedRows(ex, None, EDGE, cells=CELLS, staged=True)
    with pytest.raises(ValueError, match="chunk"):
        PushedRows(ex, None, EDGE, cells=CELLS, phi_cache=True, chunk=0)
    with pytest.raises(ValueError, match="at least one"):
        PushedRows(ex, None, CELLS.columns(0, 0), cells=CELLS)
    pr = PushedRows(ex, None, EDGE, cells=CELLS)
    for P in (0, ex.P + 1):
        with pytest.raises(ValueError, match=r"P must be in \[1, 24\]"):
            pr.evaluate(pr.host_constants(), NU, NUMPY, P=P)
    with pytest.raises(ValueError, match="staged=True"):
        pr.evaluate_staged(pr.host_constants(), NU, NUMPY, 8)


def _sin_rows() -> tuple[PushedRows, dict]:
    cells = CellSet.all_active(wdm.Grid(T.SIN_NF, T.SIN_NT, T.SIN_DT))
    ex = ExactTimeCovariance(LogLinearPSD(f_floor=4.0, f_ref=1.0, T=T.SIN_T), T.SIN_N, T.SIN_DT, P=24)
    return PushedRows(ex, None, cells.columns(8, 7), cells=cells), T.sinusoid_mid()


def test_g6_picked_exact_rows():
    """The 10 golden rows of the 959 (each row is computed independently, so the subset is exact)."""
    pr, s = _sin_rows()
    pick = s["arrays"]["exact_rows_pick"]
    assert pr.e.size == s["g6"]["exact_rows"]["n_edge"] == 959
    assert T.sha256(pr.e) == s["g6"]["exact_rows"]["edge_cells_sha256"]
    sub = PushedRows(pr.exact, None, CellSet(pr.cells.grid, tuple(np.asarray(pr.cells.m)[pr.e][pick]),
                                             tuple(np.asarray(pr.cells.n)[pr.e][pick])), cells=pr.cells)
    rows = np.asarray(sub.evaluate(sub.host_constants(), T.SIN_NU, NUMPY))
    want = s["arrays"]["exact_rows_sample"]
    assert np.max(np.abs(rows - want)) <= 1e-12 * np.max(np.abs(want))
    assert np.allclose(np.linalg.norm(rows, axis=1), s["arrays"]["exact_rows_l2"][pick], rtol=1e-12, atol=0)


@pytest.mark.slow
def test_g6_all_exact_rows():
    pr, s = _sin_rows()
    rows = np.asarray(pr.evaluate(pr.host_constants(), T.SIN_NU, NUMPY))
    T.assert_summary(rows, {**s["g6"]["exact_rows"]})
    assert np.allclose(np.linalg.norm(rows, axis=1), s["arrays"]["exact_rows_l2"], rtol=1e-12, atol=0)


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo"])
def test_g6_gb_exact_rows(case):
    c = T.gb_case(case)
    edge = c.cells.columns(5, 4)
    U = c.atoms(c.cells, c.window.k0, c.window.n)
    for ch in range(2):
        pr = PushedRows(ExactFourierCovariance(c.psd, c.window, channel=ch), None, edge, cells=c.cells, atoms=U)
        assert np.array_equal(pr.e, c.A["V_exact_rows"])
        R = np.asarray(pr.evaluate(pr.host_constants(), c.mu[ch], NUMPY))
        T.assert_summary(R, c.g6["fingerprints"][f"exact_rows_c{ch}"])
        sel = c.g6["scalars"]["exact_rows_selected"]
        assert np.max(np.abs(R[sel] - c.A[f"exact_rows_c{ch}_sel"])) <= 1e-12 * np.max(np.abs(R))
    with pytest.raises(ValueError, match="atoms"):
        PushedRows(ExactFourierCovariance(c.psd, c.window), None, edge, cells=c.cells, atoms=U[:, :-1])
    # the map's atoms must sit on the exact covariance's bins; the time options do not apply
    from domainia.domains import FourierDomain, WDMDomain
    from domainia.maps import FourierToWDM

    w = c.window
    shifted = FourierToWDM(FourierDomain(w.N, w.dt, w.k0 + 1, w.n, w.units), WDMDomain(c.cells), restrict=True)
    with pytest.raises(ValueError, match="src"):
        PushedRows(ExactFourierCovariance(c.psd, w), shifted, edge)
    with pytest.raises(ValueError, match="FourierToWDM map or atoms"):
        PushedRows(ExactFourierCovariance(c.psd, w), None, edge, cells=c.cells)
    with pytest.raises(ValueError, match="time rows only"):
        PushedRows(ExactFourierCovariance(c.psd, w), None, edge, cells=c.cells, atoms=U, order="adaptive")


def test_g6_gb_12mo_has_no_exact_rows():
    """The periodic 12-month record keeps every column (no lead/trail edge columns: the
    registry's rule, test-local here), so its exact-edge rows are empty: the golden
    ``V_exact_rows`` is the empty int64 array, which is what the edge set
    ``cells.columns(0, 0)`` gives, and the exact variant's cells are all the cells."""
    c = T.gb_case("gb_12mo")
    want = c.A["V_exact_rows"]
    edge = c.cells.columns(0, 0)
    got = positions(edge, c.cells)
    assert want.shape == got.shape == (0,) and got.dtype == want.dtype == np.int64
    assert c.cells.complement(edge) == c.cells
    assert c.g6["scalars"]["V_exact"]["P"] == len(c.cells.m) == 3 * c.NT


@pytest.mark.jax
@pytest.mark.parametrize("kw", [{}, {"order": "adaptive", "phi_cache": True, "staged": True, "chunk": 8}])
def test_jax_rows_match_numpy(jax_be, kw):
    import jax.numpy as jnp

    pr = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, **kw)
    ref = np.asarray(pr.evaluate(pr.constants(NUMPY), NU, NUMPY))
    C = pr.constants(jax_be)
    P = pr.order_for(NU)
    got = (pr.evaluate_staged(C, jnp.asarray(NU), jax_be, P) if pr.staged
           else pr.evaluate(C, jnp.asarray(NU), jax_be, P=P))
    assert type(got).__module__.startswith("jax")
    assert np.max(np.abs(np.asarray(got) - ref)) <= 1e-12 * np.max(np.abs(ref))


def _phi_reference(pr, C):
    """S2's host build of Phi (one stacked array; ``scipy.fft.rfft(workers=4)``)."""
    import scipy.fft

    atoms, T = C["atoms"], C["T"]
    pad = (-atoms.shape[0]) % pr.chunk
    A = np.concatenate([atoms, np.zeros((pad, atoms.shape[1]))]).reshape(-1, pr.chunk, atoms.shape[1])
    return np.stack([scipy.fft.rfft(T[None] * a[:, None, :], axis=-1, workers=4) for a in A])


def test_phi_is_built_per_chunk_on_the_backend():
    """S4 deferral (c): Phi is not a host constant; ``complete_constants`` builds it chunk by
    chunk on the backend (NumPy: the S2 numbers bit for bit, as a tuple of chunks)."""
    pr = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, phi_cache=True, chunk=5)
    assert "Phi" not in pr.host_constants()
    C = pr.constants(NUMPY)
    want = _phi_reference(pr, C)
    assert isinstance(C["Phi"], tuple) and len(C["Phi"]) == -(-EDGE.size // 5) == want.shape[0]
    for got, ref in zip(C["Phi"], want, strict=True):
        assert np.array_equal(got, ref)
    assert pr.constants(NUMPY) is C                                      # cached per backend


@pytest.mark.jax
def test_phi_chunks_live_on_the_device(jax_be):
    import jax

    pr = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, phi_cache=True, chunk=5)
    Cj, Cn = pr.constants(jax_be), pr.constants(NUMPY)
    assert all(isinstance(c, jax.Array) for c in Cj["Phi"])
    scale = max(float(np.max(np.abs(c))) for c in Cn["Phi"])
    assert max(float(np.max(np.abs(np.asarray(a) - b))) for a, b in zip(Cj["Phi"], Cn["Phi"], strict=True)) \
        <= 1e-14 * scale
