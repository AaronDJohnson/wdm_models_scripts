"""D-A3: the dense oracles of ``domainia.testing`` against independent constructions."""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from wdm.evolutionary import (eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan, exact_covariance,
                              time_domain_covariance)
from domainia import testing as T
from domainia.domains import CellSet

# The WT test_matrix.py configuration: 8 x 16 cells, dt = 1/16, f_min = 0.5, f_ref = 1.
NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
GRID = wdm.Grid(NF, NT, DT)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    """``S(f, t) = (max(f, f_min)/f_ref)^(ac + ad (t/T - 1/2))``, elementwise (A_n = 1)."""
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_dense_W_rows_are_the_atoms(rng):
    cells = CellSet.all_active(GRID)
    W = T.dense_W(cells)
    x = rng.standard_normal(N)
    assert np.max(np.abs(W @ x - wdm.grid.pack(wdm.forward_time(x, NF, NT, DT), NF, NT))) < 1e-12
    assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
    sub = cells.drop_columns(3, 1)
    assert np.array_equal(T.dense_W(sub), W[sub.packed_positions()])


def test_congruence_is_the_exact_covariance():
    Sn = drift(-2.0, 1.2)
    cells = CellSet.all_active(GRID)
    got = T.wdm_congruence(time_domain_covariance(N, DT, Sn), cells)
    assert rel(got, exact_covariance(NF, NT, DT, Sn)) < 1e-12


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_eq45_is_exact_for_stationary_noise(ac):
    """WT test_matrix.py::test_eq45_is_exact_for_stationary_noise."""
    C = T.eq45_dense(CellSet.all_active(GRID), drift(ac, 0.0))
    assert rel(C, exact_covariance(NF, NT, DT, drift(ac, 0.0))) < 1e-10


def test_eq45_differs_from_exact_at_second_order_with_drift():
    """WT test_matrix.py::test_eq45_differs_from_exact_at_second_order_with_drift (the
    test's small edges: 3 leading and 1 trailing column)."""
    cells = CellSet.all_active(GRID)
    C = T.eq45_dense(cells, drift(-2.0, 1.2))
    exact = exact_covariance(NF, NT, DT, drift(-2.0, 1.2))
    interior = ~np.isin(cells.n_array, wdm.grid.edge_columns(NT, 3, 1))
    d = np.diag(exact)[interior]
    assert np.max(np.abs(np.diag(C)[interior] / d - 1)) < 0.05


@pytest.mark.parametrize("cut", [None, (3, 1)])
def test_blocktri_dense_of_the_fast_blocks_is_the_direct_sum(cut):
    """Adapted from WT test_matrix.py::test_fast_blocks_equal_the_direct_sum (here the 8 x 16
    grid and one noise point, through blocktri_dense; WT used its 8 x 64 WIDE grid and three
    points).  The padded-grid case is S1's packages/wdm/tests/test_evolutionary_kernels.py::
    test_bessel_blocks_match_the_direct_sum_and_gram."""
    cells = CellSet.all_active(GRID)
    if cut:
        cells = cells.drop_columns(*cut)
    plan = eq45_plan(NF, NT, DT, cells.keep_mask(), P=24)
    Lf = L_of(plan.f)
    diag, up = eq45_blocks_cheb(plan, lambda k: eq45_coeffs_bessel(Lf[k], -2.0, 1.2, 24, r=1.0))
    C = T.blocktri_dense(diag, up)
    assert np.array_equal(C, C.T)
    assert rel(C, T.eq45_dense(cells, drift(-2.0, 1.2))) < 1e-10


def test_hhat_is_the_dft_of_the_square_root_envelope(rng):
    L = 64
    basis = np.stack([np.cos(2 * np.pi * np.arange(L) / L), np.sin(4 * np.pi * np.arange(L) / L)])
    mu = np.array([0.3, -0.2])
    lags = np.arange(-7, 8)
    t = np.arange(L)
    want = np.array([np.sum(np.exp(0.5 * (mu @ basis)) * np.exp(-2j * np.pi * q * t / L)) / L
                     for q in lags])
    assert np.max(np.abs(T.hhat(mu, basis, lags) - want)) < 1e-15


def test_eq54_is_the_covariance_of_a_modulated_stationary_process(rng):
    """x_t = a_t y_t, y stationary (two-sided PSD S): the bins X = F x have
    E[X_j X_l^*] = sum_k h[j-k] (N S_k) h^*[l-k], h = F a / N.  With a compact h (one
    harmonic) and the grid away from DC and Nyquist, Eq. (54) on the grid is exact."""
    Nr, j0, G, margin = 256, 40, 20, 4
    t = np.arange(Nr)
    a = 1.0 + 0.3 * np.cos(2 * np.pi * 3 * t / Nr)
    k = np.fft.fftfreq(Nr) * Nr
    S = 1.0 / (1.0 + (np.abs(k) / 30.0) ** 2)                            # two-sided, even
    F = np.exp(-2j * np.pi * np.outer(t, t) / Nr)
    Sigma_y = (F.conj().T @ np.diag(S) @ F).real / Nr
    Sigma_x = a[:, None] * Sigma_y * a[None, :]
    want = (F @ Sigma_x @ F.conj().T)[j0:j0 + G, j0:j0 + G]
    hfull = F @ a / Nr                                                  # h[q], q mod Nr
    lags = np.arange(-(G + margin), G + margin + 1)
    h = hfull[lags % Nr]
    src = j0 - margin + np.arange(G + 2 * margin)
    got = T.eq54_covariance(np.zeros(G), Nr * S[src], h, margin=margin)
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    U = rng.standard_normal((6, G)) + 1j * rng.standard_normal((6, G))
    lam = T.dense_lambda_generic(U, np.zeros(G), Nr * S[src], h, margin=margin)
    assert np.array_equal(lam, 0.5 * np.real(U @ got @ U.conj().T))


def test_eq54_with_a_constant_envelope_is_diagonal():
    G, margin = 12, 3
    h = np.zeros(2 * (G + margin) + 1, complex)
    h[G + margin] = 1.0                                                 # lag 0
    sI = np.linspace(1.0, 2.0, G)
    sG = np.linspace(3.0, 5.0, G + 2 * margin)
    M = T.eq54_covariance(sI, sG, h, margin=margin)
    want = np.diag(sI + sG[margin:margin + G])
    assert np.max(np.abs(M - want)) < 1e-15 * np.max(want)                # sqrt(s)^2 = s to 1 ulp
    with pytest.raises(ValueError, match="G \\+ 2 margin"):
        T.eq54_covariance(sI, sG[:-1], h, margin=margin)
