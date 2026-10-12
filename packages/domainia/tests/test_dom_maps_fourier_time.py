"""D-B3: TimeToFourier, CropTime, ZeroPadTime -- the Whittle bins, the crop, the pad."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from dom_db_inputs import (SIN_CUT, SIN_DT, SIN_FMIN, SIN_NF, SIN_NT, SIN_PAD_COLUMNS, DenseCov,
                           jax_or_skip, sinusoid_data)

from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import Compose, CropTime, TimeToFourier, TimeToWDM, ZeroPadTime

N, DT = SIN_NF * SIN_NT, SIN_DT


def _oracle_keep(n: int, fmin: float) -> np.ndarray:
    """WT FourierBand / MatrixWhittle keep, verbatim (band.py:17-26, matrix.py:883-884)."""
    f = np.fft.rfftfreq(n, d=DT)
    keep = (f >= max(fmin, 0.0)) & (f < 0.5 / DT)
    keep[[0, -1]] = False
    return keep


def test_sinusoid_whittle_raw_bins_are_bitwise():
    d = sinusoid_data()
    F = TimeToFourier(TimeDomain(N, DT), fmin=SIN_FMIN)
    assert F.dst == FourierDomain(N, DT, 64, 8128, "rfft") and F.kind == "analysis"
    assert 2 * F.dst.n == 16256                        # spec §4.6: 16256 kept, 128 discarded
    R = F.apply(d)
    assert np.array_equal(R, np.fft.rfft(d)[_oracle_keep(N, SIN_FMIN)])


def test_sinusoid_whittle_cut_crop_and_bins_are_bitwise():
    d = sinusoid_data()
    lo, hi = SIN_CUT[0] * SIN_NF, (SIN_NT - SIN_CUT[1]) * SIN_NF
    crop = CropTime(TimeDomain(N, DT), lo, hi - lo)
    assert (lo, hi - lo) == (384, 15680)
    assert crop.dst == TimeDomain(15680, DT, 384 * DT)
    F = TimeToFourier(crop.dst, fmin=SIN_FMIN)
    W = Compose(F, crop)
    R = W.apply(d)
    assert np.array_equal(R, np.fft.rfft(d[slice(lo, hi)])[_oracle_keep(15680, SIN_FMIN)])
    assert (F.dst.k0, F.dst.n) == (62, 7778)


def test_fourier_rows_adjoint_and_orthonormal_coordinates():
    td = TimeDomain(128, 0.5)
    F = TimeToFourier(td, fmin=0.2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 128))
    Fd = F.dense()
    assert np.max(np.abs(x @ Fd.T - F.apply(x))) <= 1e-12
    y = rng.standard_normal((3, F.dst.n)) + 1j * rng.standard_normal((3, F.dst.n))
    lhs = np.sum(np.real(np.conj(y) * F.apply(x)))
    assert abs(lhs - np.sum(x * F.adjoint(y))) <= 1e-12 * np.sum(np.abs(y))
    O = F.orthonormal_rows()
    assert np.max(np.abs(O @ O.T - np.eye(2 * F.dst.n))) <= 1e-14
    R = F.apply(x[0])
    coords = np.sqrt(2.0 / 128) * np.stack([R.real, -R.imag], axis=1).ravel()
    assert np.max(np.abs(O @ x[0] - coords)) <= 1e-14


def test_push_of_a_stationary_covariance_is_the_whittle_diagonal():
    """F Sigma F^H = diag(N S(f_k) / (2 dt)) for the periodic stationary Sigma (WT FreqWhittle)."""
    n, dt = 128, 0.5
    F = TimeToFourier(TimeDomain(n, dt), fmin=0.2)
    Sn = lambda f, t: np.maximum(f, 0.05) ** -1.5 + 0.0 * t  # noqa: E731
    Sigma = wdm.evolutionary.time_domain_covariance(n, dt, Sn)
    P = F.push(DenseCov(Sigma)).dense()
    fk = np.fft.rfftfreq(n, dt)[F.dst.k0:F.dst.k0 + F.dst.n]
    assert np.max(np.abs(np.real(np.diag(P)) / (n * Sn(fk, 0.0) / (2 * dt)) - 1)) <= 1e-13
    assert np.max(np.abs(P - np.diag(np.diag(P)))) <= 1e-13 * np.max(np.abs(P))


def test_crop_and_zero_pad_adjoints():
    td = TimeDomain(96, 0.25, 1.0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 96))
    C = CropTime(td, 10, 40)
    y = rng.standard_normal((2, 40))
    assert np.sum(C.apply(x) * y) == pytest.approx(np.sum(x * C.adjoint(y)), rel=1e-14)
    assert C.dst.t0 == 1.0 + 10 * 0.25
    with pytest.raises(ValueError, match="not inside"):
        CropTime(td, 80, 40)
    Z = ZeroPadTime(td, 32)
    assert Z.dst == TimeDomain(128, 0.25, 1.0)
    z = Z.apply(x)
    assert np.array_equal(z[:, :96], x) and np.all(z[:, 96:] == 0)
    w = rng.standard_normal((2, 128))
    assert np.sum(z * w) == pytest.approx(np.sum(x * Z.adjoint(w)), rel=1e-14)


def test_sinusoid_pad_record_is_bitwise_the_oracle_extension():
    """``TimeToWDM(extended) . ZeroPadTime`` = WT ``PaddedMarginal.solve``'s
    ``g.forward_many(ext)`` with ``ext[:, :n] = X`` (matrix.py:640-643)."""
    td = TimeDomain(N, DT)
    Z = ZeroPadTime(td, SIN_PAD_COLUMNS * SIN_NF)
    g = wdm.Grid(SIN_NF, SIN_NT + SIN_PAD_COLUMNS, DT)
    M = Compose(TimeToWDM(Z.dst, WDMDomain(CellSet.all_active(g))), Z)
    X = np.random.default_rng(2).standard_normal((3, N))
    ext = np.zeros((3, g.N))
    ext[:, :N] = X
    ref = wdm.forward_time_many(ext, g.Nf, g.Nt, DT)[:, wdm.active_mask(g.Nf, g.Nt)]
    assert np.array_equal(M.apply(X), ref)


@pytest.mark.jax
def test_jax_parity():
    jax, JB = jax_or_skip()
    d = sinusoid_data()
    F = TimeToFourier(TimeDomain(N, DT), fmin=SIN_FMIN)
    out = jax.jit(lambda m, x: m.apply(x))(F, JB.asarray(d))
    assert isinstance(out, jax.Array)
    ref = F.apply(d)
    assert np.max(np.abs(np.asarray(out) - ref)) <= 1e-13 * np.max(np.abs(ref))
    Z = ZeroPadTime(TimeDomain(N, DT), 1024)
    C = CropTime(TimeDomain(N, DT), 384, 15680)
    for M in (Z, C):
        o = jax.jit(lambda m, x: m.apply(x))(M, JB.asarray(d))
        assert isinstance(o, jax.Array) and np.array_equal(np.asarray(o), M.apply(d))
