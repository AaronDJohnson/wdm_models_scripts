"""D-B7: BandAveragedVariance and MidpointVariance -- sinusoid G6 bitwise, GB G6, JAX."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from dom_db_inputs import (GB_CASES, SIN_CUT, SIN_DT, SIN_FMIN, SIN_NF, SIN_NT, SIN_PAD_COLUMNS,
                           SIN_T, TableEnvelope, gb_case, jax_or_skip, same_build, sha256_array,
                           sinusoid_g6, to_float)

from domainia.domains import CellSet, FourierDomain
from domainia.spectra import GenericPSD, LogLinearPSD, SeparablePSD, TabulatedSpectrum
from domainia.variance import BandAveragedVariance, MidpointVariance

PSD = LogLinearPSD(f_floor=SIN_FMIN, f_ref=1.0, T=SIN_T)
#: GB variances use a 5-term BLAS dot (mu @ e(t)) and a table envelope: compared by value.
GB_RTOL = 1e-14


def _truth(meta):
    return to_float(meta["alpha_c"]), to_float(meta["alpha_d"])


def _cells(Nt):
    return CellSet.all_active(wdm.Grid(SIN_NF, Nt, SIN_DT))


def test_g6_sinusoid_raw_variances_are_bitwise():
    meta, arrays = sinusoid_g6()
    cells = _cells(SIN_NT)
    vb = BandAveragedVariance(PSD, cells)(_truth(meta))
    vm = MidpointVariance(PSD, cells)(_truth(meta))
    assert vb.shape == vm.shape == (16384,)
    assert np.array_equal(vb, arrays["band_variance_raw"])
    assert np.array_equal(vm, arrays["mid_variance_raw"])
    assert sha256_array(vb) == meta["variances"]["band_raw"]["sha256"]
    assert sha256_array(vm) == meta["variances"]["mid_raw"]["sha256"]


def test_g6_sinusoid_pad_variances_are_bitwise():
    """The padded grid (64 x 272): f_off from N' dt, alpha from the DATA T = 16 s."""
    meta, _ = sinusoid_g6()
    cells = _cells(SIN_NT + SIN_PAD_COLUMNS)
    vb = BandAveragedVariance(PSD, cells)(_truth(meta))
    vm = MidpointVariance(PSD, cells)(_truth(meta))
    assert vb.shape == (17408,)
    assert sha256_array(vb) == meta["variances"]["band_pad"]["sha256"]
    assert sha256_array(vm) == meta["variances"]["mid_pad"]["sha256"]
    wrong_T = LogLinearPSD(SIN_FMIN, 1.0, 272 * SIN_NF * SIN_DT)
    assert sha256_array(BandAveragedVariance(wrong_T, cells)(_truth(meta))) \
        != meta["variances"]["band_pad"]["sha256"]


def test_cut_cells_are_the_full_vector_restricted():
    meta, arrays = sinusoid_g6()
    cells = _cells(SIN_NT)
    cut = cells.drop_columns(*SIN_CUT)
    keep = ~np.isin(np.asarray(cells.n), wdm.grid.edge_columns(SIN_NT, *SIN_CUT))
    assert np.array_equal(BandAveragedVariance(PSD, cut)(_truth(meta)),
                          arrays["band_variance_raw"][keep])
    assert np.array_equal(MidpointVariance(PSD, cut)(_truth(meta)),
                          arrays["mid_variance_raw"][keep])


def test_variances_reproduce_the_wdm_noise_grids():
    """Ported from WT tests/test_matrix.py:67-75 on its grid (8, 16, 1/16), f_min = 0.5:
    the WT noise grids are wdm.noise.evolutionary_band_variance / evolutionary_variance."""
    g = wdm.Grid(8, 16, 1 / 16)
    psd = LogLinearPSD(0.5, 1.0, g.N * g.dt)
    cells = CellSet.all_active(g)
    mask = wdm.active_mask(g.Nf, g.Nt)
    for nu in ((-2.0, 1.2), (-1.1, -2.5)):
        band = wdm.noise.evolutionary_band_variance(g.Nf, g.Nt, g.dt, psd.bind(nu))[mask]
        mid = wdm.evolutionary_variance(g.Nf, g.Nt, g.dt,
                                        lambda f, t: psd(f[:, None], t[None, :], nu))[mask]
        assert np.allclose(BandAveragedVariance(psd, cells)(nu), band, rtol=1e-13, atol=0)
        assert np.allclose(MidpointVariance(psd, cells)(nu), mid, rtol=1e-13, atol=0)


def test_generic_psd_reproduces_the_loglinear_path():
    """wdm.noise.evolutionary_band_variance on the bound PSD == the WT band_variance port."""
    meta, arrays = sinusoid_g6()
    cells = _cells(SIN_NT)
    gen = GenericPSD(lambda f, t, nu: PSD(f, t, nu))
    assert np.array_equal(BandAveragedVariance(gen, cells)(_truth(meta)), arrays["band_variance_raw"])
    assert np.array_equal(MidpointVariance(gen, cells)(_truth(meta)), arrays["mid_variance_raw"])


def _gb_psd(c):
    p = c["prep"]
    store = FourierDomain(c["N"], c["DT"], int(p["store_offset"]), int(p["store_n"]), "sigma0")
    env = TableEnvelope(c["g6"]["basis_half"], c["grid"].dT)
    return SeparablePSD(((TabulatedSpectrum(store, p["store_sI"]), None),
                         (TabulatedSpectrum(store, p["store_sG"]), env)), ("A", "E"))


@pytest.mark.parametrize("case", GB_CASES)
def test_g6_gb_band_means(case):
    """GB band_psd (matrix.py:470-483): the band-centre values are an index (bitwise
    everywhere); the phi^2-weighted means are ``s[:, window] @ w2``, a BLAS gemv -- bitwise
    on the recording build, 1e-14 of max elsewhere."""
    c = gb_case(case)
    cells = CellSet.bands(c["grid"], c["bands"])
    psd = _gb_psd(c)
    NT = c["NT"]
    exact_band = same_build(c["meta"]["build"])
    band, mid = BandAveragedVariance(psd, cells), MidpointVariance(psd, cells)
    for how, v in (("band", band), ("mid", mid)):
        for j, name in enumerate(("sI", "sG")):
            got, ref = v.means[j][:, ::NT], c["g6"][f"bandpsd_{how}_{name}"]
            assert got.shape == ref.shape
            if how == "mid" or exact_band:
                assert np.array_equal(got, ref)
            else:
                assert np.max(np.abs(got - ref)) <= 1e-14 * np.max(np.abs(ref))


@pytest.mark.parametrize("case", GB_CASES)
def test_g6_gb_variances_at_the_anchor(case):
    c = gb_case(case)
    cells = CellSet.bands(c["grid"], c["bands"])
    psd = _gb_psd(c)
    mu = c["prep"]["mu_fit"]
    for name, cls in (("var_band_raw", BandAveragedVariance), ("var_mid_raw", MidpointVariance)):
        v = cls(psd, cells)
        got = np.stack([v(mu[ch], channel=ch) for ch in range(2)])
        ref = c["g6"][name]
        assert got.shape == ref.shape
        assert np.max(np.abs(got / ref - 1.0)) <= GB_RTOL


def test_separable_spectra_must_be_on_the_cells_record():
    c = gb_case("gb_3mo")
    g = wdm.Grid(c["N"] // c["NT"], c["NT"] + 16, c["DT"])        # the padded record
    with pytest.raises(ValueError, match="tabulated on an N="):
        BandAveragedVariance(_gb_psd(c), CellSet.bands(g, c["bands"]))


@pytest.mark.jax
def test_jax_parity_under_jit():
    jax, JB = jax_or_skip()
    meta, arrays = sinusoid_g6()
    cells = _cells(SIN_NT)
    nu = JB.asarray(list(_truth(meta)))
    for cls, key in ((BandAveragedVariance, "band_variance_raw"), (MidpointVariance, "mid_variance_raw")):
        v = cls(PSD, cells).on(JB)
        out = jax.jit(lambda var, p: var(p))(v, nu)
        assert isinstance(out, jax.Array)
        assert np.max(np.abs(np.asarray(out) / arrays[key] - 1.0)) <= 1e-13
    c = gb_case("gb_3mo")
    gcells = CellSet.bands(c["grid"], c["bands"])
    mu = c["prep"]["mu_fit"]
    for cls in (BandAveragedVariance, MidpointVariance):
        v = cls(_gb_psd(c), gcells)
        out = jax.jit(lambda var, m: var(m, channel=1))(v.on(JB), JB.asarray(mu[1]))
        assert isinstance(out, jax.Array)
        assert np.max(np.abs(np.asarray(out) / v(mu[1], channel=1) - 1.0)) <= 1e-13
