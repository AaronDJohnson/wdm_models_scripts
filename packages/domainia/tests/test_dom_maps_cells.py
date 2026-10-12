"""D-B5: FourierToWDM (dense golden, fft opt-in, restrict, after) -- GB G6 V.C and cells."""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from dom_db_inputs import (GB_CASES, GB_CUT, GB_PAD_COLUMNS, DenseCov, check_blas_golden,
                           fingerprint_gap, gb_case, jax_or_skip, sha256_array)

from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import Compose, FourierCrop, FourierToWDM, FourierZeroPad, TimeToWDM

#: tolerance of a BLAS-touched G6 array compared by value (relative to its max |value|)
G6_RTOL = 1e-13


def _band_pos(c, cells):
    """GB ``V.b``: each cell's band position among the analysis bands (int64)."""
    return np.searchsorted(np.asarray(c["bands"]), np.asarray(cells.m)).astype(np.int64)


def _store(c):
    p = c["prep"]
    return FourierDomain(c["N"], c["DT"], int(p["store_offset"]), int(p["store_n"]), "sigma0")


def _raw(c, impl="dense"):
    return FourierToWDM(_store(c), WDMDomain(CellSet.bands(c["grid"], c["bands"])), impl=impl)


def _pad(c):
    """``FourierToWDM(padded window) . FourierZeroPad(store)`` as one dense matrix (GB pad)."""
    g = wdm.Grid(c["N"] // c["NT"], c["NT"] + GB_PAD_COLUMNS, c["DT"])
    kept = CellSet.bands(g, c["bands"]).drop_columns(0, GB_PAD_COLUMNS)    # n < NT
    h = g.Nt // 2
    b0, b1 = c["bands"][0], c["bands"][-1]
    window = FourierDomain(g.N, c["DT"], h * b0 - h, h * (b1 - b0) + g.Nt, "sigma0")
    zp = FourierZeroPad(_store(c), window)
    return FourierToWDM(window, WDMDomain(kept)), zp


@pytest.mark.parametrize("case", GB_CASES)
def test_g6_cell_matrices_raw_cut_exact(case):
    """``V.C`` / ``V.U`` carry ``1/np.linalg.norm(phi)`` (a BLAS ddot): sha256 on the
    recording build, fingerprint values elsewhere.  The cell orders ``V.b``, ``V.n`` are
    integers: sha256 everywhere."""
    c = gb_case(case)
    fps, sc, build = c["meta"]["fingerprints"], c["meta"]["scalars"], c["meta"]["build"]
    M = _raw(c)
    assert (M.lo, M.lo + M.width) == tuple(sc["V_raw"]["cols"])     # the 480-bin band window
    assert M.src.k0 + M.lo == sc["V_raw"]["g0"] and M.width == sc["V_raw"]["G"]
    for v in ("V_raw", "V_exact"):                                   # C = U for raw and exact
        assert check_blas_golden(fps[f"{v}_C"], M.C, build, G6_RTOL)
        assert check_blas_golden(fps[f"{v}_U"], M.C, build, G6_RTOL)
        assert sha256_array(_band_pos(c, M.dst.cells)) == fps[f"{v}_b"]["sha256"]
        assert sha256_array(np.asarray(M.dst.cells.n, dtype=np.int64)) == fps[f"{v}_n"]["sha256"]
    cut = CellSet.bands(c["grid"], c["bands"]).drop_columns(*GB_CUT[case])
    if GB_CUT[case] != (0, 0):
        # GB tests/test_matrix.py:272-281: nine columns of three bands, n in [5, NT - 5]
        assert len(cut.m) == M.shape[0] - 3 * 9
        assert min(cut.n) == 5 and max(cut.n) == c["NT"] - 5
    else:                                        # 12 months: the periodic year is not cut
        assert cut == M.dst.cells
    Mc = FourierToWDM(_store(c), WDMDomain(cut))
    assert check_blas_golden(fps["V_cut_C"], Mc.C, build, G6_RTOL)
    assert check_blas_golden(fps["V_cut_U"], Mc.C, build, G6_RTOL)
    assert sha256_array(_band_pos(c, cut)) == fps["V_cut_b"]["sha256"]
    assert sha256_array(np.asarray(cut.n, dtype=np.int64)) == fps["V_cut_n"]["sha256"]
    assert fingerprint_gap(fps["V_cut_cells"], Mc.apply(c["prep"]["store_z"])) <= G6_RTOL


@pytest.mark.parametrize("case", GB_CASES)
def test_g6_cells_of_the_store_data(case):
    c = gb_case(case)
    cells = _raw(c).apply(c["prep"]["store_z"])
    ref = c["g6"]["V_raw_cells"]
    assert cells.shape == ref.shape
    assert np.max(np.abs(cells - ref)) <= G6_RTOL * np.max(np.abs(ref))


@pytest.mark.parametrize("case", GB_CASES)
def test_g6_pad_cell_matrix_and_its_cells(case):
    c = gb_case(case)
    fps, build = c["meta"]["fingerprints"], c["meta"]["build"]
    W, zp = _pad(c)
    assert check_blas_golden(fps["V_pad_U"], W.C, build, G6_RTOL)
    assert sha256_array(_band_pos(c, W.dst.cells)) == fps["V_pad_b"]["sha256"]
    assert sha256_array(np.asarray(W.dst.cells.n, dtype=np.int64)) == fps["V_pad_n"]["sha256"]
    Mp = W.after(zp)
    assert Mp.src == _store(c) and Mp.C.shape == tuple(fps["V_pad_C"]["shape"])
    assert Mp.window == W.window == W.src            # the padded record's band window
    assert check_blas_golden(fps["V_pad_C"], Mp.C, build, G6_RTOL)
    cells = Mp.apply(c["prep"]["store_z"])
    ref = c["g6"]["V_pad_cells"]
    assert np.max(np.abs(cells - ref)) <= G6_RTOL * np.max(np.abs(ref))


@pytest.mark.parametrize("case", GB_CASES)
def test_pad_cells_equal_raw_cells_away_from_the_edges(case):
    """Ported from GB tests/test_matrix.py:52-72 (raw and pad; the taper is dropped)."""
    c = gb_case(case)
    NT, z = c["NT"], c["prep"]["store_z"]
    raw_map = _raw(c)
    W, zp = _pad(c)
    assert W.dst.cells.grid.Nt == NT + GB_PAD_COLUMNS
    assert np.array_equal(np.asarray(W.dst.cells.n), np.asarray(raw_map.dst.cells.n))
    raw, pad = raw_map.apply(z), W.after(zp).apply(z)
    n = np.asarray(raw_map.dst.cells.n)
    scale = np.max(np.abs(raw))
    mid = (n > 24) & (n < NT - 25)
    assert np.max(np.abs(pad[:, mid] - raw[:, mid])) < 1e-4 * scale
    assert np.max(np.abs(pad[:, n == NT // 2] - raw[:, n == NT // 2])) < 1e-7 * scale
    assert np.max(np.abs(pad[:, n == 0] - raw[:, n == 0])) > 0.1 * scale


@pytest.mark.parametrize("case", GB_CASES)
def test_fft_cells_match_dense_cells(case):
    """Strategy cross-check (spec §3.2): fft vs dense GB cells <= 1e-12 relative."""
    c = gb_case(case)
    z = c["prep"]["store_z"]
    dense, fft = _raw(c).apply(z), _raw(c, "fft").apply(z)
    assert np.max(np.abs(fft - dense)) <= 1e-12 * np.max(np.abs(dense))
    W, zp = _pad(c)
    a = W.after(zp).apply(z)
    b = Compose(FourierToWDM(W.src, W.dst, impl="fft"), zp).apply(z)
    assert np.max(np.abs(b - a)) <= 1e-12 * np.max(np.abs(a))


def test_push_of_a_stationary_bin_spectrum_is_the_separable_kernel():
    """``1/2 Re(C diag(s) C^H)`` == ``wdm.evolutionary.eq45_separable_kernels`` == G6 K^I, K^G."""
    c = gb_case("gb_3mo")
    M = _raw(c)
    fps = c["meta"]["fingerprints"]
    for key, s in (("KI", c["prep"]["store_sI"]), ("KG", c["prep"]["store_sG"])):
        K = np.stack([M.push(DenseCov(np.diag(s[ch]).astype(complex))).dense() for ch in range(2)])
        oracle = np.stack([wdm.evolutionary.eq45_separable_kernels(M.C, s[ch, M.lo:M.lo + M.width])
                           for ch in range(2)])
        assert np.max(np.abs(K - oracle)) <= 1e-14 * np.max(np.abs(oracle))
        assert fingerprint_gap(fps[key], K) <= G6_RTOL


def test_restrict_truncates_the_atoms_to_a_sub_window():
    """GB exact rows use the atoms on the analysis window only (matrix.py:686)."""
    c = gb_case("gb_3mo")
    p = c["prep"]
    full = _raw(c)
    window = FourierDomain(c["N"], c["DT"], int(p["bin_offset"]), int(p["n_bins"]), "sigma0")
    cells = WDMDomain(CellSet.bands(c["grid"], c["bands"]))
    with pytest.raises(ValueError, match="do not cover"):
        FourierToWDM(window, cells)
    Mr = FourierToWDM(window, cells, restrict=True)
    off = int(p["bin_offset"]) - (full.src.k0 + full.lo)
    assert (Mr.lo, Mr.width) == (0, 419)
    assert np.array_equal(Mr.C, full.C[:, off:off + 419])
    with pytest.raises(ValueError, match="impl='dense' only"):
        FourierToWDM(window, cells, impl="fft", restrict=True)
    # after() rebuilds the atoms on the band window: truncated atoms are refused
    short = FourierDomain(c["N"] // 2, c["DT"], int(p["bin_offset"]) // 2, 8, "sigma0")
    with pytest.raises(ValueError, match="the band window"):
        Mr.after(FourierZeroPad(short, window))
    # bins of another domain are refused, not read at the wrong offsets: the store fed to
    # the band-window and analysis-window maps, the whole rfft fed to the store map
    z = p["store_z"]
    zfull = np.zeros((2, c["N"] // 2 + 1), complex)
    zfull[:, full.src.k0:full.src.k0 + full.src.n] = z
    with pytest.raises(ValueError, match="last axis"):
        Mr.apply(z)
    for impl in ("dense", "fft"):
        with pytest.raises(ValueError, match="last axis"):
            FourierToWDM(full.window, cells, impl=impl).apply(z)
        with pytest.raises(ValueError, match="last axis"):
            FourierToWDM(full.src, cells, impl=impl).apply(zfull)


def test_adjoint_and_the_record_scale():
    rng = np.random.default_rng(0)
    c = gb_case("gb_3mo")
    M = _raw(c)
    w = rng.standard_normal((2, M.shape[0]))
    z = rng.standard_normal((2, M.src.n)) + 1j * rng.standard_normal((2, M.src.n))
    lhs = np.sum(w * M.apply(z))
    assert abs(lhs - np.real(np.vdot(M.adjoint(w), z))) <= 1e-12 * abs(lhs)
    with pytest.raises(ValueError, match="last axis"):
        M.adjoint(w[:, :-1])
    # on a whole record z = rfft(x), the cells are sqrt(N / (2 dt)) times the WDM transform
    g = wdm.Grid(8, 16, 0.5)
    x = rng.standard_normal(g.N)
    cells = WDMDomain(CellSet.bands(g, tuple(range(1, 8))))
    lhs = FourierToWDM(FourierDomain(g.N, g.dt, 0, g.N // 2 + 1, "rfft"), cells).apply(np.fft.rfft(x))
    rhs = np.sqrt(g.N / (2 * g.dt)) * TimeToWDM(TimeDomain(g.N, g.dt), cells).apply(x)
    assert np.max(np.abs(lhs - rhs)) <= 1e-14 * np.max(np.abs(rhs))
    # after(FourierCrop): the composite matrix is the composition, and on a band-limited
    # 160-sample record it is the WDM transform of the cropped samples [16, 144)
    Z = np.zeros(81, complex)
    Z[8:57] = rng.standard_normal(49) + 1j * rng.standard_normal(49)    # no DC, no Nyquist
    xr = np.fft.irfft(Z, n=160)
    W = FourierToWDM(FourierDomain(g.N, g.dt, 0, 64, "rfft"), cells)    # the band windows
    crop = FourierCrop(FourierDomain(160, g.dt, 0, 81, "rfft"), W.src, 16)
    got = W.after(crop).apply(Z)
    assert np.max(np.abs(got - Compose(W, crop).apply(Z))) <= 1e-12 * np.max(np.abs(got))
    ref = np.sqrt(g.N / (2 * g.dt)) * TimeToWDM(TimeDomain(g.N, g.dt), cells).apply(xr[16:144])
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))
    # a composite's src is the outer record, not the band window: it cannot be chained
    crop2 = FourierCrop(FourierDomain(192, g.dt, 0, 97, "rfft"), crop.src, 8)
    with pytest.raises(ValueError, match="the band window"):
        W.after(crop).after(crop2)
    # the band window lies on the cells' grid record (N = 128), never on a composite's
    # outer record: a composite whose src has the window's bins [0, 64) is still refused
    Wc = W.after(FourierCrop(FourierDomain(160, g.dt, 0, 64, "rfft"), W.src, 16))
    assert Wc.window == W.window == W.src and Wc.src != Wc.window
    with pytest.raises(ValueError, match="the band window"):
        Wc.after(FourierCrop(FourierDomain(192, g.dt, 0, 64, "rfft"), Wc.src, 8))


def test_edge_bands_are_refused():
    g = wdm.Grid(8, 16, 0.5)
    with pytest.raises(ValueError, match="interior bands"):
        FourierToWDM(FourierDomain(g.N, g.dt, 0, 65, "rfft"), WDMDomain(CellSet.all_active(g)))


@pytest.mark.jax
@pytest.mark.parametrize("case", ["gb_3mo", "gb_12mo"])
def test_jax_parity(case):
    jax, JB = jax_or_skip()
    c = gb_case(case)
    z = c["prep"]["store_z"]
    W, zp = _pad(c)
    for M in (_raw(c), _raw(c, "fft"), W.after(zp)):
        out = jax.jit(lambda m, v: m.apply(v))(M.on(JB), JB.asarray(z))
        assert isinstance(out, jax.Array)
        ref = M.apply(z)
        assert np.max(np.abs(np.asarray(out) - ref)) <= 1e-13 * np.max(np.abs(ref))
