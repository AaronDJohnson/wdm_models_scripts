"""D-B4: pad_maps / crop_maps (exact Dirichlet kernels), FourierZeroPad, FourierCrop."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import DenseCov, gb_case, jax_or_skip

from domainia.domains import FourierDomain
from domainia.maps import FourierCrop, FourierZeroPad, crop_maps, pad_maps

NS = 256


def _band_limited(rng):
    """A record whose spectrum lives on bins 20..60 (no DC, no Nyquist): the maps are exact."""
    X = np.zeros(NS // 2 + 1, complex)
    X[20:61] = rng.standard_normal(41) + 1j * rng.standard_normal(41)
    return X, np.fft.irfft(X, n=NS)


SRC = FourierDomain(NS, 1.0, 20, 41, "rfft")


def test_zero_pad_reproduces_the_padded_record_rfft():
    X, x = _band_limited(np.random.default_rng(0))
    Np = 320
    padded = np.zeros(Np)
    padded[:NS] = x
    Xp = np.fft.rfft(padded)
    Z = FourierZeroPad(SRC, FourierDomain(Np, 1.0, 5, 80, "rfft"))
    assert Z.kind == "widely" and Z.scale == np.sqrt(NS / Np)
    assert np.max(np.abs(Z.apply(X[20:61]) - Z.scale * Xp[5:85])) <= 1e-14 * np.max(np.abs(Xp))


def test_crop_reproduces_the_cropped_record_rfft():
    X, x = _band_limited(np.random.default_rng(1))
    start, Nc = 24, 200
    Xc = np.fft.rfft(x[start:start + Nc])
    C = FourierCrop(SRC, FourierDomain(Nc, 1.0, 10, 70, "rfft"), start)
    assert np.max(np.abs(C.apply(X[20:61]) - Xc[10:80])) <= 1e-14 * np.max(np.abs(Xc))


def test_maps_match_direct_transforms():
    """Ported from GB tests/test_matrix.py:19-40 (pad and crop; the taper map is dropped):
    Nf = 24, NT = 16, store bins 40..89, atol 1e-13."""
    rng = np.random.default_rng(1)
    Nf, NT = 24, 16
    N = NT * Nf
    j = np.arange(40, 90)
    z = np.zeros(N // 2 + 1, complex)
    z[j] = rng.standard_normal(j.size) + 1j * rng.standard_normal(j.size)
    x = np.fft.irfft(z, n=N)
    store = FourierDomain(N, 1.0, 40, 50, "rfft")
    Np = (NT + 4) * Nf
    Z = FourierZeroPad(store, FourierDomain(Np, 1.0, 30, 80, "rfft"))
    want = np.fft.rfft(np.r_[x, np.zeros(Np - N)])[30:110]
    assert np.allclose(Z.apply(z[j]) / Z.scale, want, rtol=0, atol=1e-13)
    start, Nc = 3 * Nf, (NT - 4) * Nf
    C = FourierCrop(store, FourierDomain(Nc, 1.0, 20, 50, "rfft"), start)
    assert np.allclose(C.apply(z[j]), np.fft.rfft(x[start:start + Nc])[20:70], rtol=0, atol=1e-13)


def test_functions_are_the_oracle_kernels():
    """pad_maps/crop_maps keep the oracle signature (D-C's mask_rows calls crop_maps on chunks)."""
    A, B = pad_maps(NS, 320, np.arange(5, 85), np.arange(20, 61))
    Z = FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft"))
    assert np.array_equal(A, Z.A) and np.array_equal(B, Z.B)
    A2, B2 = crop_maps(NS, 24, 200, np.arange(10, 80), np.arange(20, 61))
    C = FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)
    assert np.array_equal(A2, C.A) and np.array_equal(B2, C.B)
    # chunking the input bins is exact (elementwise kernels)
    Ac, Bc = crop_maps(NS, 24, 200, np.arange(10, 80), np.arange(20, 40))
    assert np.array_equal(Ac, A2[:, :20]) and np.array_equal(Bc, B2[:, :20])


@pytest.mark.parametrize("cls", ["pad", "crop"])
def test_widely_linear_adjoint_and_no_push(cls):
    rng = np.random.default_rng(2)
    if cls == "pad":
        M = FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft"))
    else:
        M = FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)
    z = rng.standard_normal((2, 41)) + 1j * rng.standard_normal((2, 41))
    y = rng.standard_normal((2, M.dst.n)) + 1j * rng.standard_normal((2, M.dst.n))
    lhs = np.real(np.vdot(y, M.apply(z)))
    assert abs(lhs - np.real(np.vdot(M.adjoint(y), z))) <= 1e-12 * abs(lhs)
    with pytest.raises(TypeError, match="widely linear"):
        M.push(DenseCov(np.eye(41)))
    # no complex row matrix exists (rows of the adjoint would be A[k] + conj(B[k])): refuse
    with pytest.raises(TypeError, match="widely linear"):
        M.rows([0])
    with pytest.raises(TypeError, match="widely linear"):
        M.dense()
    if cls == "crop":           # start is a sample index (CropTime's operator.index rule)
        assert type(FourierCrop(SRC, M.dst, np.int64(24)).start) is int
        with pytest.raises(TypeError):
            FourierCrop(SRC, M.dst, 24.0)


def test_gb_whittle_cut_data_and_template_sites_agree():
    """GB whittle_map: data ``A @ z.T + B @ conj(z).T`` and template ``(h @ A.T) + (conj(h) @ B.T)``
    (matrix.py:892, 897): apply_data() is the data site bitwise, apply() the template site;
    the two orientations differ by ~1.8e-15 relative (BLAS)."""
    c = gb_case("gb_3mo")
    p = c["prep"]
    Nf = c["N"] // c["NT"]
    store = FourierDomain(c["N"], c["DT"], int(p["store_offset"]), int(p["store_n"]), "sigma0")
    lead, trail = 5, 4
    crop = FourierDomain((c["NT"] - lead - trail) * Nf, c["DT"], int(p["crop_k0"]),
                         int(p["crop_n"]), "sigma0")
    C = FourierCrop(store, crop, lead * Nf)
    z = p["store_z"]
    data_site = (C.A @ z.T + C.B @ np.conj(z).T).T
    assert np.array_equal(C.apply_data(z), data_site)
    out = C.apply(z)
    assert out.shape == (2, 347)
    assert np.max(np.abs(out - data_site)) <= 1e-14 * np.max(np.abs(data_site))


@pytest.mark.jax
def test_jax_parity():
    jax, JB = jax_or_skip()
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2, 41)) + 1j * rng.standard_normal((2, 41))
    for M in (FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft")),
              FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)):
        out = jax.jit(lambda m, v: m.apply(v))(M.on(JB), JB.asarray(z))
        assert isinstance(out, jax.Array)
        ref = M.apply(z)
        assert np.max(np.abs(np.asarray(out) - ref)) <= 1e-13 * np.max(np.abs(ref))
