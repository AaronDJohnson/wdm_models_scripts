"""D-C10: PadMask (GB ``pad_mask``/``mask_rows``) and the GB padded models at the anchor.

* The test-local padded data map reproduces ``V.C`` of the pad variant (pins the helper).
* PadMask: 99 rows (17 leading + 16 trailing columns x 3 bands) at every duration; ``E``,
  ``Kpos`` and ``A_E`` against G6.
* ``A Lambda_R A^T`` for band/mid/tri_pad per channel (fingerprints) and the per-channel
  (quad, logdet) at the anchor; the opt-in ``schur`` factor within the G3 gate.
12 months (2700 cells, dense) is ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

import domtest_inputs as T
from domainia.covariance import DiagonalCovariance, Eq45Covariance, PadMask, ZeroPadded
from domainia.domains import CellSet

CASES = ["gb_3mo", "gb_6mo", pytest.param("gb_12mo", marks=pytest.mark.slow)]


def _mask(c: T.GBCase) -> tuple[PadMask, np.ndarray]:
    C, b, n = c.pad_map_rows()
    rho = np.asarray(c.P["rho_pad"], float)[b, n]
    mask = PadMask.build(map_rows=C, rho=rho, kept_band=b, kept_n=n, R_grid=c.pad_grid, R_bands=c.R_bands,
                         record_N=c.N, store_k0=c.store_k0, store_n=c.store_n, k0=c.k0)
    return mask, C


def _base(c: T.GBCase, base: str, ch: int):
    """Lambda_R on the store bands x NT' columns (GB ``pad_static``): the band / mid
    variances (test-local: the model layer's separable BandAveraged/Midpoint variance) or
    the block-tridiagonal separable Eq. (45)."""
    R = CellSet.bands(c.pad_grid, c.R_bands)
    if base == "tri":
        return Eq45Covariance(c.pad_psd, R, builder="separable", assembly="blocktri", layout="ragged", channel=ch)
    from wdm.filter import phi_centered

    NTp, nR = c.NTp, c.R_bands.size
    sI, sG = np.asarray(c.P["pad_store_sI"], float), np.asarray(c.P["pad_store_sG"], float)
    idx = ((NTp // 2) * (c.R_bands - c.R_bands[0]))[:, None] + np.arange(NTp)[None, :]
    if base == "band":
        phi = phi_centered(c.Nf, NTp, c.DT)
        w2 = phi ** 2 / np.sum(phi ** 2)
        sIb = np.stack([sI[:, idx[b]] @ w2 for b in range(nR)], axis=1)
        sGb = np.stack([sG[:, idx[b]] @ w2 for b in range(nR)], axis=1)
    else:
        centre = idx[:, NTp // 2]
        sIb, sGb = sI[:, centre], sG[:, centre]
    sIr, sGr = np.repeat(sIb, NTp, axis=1)[ch], np.repeat(sGb, NTp, axis=1)[ch]
    Bn = c.env.basis(np.tile(np.arange(NTp), nR) * (float(c.P["T"]) / c.NT))
    return DiagonalCovariance(lambda nu, be: 0.5 * (be.asarray(sIr) + be.xp.exp(nu @ be.asarray(Bn)) * be.asarray(sGr)),
                              nR * NTp)


@pytest.mark.parametrize("case", CASES)
def test_g6_pad_map_and_mask(case):
    c = T.gb_case(case)
    mask, C = _mask(c)
    fp = c.g6["fingerprints"]
    T.assert_summary(C, fp["V_pad_C"])
    assert mask.E.size == c.g6["scalars"]["pad_mask"]["n_rows"] == 99
    assert np.array_equal(mask.E, c.A["pad_E"]) and np.array_equal(mask.Kpos, c.A["pad_Kpos"])
    assert (mask.k0, mask.NT_ext) == (c.g6["scalars"]["pad_mask"]["k0"], c.g6["scalars"]["pad_mask"]["R_NT"])
    T.assert_summary(mask.AE, fp["pad_AE"])
    assert np.max(np.abs(mask.AE[0] - c.A["pad_AE_row0"])) <= 1e-12 * np.max(np.abs(mask.AE[0]))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("base", ["band", "mid", "tri"])
def test_g6_pad_covariance_and_quad_logdet(case, base):
    c = T.gb_case(case)
    mask, C = _mask(c)
    r = c.A["V_pad_cells"] - np.real(c.A["store_template"] @ C.T)
    got, schur = [], []
    for ch in range(2):
        b = _base(c, base, ch)
        zp = ZeroPadded(b, strategy="cell_congruence", mask=mask, n_bands=c.bands.size, n_columns=c.NT)
        Z = zp.dense(c.mu[ch])
        T.assert_summary(Z, c.g6["fingerprints"][f"cov_{base}_pad_c{ch}"])
        assert np.allclose(np.diagonal(Z), c.A[f"cov_{base}_pad_diag"][ch], rtol=1e-13, atol=0)
        F = zp.factor(c.mu[ch])
        got.append([float(F.quad(r[ch], form="whiten")), float(F.logdet)])
        Fs = ZeroPadded(b, strategy="cell_congruence", mask=mask, n_bands=c.bands.size, n_columns=c.NT,
                        factor="schur").factor(c.mu[ch])
        schur.append([float(Fs.quad(r[ch])), float(Fs.logdet)])
    want = c.quad_logdet(f"{base}_pad")
    assert np.allclose(np.array(got), want, rtol=1e-12, atol=0)
    assert np.max(np.abs(np.array(schur).sum(axis=1) - want.sum(axis=1))) <= 2e-8


@pytest.mark.parametrize("case", ["gb_3mo", "gb_6mo", "gb_12mo"])
def test_rho_pad_is_the_kept_energy(case):
    """Ported GB test_matrix.py::test_rho (the pad half; ``rho_win`` belongs to the dropped
    ``*_win`` models): the prepared kept-energy fraction is 1/2 at column 0, 1 mid-record,
    below 0.95 at the last column and inside (0.4, 1]."""
    c = T.gb_case(case)
    pad, NT = np.asarray(c.P["rho_pad"], float), c.NT
    assert np.allclose(pad[:, 0], 0.5, atol=1e-3) and np.allclose(pad[:, NT // 2], 1.0, atol=1e-12)
    assert np.all(pad[:, -1] < 0.95) and np.all(pad <= 1 + 1e-12) and np.all(pad > 0.4)


def test_pad_is_m_lambda_m():
    """Ported GB test_matrix.py::test_pad_is_m_lambda_m (3 months): with every kept row computed
    (``edge_tol`` below 0) ``A`` is the dense data mask; its rows ``E`` are the PadMask's, it is
    the identity at ``Kpos`` away from the edges, white noise keeps the data's own variance
    except for the leakage from beyond the store, and ``ZeroPadded(cell_congruence)`` is
    ``A Lambda_R A^T`` for the band and the tri base (both channels), positive definite."""
    c = T.gb_case("gb_3mo")
    mask, Cm = _mask(c)
    _, b, n = c.pad_map_rows()
    kw = dict(map_rows=Cm, rho=np.asarray(c.P["rho_pad"], float)[b, n], kept_band=b, kept_n=n,
              R_grid=c.pad_grid, R_bands=c.R_bands, record_N=c.N, store_k0=c.store_k0,
              store_n=c.store_n, k0=c.k0)
    assert list(c.R_bands) == c.g6["scalars"]["pad_mask"]["R_bands"]
    full = PadMask.build(**kw, edge_tol=-1.0)
    A = full.AE
    assert np.array_equal(full.E, np.arange(n.size)) and np.array_equal(full.Kpos, mask.Kpos)
    np.testing.assert_allclose(A[mask.E], mask.AE, rtol=0, atol=1e-13)
    J = np.zeros_like(A)
    J[np.arange(n.size), mask.Kpos] = 1.0
    inner = np.setdiff1d(np.arange(n.size), mask.E)
    assert np.abs(A - J)[inner].max() < 1e-4 and np.abs(A - J)[n == 0].max() > 0.1
    ratio = np.diag(A @ A.T) / np.diag(np.real(Cm @ np.conj(Cm).T))
    assert np.allclose(ratio[n == c.NT // 2], 1.0, atol=1e-8)
    assert 0.98 < ratio[n == 0].min() < 1.0
    for base in ("band", "tri"):
        for ch in range(2):
            lam_R = _base(c, base, ch)
            dense = A @ lam_R.dense(c.mu[ch]) @ A.T
            got = ZeroPadded(lam_R, strategy="cell_congruence", mask=mask, n_bands=c.bands.size,
                             n_columns=c.NT).dense(c.mu[ch])
            assert np.abs(got - dense).max() < 1e-5 * np.abs(dense).max(), (base, ch)
            np.linalg.cholesky(0.5 * (got + got.T))
    # inconsistent inputs are refused before any arithmetic (each returned a wrong mask silently)
    skip = np.r_[c.R_bands[:3], c.R_bands[4:], c.R_bands[-1] + 1]
    for over, match in [({"chunk": 0}, "chunk"), ({"chunk": -1}, "chunk"),
                        ({"rho": kw["rho"][:-5]}, "one entry per row"), ({"kept_n": n[:-1]}, "one entry per row"),
                        ({"map_rows": Cm[:, :-1]}, "map_rows must be"), ({"R_bands": skip}, "contiguous"),
                        ({"kept_n": n + (c.NTp - c.NT + 1)}, "kept_n must lie"),
                        ({"k0": c.R_bands.size - c.bands.size + 1}, "k0 \\+ kept_band"),
                        ({"k0": -1}, "k0 \\+ kept_band")]:
        with pytest.raises(ValueError, match=match):
            PadMask.build(**{**kw, **over})
