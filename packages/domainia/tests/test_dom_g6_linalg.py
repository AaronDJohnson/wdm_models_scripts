"""D-A8: G6 at NumPy tolerance for the linear algebra (spec section 7.2 G6, 7.6 S2).

The covariance pieces are built test-locally from ``wdm`` and the shipped data (the model
layer that owns them arrives in S3/S5): the sinusoid's Eq. (45) blocks at the truth
(manuscript_mid, alpha_c = -2, alpha_d = 1.2, A_n outside) through ``wdm.evolutionary``, the
residual cells from ``data/sinusoid/manuscript_mid/toy_data.npz``, and the GB exact
covariance M' from ``data/galactic_binary/prepared/gb_*.npz`` at ``mu_fit``.  The factors'
``(quad, log-det)`` must equal the G6 ``parts`` / ``scalars``: bitwise on the recording
build, within ``1e-9 + 1e-12 |value|`` elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import wdm
from wdm.backend import NUMPY
from wdm.evolutionary import eq45_blocks_cheb, eq45_coeffs_bessel, eq45_plan
from domainia import linalg as la
from domainia.covariance import BlockTriCovFactor, DiagFactor, HermitianCovariance, padded_factor
from domainia.domains import CellSet

pytestmark = pytest.mark.golden

ROOT = Path(__file__).resolve().parents[3]                            # wdm_models_scripts
SIN_G6 = ROOT / "golden" / "sinusoid" / "intermediates" / "manuscript_mid_truth"
SIN_DATA = ROOT / "data" / "sinusoid" / "manuscript_mid" / "toy_data.npz"
NF, NT, DT = 64, 256, 2.0**-10
N = NF * NT
AC, AD = -2.0, 1.2
A_H, F0, PHI0 = 0.30421576834679953, 137.0, 0.6                      # manuscript_mid truth
F_MIN, F_REF = 4.0, 1.0


def gate(got, want, *, bitwise):
    if bitwise:
        assert got == want, (got, want)
    assert abs(got - want) <= 1e-9 + 1e-12 * abs(want), (got, want)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


@pytest.fixture(scope="module")
def sin(golden_io):
    if not (SIN_G6.with_suffix(".json").is_file() and SIN_DATA.is_file()):
        pytest.skip("sinusoid G6 golden or toy data not present")
    g6 = golden_io.read_json(SIN_G6.with_suffix(".json"))
    with np.load(SIN_G6.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(SIN_DATA) as z:
        d = z["d"]
    h = A_H * np.cos(2.0 * np.pi * F0 * (np.arange(N) * DT) + PHI0)   # WT likelihood.signal
    return dict(g6=g6, arrays=arrays, d=d, x=d - h,
                y=(wdm.grid.pack(wdm.forward_time(d, NF, NT, DT), NF, NT)
                   - wdm.grid.pack(wdm.forward_time(h, NF, NT, DT), NF, NT)))


def tri_blocks(Nt, keep, t_max_ratio):
    plan = eq45_plan(NF, Nt, DT, keep, P=24)
    Lf = L_of(plan.f)
    return eq45_blocks_cheb(plan, lambda k: eq45_coeffs_bessel(Lf[k], AC, AD, 24, r=t_max_ratio))


@pytest.mark.parametrize("model, cut", [("tri_raw", None), ("tri_cut", (6, 5))])
def test_sinusoid_tri_quad_and_logdet(sin, same_build, golden_io, model, cut):
    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    if cut:
        cells = cells.drop_columns(*cut)
    diag, up = tri_blocks(NT, cells.keep_mask(), 1.0)
    y = sin["y"][cells.packed_positions()]
    part = sin["g6"]["parts"][model]
    bitwise = same_build(sin["g6"]["runtime"])
    f = la.blocktri_factor(diag, up, layout="ragged")
    gate(f.logdet, golden_io.to_float(part["logdet_C"]), bitwise=bitwise)
    gate(float(y @ la.blocktri_solve(f, y)), golden_io.to_float(part["quad_C"]), bitwise=bitwise)
    fp = la.blocktri_factor(diag, up, layout="padded", band_layout=la.BandLayout.from_cells(cells))
    assert abs(float(fp.logdet) - f.logdet) <= 1e-12 * abs(f.logdet)        # padded vs ragged
    w = la.blocktri_whiten(fp, y)
    assert abs(float(w @ w) - golden_io.to_float(part["quad_C"])) <= 1e-12 * golden_io.to_float(
        part["quad_C"])


@pytest.mark.parametrize("base, key", [("band", "band_variance_raw"),
                                       ("mid", "mid_variance_raw")])
def test_sinusoid_diagonal_quad_and_logdet(sin, same_build, golden_io, base, key):
    """{band,mid}_raw on all 16384 cells and {band,mid}_cut on the 15681 cells of
    drop_columns(6, 5): a cut cell keeps its raw variance (same literal cell time), so
    both come from the G6 raw variance vector."""
    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    bitwise = same_build(sin["g6"]["runtime"])
    for edge, kept in (("raw", cells), ("cut", cells.drop_columns(6, 5))):
        pos = kept.packed_positions()
        f = DiagFactor.build(sin["arrays"][key][pos], NUMPY)
        y = sin["y"][pos]
        part = sin["g6"]["parts"][f"{base}_{edge}"]
        gate(float(f.logdet), golden_io.to_float(part["logdet_C"]), bitwise=bitwise)
        gate(float(y @ f.solve(y)), golden_io.to_float(part["quad_C"]), bitwise=bitwise)


def test_sinusoid_padded_marginal(sin, same_build, golden_io):
    """tri_pad and mid_pad: Lambda on all 17408 cells of the 64 x 272 grid (alpha continued,
    r = 272/256), W_p the transforms of the 1024 padding impulses, WT PaddedMarginal order."""
    Ntp = NT + 16
    Np = NF * Ntp
    unit = np.zeros((Np - N, Np))
    unit[np.arange(Np - N), np.arange(N, Np)] = 1.0
    Wp = wdm.forward_time_many(unit, NF, Ntp, DT)[:, wdm.active_mask(NF, Ntp)].T
    wp_fp = sin["g6"]["maps"]["Wp"]
    assert list(Wp.shape) == wp_fp["shape"]
    idx = np.asarray(wp_fp["sample_index"])
    assert np.array_equal(Wp.reshape(-1)[idx], np.asarray(wp_fp["sample_values"]))
    m, n = wdm.grid.packed_cells(NF, Ntp)
    bt = la.blocktri_factor(*tri_blocks(Ntp, None, Ntp / NT), layout="ragged")
    lam = {"tri_pad": BlockTriCovFactor(bt, bt.logdet, bt.ok, "ragged"),
           "mid_pad": DiagFactor.build(0.5 * np.exp((AC + AD * (n * wdm.delta_T(NF, DT) / (N * DT) - 0.5))
                                                    * L_of(wdm.band_frequencies(NF, DT))[m]), NUMPY)}
    bitwise = same_build(sin["g6"]["runtime"])
    x = sin["x"]
    for model, base in lam.items():
        f = padded_factor(base, Wp, grid=wdm.Grid(NF, Ntp, DT), n=N, chol_impl="scipy", be=NUMPY)
        ext = np.zeros((1, Np))
        ext[0, :N] = x
        w = wdm.forward_time_many(ext, NF, Ntp, DT)[:, wdm.active_mask(NF, Ntp)].T
        full = np.zeros((1, NF + 1, Ntp))
        full[:, wdm.active_mask(NF, Ntp)] = f.solve_cells(w, C={"Wp": Wp, "base": None}).T
        Cx = DT * wdm.inverse_time_many(full, NF, Ntp, DT)[0, :N]          # PaddedMarginal.solve
        part = sin["g6"]["parts"][model]
        gate(float(f.logdet), golden_io.to_float(part["logdet_C"]), bitwise=bitwise)
        gate(float(x @ Cx), golden_io.to_float(part["quad_C"]), bitwise=bitwise)


@pytest.mark.parametrize("case, block", [("gb_3mo", None), ("gb_6mo", None), ("gb_12mo", 450)])
def test_gb_exact_covariance_factor(same_build, golden_io, case, block):
    """M' = diag(s_I) + sqrt(s_G s_G) conj(ghat[l - j]) on the nominal bins (GB
    likelihoods.full_covariance); dense complex Cholesky at 3/6 months, the block form
    with b = NT/2 = 450 at 12 months (fullcache.noise_state)."""
    g6p = ROOT / "golden" / "galactic_binary" / f"intermediates_{case}"
    prep = ROOT / "data" / "galactic_binary" / "prepared" / f"{case}.npz"
    if not (g6p.with_suffix(".json").is_file() and prep.is_file()):
        pytest.skip(f"{case} G6 golden or prepared data not present")
    g6 = golden_io.read_json(g6p.with_suffix(".json"))
    with np.load(g6p.with_suffix(".npz")) as z:
        ghat_g6 = [z["ghat_c0"], z["ghat_c1"]]
    with np.load(prep) as p:
        k = slice(int(p["freq_lo"]), int(p["freq_lo"]) + int(p["freq_n"]))
        basis_L, mu_fit = p["basis_L"], p["mu_fit"]
        sI_all, sG_all, z_all = p["sI"], p["sG"], p["z"]
    assert g6["scalars"]["exact_block"] == block
    bitwise = same_build(g6["build"])
    for c in range(2):
        sI, sG, zc = sI_all[c, k], sG_all[c, k], z_all[c, k]
        n = sI.shape[0]
        R = np.fft.rfft(np.exp(mu_fit[c] @ basis_L)) / basis_L.shape[1]
        lags = np.arange(-(n - 1), n)
        gh = np.where(lags < 0, np.conj(R[np.abs(lags)]), R[np.abs(lags)])
        assert np.array_equal(gh[n - 1:], ghat_g6[c])
        lag = np.arange(n)[None, :] - np.arange(n)[:, None] + n - 1
        root = np.sqrt(sG)
        M = np.diag(sI.astype(complex)) + (root[:, None] * root[None, :]) * np.conj(gh)[lag]
        f = HermitianCovariance(M, block=block).factor(np.zeros(1), NUMPY)
        u = f.whiten(zc)
        want = g6["scalars"][f"Mprime_c{c}"]
        gate(float(f.logdet), golden_io.to_float(want["logdet"]), bitwise=bitwise)
        gate(float(np.real(np.vdot(u, u))), golden_io.to_float(want["q0"]), bitwise=bitwise)


def test_the_gate_bites():
    """The bitwise branch rejects a one-ulp change; the tolerance branch rejects 1e-9 + 1e-12|x|."""
    want = -183465.6485318497
    with pytest.raises(AssertionError):
        gate(float(np.nextafter(want, 0.0)), want, bitwise=True)
    gate(float(np.nextafter(want, 0.0)), want, bitwise=False)
    with pytest.raises(AssertionError):
        gate(want + 1e-9 + 1e-12 * abs(want) + 1e-6, want, bitwise=False)
