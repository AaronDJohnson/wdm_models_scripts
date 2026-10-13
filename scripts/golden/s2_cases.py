"""S2 gate cases: the sinusoid manuscript_mid truth and the GB anchors assembled from
domainia building blocks exactly as spec sections 4.8 and 4.9 sketch them.

These are previews of the S3/S6 wiring, not the model layer: ``wdm_models`` (configs,
priors, waveforms, the 15-model registry) does not exist in S2, so its inputs are built
here, test-locally, from ``data/`` and ``golden/`` and named as such:

* the sinusoid signal ``A_h cos(2 pi f0 t + phi0)``, ``t_j = j dt`` (WT
  ``likelihood.signal``, ``wdm_toys/likelihood.py:110-117``) -- S3 owns it;
* the GB template: the golden gbonfly store template at the injection
  (``golden/galactic_binary/intermediates_<case>.npz["store_template"]``), so the GB
  cases are evaluated at the anchor only -- S5 owns the waveform;
* the GB annual envelope (``galactic_binary/envelope.py`` basis and
  ``likelihoods.ghat_at``) as :class:`G6Envelope` -- S5 owns ``AnnualEnvelope``.

Every number is read with ``goldens_io.read_json`` / ``to_float``; arrays produced by
BLAS are compared by value with a tolerance, never by sha256.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from types import SimpleNamespace

import numpy as np

import wdm
from domainia.covariance import (ChannelStack, DiagonalCovariance, Eq45Covariance,
                                 ExactFourierCovariance, ExactTimeCovariance, HermitianCovariance,
                                 PushedRows, SchurEdge, ZeroPadded)
from domainia.data import FourierBins, TimeSeries
from domainia.density import ComplexGaussian, RealGaussian
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.likelihood import FourierWhittle, GaussianLikelihood, Params, StaticSignal
from domainia.maps import BinSelect, FourierToWDM, Identity, TimeToWDM
from domainia.nuisance import FourierBandNuisance
from domainia.spectra import LogLinearPSD, SeparablePSD, TabulatedSpectrum
from domainia.variance import BandAveragedVariance, MidpointVariance
from scripts.golden.goldens_io import read_json, read_npz, to_float
from scripts.golden.paths import DATA, GOLDEN
from scripts.golden.workers._common import runtime_fingerprint

# --- sinusoid (spec section 4.8) -------------------------------------------------------------
SIN_CASE = "manuscript_mid"
NF, NT, DT = 64, 256, 2.0 ** -10
N = NF * NT
FMIN = 4.0
CUT, EDGE, PAD = (6, 5), (8, 7), 16
DRIFT = Params(("A_h", "f_0", "phi_0", "log10_A_n", "alpha_c", "alpha_d"), (0, 1, 2), (4, 5))
WHITTLE = Params(("A_h", "f_0", "phi_0", "log10_A_n", "alpha"), (0, 1, 2), (3, 4))
#: The S2 preview set: every edge treatment and both stationary forms (spec, S2 gate (a)).
SIN_DEFAULT = ("whittle_raw", "whittle_cut", "band_raw", "mid_cut")
SIN_SLOW = ("tri_raw", "band_pad", "band_exact")
#: Measured bitwise on the recording build (spec section 3.6 and the P0 measurement).
SIN_BITWISE = ("whittle_raw", "whittle_cut", "band_raw", "mid_raw", "band_cut", "mid_cut",
               "tri_raw", "tri_cut")

# --- galactic binary (spec section 4.9) ------------------------------------------------------
GLASS = ("f0T", "cos_colat", "lon", "lnA", "cos_iota", "psi", "phi0", "fdotT2")   # S5 owns these
ENV = ("logg_mean", "ann_c", "ann_s", "semi_c", "semi_s")
GB_PARAMS = Params(GLASS + tuple(f"{n}_{c}" for c in "AE" for n in ENV), tuple(range(8)),
                   tuple(range(8, 18)))
GB_WHITTLE = Params(GLASS + ("log_gbar_A", "log_gbar_E"), tuple(range(8)), (8, 9))
GB_CASE = "gb_3mo"
GB_MODELS = ("whittle_raw", "band_raw", "tri_exact", "exact")


def recording_build() -> bool:
    """True on the build that recorded the sinusoid G6 file (python, machine, numpy, scipy,
    BLAS): there the bitwise-class models must reproduce it exactly."""
    rec = read_json(GOLDEN / "sinusoid" / "intermediates" / f"{SIN_CASE}_truth.json")["runtime"]
    now = runtime_fingerprint()
    return all(rec.get(k) == now.get(k) for k in ("python", "machine", "numpy", "scipy", "blas"))


def sinusoid_gate(model: str) -> tuple[float, float]:
    """``(atol, rtol)``: the approved NumPy reference gate (golden/sinusoid/tolerances_measured.json)."""
    g = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")["approved"]["numpy"]["references"]
    return to_float(g[model]["atol"]), to_float(g[model]["rtol"])


def gb_gate(case: str, model: str) -> float:
    """The approved G3 NumPy gate (uniform 1e-8, golden/galactic_binary/tolerances_measured.json)."""
    return to_float(read_json(GOLDEN / "galactic_binary" / "tolerances_measured.json")
                    ["approved"]["G3"]["numpy"][f"{case}/{model}"])


def record(record_property, item: str, observed: float, gate: float) -> None:
    """The G6 reporting convention read by ``scripts/golden/s2_report.py``: one junit
    property ``g6`` per comparison."""
    record_property("g6", json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


# --- sinusoid --------------------------------------------------------------------------------
@cache
def sinusoid_golden() -> SimpleNamespace:
    path = GOLDEN / "sinusoid" / "intermediates" / f"{SIN_CASE}_truth.json"
    return SimpleNamespace(json=read_json(path), path=path)


@cache
def sinusoid_inputs() -> SimpleNamespace:
    z = read_npz(DATA / "sinusoid" / SIN_CASE / "toy_data.npz")
    cfg = {str(k): float(v) for k, v in zip(z["_cfg_keys"], z["_cfg_vals"], strict=True)}
    log10A = float(np.log10(cfg["A_n"]))
    src = [cfg["A_h"], cfg["f_0"], cfg["phi_0"]]
    return SimpleNamespace(
        d=np.asarray(z["d"], dtype=np.float64), cfg=cfg,
        truth=np.array(src + [log10A, cfg["alpha_c"], cfg["alpha_d"]]),
        truth_whittle=np.array(src + [log10A, cfg["alpha_c"]]))


def sinusoid_signal(src, t=np.arange(N) * DT):
    """WT ``likelihood.signal`` (fdot = 0): ``phase = 2 pi f0 t + phi0; A_h cos(phase)``."""
    return src[0] * np.cos(2.0 * np.pi * src[1] * t + src[2])


def _sin_base(base: str, psd: LogLinearPSD, cells: CellSet, t_max: float | None = None):
    if base == "band":
        return DiagonalCovariance(BandAveragedVariance(psd, cells), len(cells.m))
    if base == "mid":
        return DiagonalCovariance(MidpointVariance(psd, cells), len(cells.m))
    return Eq45Covariance(psd, cells, builder="bessel", t_max=t_max)


def sinusoid_likelihood(model: str):
    """One sinusoid model at manuscript_mid, NumPy ``historical`` strategies."""
    s = sinusoid_inputs()
    time = TimeDomain(N, DT)
    data = TimeSeries(s.d, time)
    psd = LogLinearPSD(f_floor=FMIN, f_ref=1.0, T=N * DT)
    signal = StaticSignal(sinusoid_signal)
    if model == "whittle_raw":
        return FourierWhittle("real-orthonormal", data, psd=psd, fmin=FMIN, signal=signal, params=WHITTLE)
    if model == "whittle_cut":
        return FourierWhittle("real-orthonormal", data, psd=psd, fmin=FMIN, signal=signal, params=WHITTLE,
                              crop=(CUT[0] * NF, (NT - CUT[1]) * NF))
    base, edge = model.split("_")
    nuis = FourierBandNuisance(N, DT, FMIN)
    if edge == "pad":
        ext = wdm.Grid(NF, NT + PAD, DT)
        cov = ZeroPadded(_sin_base(base, psd, CellSet.all_active(ext), t_max=ext.Nt * wdm.delta_T(NF, DT)),
                         strategy="time_marginal", grid=ext, pad_columns=PAD)
        return GaussianLikelihood(data, Identity(time), cov, RealGaussian(0.0, "log10_A_n"), signal=signal,
                                  params=DRIFT, nuisance=nuis.discarded_basis().T)
    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    if edge == "cut":
        cells = cells.drop_columns(*CUT)
    tmap = TimeToWDM(time, WDMDomain(cells))
    U = tmap.apply(nuis.discarded_basis()).T          # FlatMarginal applies nuisance_basis (WT)
    if edge == "exact":
        e = cells.columns(*EDGE)
        rows = PushedRows(ExactTimeCovariance(psd, N, DT, form="operator", P=24), tmap, e, order="fixed")
        cov = SchurEdge(_sin_base(base, psd, cells.complement(e)), rows, e)
    else:
        cov = _sin_base(base, psd, cells)
    return GaussianLikelihood(data, tmap, cov, RealGaussian(0.5 * N * float(np.log(DT)), "log10_A_n"),
                              signal=signal, params=DRIFT, nuisance=U)


def sinusoid_compare(model: str) -> dict:
    """``{quad, logdet, lnL}`` -> ``(value, golden, |delta|)`` at the truth."""
    s, p = sinusoid_inputs(), sinusoid_golden().json["parts"][model]
    lik = sinusoid_likelihood(model)
    theta = s.truth_whittle if model.startswith("whittle") else s.truth
    (quad, logdet), = lik.quad_logdet(theta)
    lnl = float(lik(theta[None, :])[0])
    out = {}
    for key, val in (("quad", quad), ("logdet", logdet), ("lnL", lnl)):
        ref = to_float(p[key])
        out[key] = (val, ref, abs(val - ref))
    return out
