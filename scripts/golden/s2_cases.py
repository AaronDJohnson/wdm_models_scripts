"""S2 gate cases: the sinusoid manuscript_mid truth and the GB anchors assembled from
domainia building blocks exactly as spec sections 4.8 and 4.9 sketch them.

These are previews of the S3/S6 wiring, not the model layer: ``wdm_models`` (configs,
priors, waveforms, the 15-model registry) does not exist in S2, so its inputs are built
here, test-locally, from ``data/`` and ``golden/`` and named as such:

* the sinusoid signal ``A_h cos(2 pi f0 t + phi0)``, ``t_j = j dt`` (WT
  ``likelihood.signal``, ``wdm_toys/likelihood.py:110-117``) -- S3 owns it;
* the sinusoid ``exact`` model's log-det correction: the shipped bandcov interpolant
  evaluated as WT ``BandCovariance.logdet`` does (:class:`G6BandCorrection`) -- S3 owns
  ``BandCorrection`` and the artifact checks;
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
from numpy.polynomial import chebyshev as cheb

import wdm
from domainia.covariance import (ChannelStack, DiagonalCovariance, Eq45Covariance,
                                 ExactFourierCovariance, ExactTimeCovariance, HermitianCovariance,
                                 PushedRows, SchurEdge, ZeroPadded)
from domainia.data import FourierBins, TimeSeries
from domainia.density import ComplexGaussian, RealGaussian
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.likelihood import FourierWhittle, GaussianLikelihood, Params, StaticSignal
from domainia.maps import BinSelect, FourierToWDM, Identity, TimeToWDM
from domainia.nuisance import FourierBandNuisance, ProjectedOperatorMarginal
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
#: The sinusoid exact model's shipped artifacts (data/sinusoid/cache): the fitted alpha_d^2
#: coefficient c_N of log|Sigma~| and the validated bandcov log-det interpolant.
SIN_CN = DATA / "sinusoid" / "cache" / "cN_N16384_dt0.0009765625_fmin4_fref1.npy"
SIN_BANDCOV = DATA / "sinusoid" / "cache" / "bandcov_v1_4bfb466c191c79f9879a.npz"
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


@dataclass(frozen=True, eq=False)
class G6BandCorrection:
    """WT ``BandCovariance.logdet``'s correction (``wdm_toys/bandcov.py:259-266``):
    ``chebval2d(x, y, coefficients)`` at ``x`` the affine map of ``alpha_c`` and ``y`` of
    ``asinh(alpha_d / drift_scale)`` onto [-1, 1] over the prior box (``_scaled``,
    :34-38), with ``bounds`` and ``drift_scale`` read from the artifact's identity.  Test-local
    (S3 owns ``BandCorrection``); NumPy only, as the WT oracle."""

    coefficients: np.ndarray
    bounds: np.ndarray
    drift_scale: float

    @classmethod
    def load(cls, path=SIN_BANDCOV) -> "G6BandCorrection":
        z = read_npz(path)
        ident = json.loads(str(z["metadata"]))["identity"]
        return cls(np.array(z["coefficients"], dtype=float), np.asarray(ident["bounds"], dtype=float),
                   float(ident["drift_scale"]))

    def __call__(self, nu, be=None):
        ac, ad, b = float(nu[0]), float(nu[1]), self.bounds
        x = 2 * (ac - b[0, 0]) / (b[0, 1] - b[0, 0]) - 1
        lo, hi = np.arcsinh(b[1] / self.drift_scale)
        y = 2 * (np.arcsinh(ad / self.drift_scale) - lo) / (hi - lo) - 1
        return cheb.chebval2d(x, y, self.coefficients)


def sinusoid_likelihood(model: str, *, density_order: str | None = None):
    """One sinusoid model at manuscript_mid, NumPy ``historical`` strategies.

    ``density_order`` (``GaussianLikelihood``): ``None`` is the model's own -- ``"folded"``
    for ``exact`` (WT ``FastFullCovariance.lnlike``), ``"default"`` for the matrix models.
    The ``exact`` model is not an S2 preview (its composition is an S3 item): it is here
    for the density-order check against its G1 references."""
    s = sinusoid_inputs()
    time = TimeDomain(N, DT)
    data = TimeSeries(s.d, time)
    psd = LogLinearPSD(f_floor=FMIN, f_ref=1.0, T=N * DT)
    signal = StaticSignal(sinusoid_signal)
    if model.startswith("whittle") and density_order is not None:
        raise ValueError("the Whittle forms have no density_order")
    if model == "exact":
        cov = ProjectedOperatorMarginal(
            ExactTimeCovariance(psd, N, DT, P=24, c_N=float(np.load(SIN_CN)), tol=1e-7),
            FourierBandNuisance(N, DT, FMIN), G6BandCorrection.load())
        return GaussianLikelihood(data, Identity(time), cov, RealGaussian(0.0, "log10_A_n"), signal=signal,
                                  params=DRIFT, density_order=density_order or "folded")
    order = density_order or "default"
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
                                  params=DRIFT, nuisance=nuis.discarded_basis().T, density_order=order)
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
                              signal=signal, params=DRIFT, nuisance=U, density_order=order)


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


# --- galactic binary -------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class G6Envelope:
    """``galactic_binary/envelope.py`` (``orthonormal_coefficients`` with L = 2**20,
    ``basis``) and ``likelihoods.ghat_at`` (+ the lag guard of ``envelope.ghat``), test-local."""

    T: float
    period: float
    basis_L: np.ndarray
    n_harm: int = 2
    _C: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        L = 1 << 20
        t = np.arange(L) * (self.T / L)
        B = self._raw(t)
        n = B.shape[0]
        C = np.zeros((n, n))
        C[0, 0] = 1.0
        E = [np.ones(L)]
        for i in range(1, n):
            c = np.zeros(n)
            c[i] = 1.0
            v = B[i].copy()
            for j, e in enumerate(E):
                p = float(np.mean(v * e) / np.mean(e * e))
                v -= p * e
                c -= p * C[j]
            s = float(np.sqrt(np.mean(v * v)))
            E.append(v / s)
            C[i] = c / s
        object.__setattr__(self, "_C", C)

    def _raw(self, t):
        t = np.asarray(t, dtype=float)
        rows = [np.ones_like(t)]
        for h in range(1, self.n_harm + 1):
            w = 2.0 * np.pi * h / self.period
            rows += [np.cos(w * t), np.sin(w * t)]
        return np.stack(rows)

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.n_harm

    def basis(self, t) -> np.ndarray:
        """``envelope.basis(t, T, n_harm, period)`` = ``C @ raw_basis(t)``, a host constant (D-B's
        ``Envelope`` protocol; ``variance`` and ``Eq45Covariance`` read it at the cell times)."""
        return self._C @ self._raw(t)

    def log_g(self, t, mu, be=None):
        return mu @ self.basis(t)

    def ghat(self, lags, mu, be=None):
        L = self.basis_L.shape[1]
        lags = np.asarray(lags)
        if np.any(np.abs(lags) > L // 2):
            raise ValueError("lag beyond the decimation grid's Nyquist")
        R = np.fft.rfft(np.exp(mu @ self.basis_L)) / L
        out = R[np.abs(lags)]
        return np.where(lags < 0, np.conj(out), out)


@cache
def gb_envelope(case: str) -> G6Envelope:
    a = gb_inputs(case).a
    return G6Envelope(float(a["T"]), float(a["envelope_period"]), np.asarray(a["basis_L"]))


@cache
def gb_inputs(case: str) -> SimpleNamespace:
    a = read_npz(DATA / "galactic_binary" / "prepared" / f"{case}.npz")
    gi = read_json(GOLDEN / "galactic_binary" / f"intermediates_{case}.json")
    gz = read_npz(GOLDEN / "galactic_binary" / f"intermediates_{case}.npz")
    return SimpleNamespace(a=a, json=gi, npz=gz,
                           anchor=np.concatenate([a["inject_glass"], a["mu_fit"].ravel()]),
                           anchor_whittle=np.concatenate([a["inject_glass"], a["mu_freq"]]))


def gb_likelihood(case: str, model: str):
    """One GB model at the anchor of ``case``, NumPy ``historical`` strategies."""
    g = gb_inputs(case)
    a = g.a
    Nr, dtg, nt = int(a["N"]), float(a["DT"]), int(a["NT"])
    store_dom = FourierDomain(Nr, dtg, int(a["store_offset"]), int(a["store_n"]), "sigma0")
    store = FourierBins(a["store_z"], store_dom, ChannelAxis(("A", "E")))
    signal = StaticSignal(lambda src, h=g.npz["store_template"]: h)
    nominal = FourierDomain(Nr, dtg, int(a["bin_offset"]) + int(a["freq_lo"]), int(a["freq_n"]), "sigma0")
    if model == "whittle_raw":
        lo = nominal.k0 - store_dom.k0
        k = slice(lo, lo + nominal.n)
        return FourierWhittle("complex", store, map=BinSelect(store_dom, nominal),
                              sI=a["store_sI"][:, k], sG=a["store_sG"][:, k], signal=signal, params=GB_WHITTLE)
    env = gb_envelope(case)
    grid = wdm.Grid(Nr // nt, nt, dtg)
    cells = CellSet.bands(grid, tuple(int(a["first_band"]) + b for b in range(int(a["n_bands"]))))
    # one two-channel spectrum; each covariance selects its channel (GB band_psd's 2-row matvec)
    psd = SeparablePSD(((TabulatedSpectrum(store_dom, a["store_sI"]), None),
                        (TabulatedSpectrum(store_dom, a["store_sG"]), env)), ("A", "E"))
    covs = []
    for c in range(2):
        if model == "band_raw":
            covs.append(DiagonalCovariance(BandAveragedVariance(psd, cells), len(cells.m), channel=c))
        elif model == "tri_exact":
            edge = cells.columns(*(int(v) for v in a["matrix_cut"]))
            window = FourierDomain(Nr, dtg, int(a["bin_offset"]), int(a["n_bins"]), "sigma0")
            rows = PushedRows(ExactFourierCovariance(psd, window, channel=c),
                              FourierToWDM(window, WDMDomain(cells), restrict=True), edge)
            covs.append(SchurEdge(Eq45Covariance(psd, cells.complement(edge), builder="separable",
                                                 assembly="dense", channel=c),
                                  rows, edge, assembly="dense", chol_impl="numpy"))
        elif model == "exact":
            covs.append(HermitianCovariance(ExactFourierCovariance(psd, nominal, channel=c), block=None))
        else:
            raise ValueError(f"{model!r} is not in the S2 GB preview set {GB_MODELS}")
    stack = ChannelStack(tuple(covs), len(ENV))
    if model == "exact":
        return GaussianLikelihood(store, BinSelect(store_dom, nominal), stack, ComplexGaussian(),
                                  signal=signal, params=GB_PARAMS, strategy="expanded")
    # GB _dense_quad_logdet whitens; the diagonal models sum r**2 / v (matrix.py:919, 928-938)
    return GaussianLikelihood(store, FourierToWDM(store_dom, WDMDomain(cells), impl="dense"), stack,
                              RealGaussian(), signal=signal, params=GB_PARAMS,
                              quad_form="elementwise" if model == "band_raw" else "whiten")


def gb_compare(case: str, model: str) -> dict:
    """``{c<k>/quad, c<k>/logdet, lnL}`` -> ``(value, golden, |delta|)`` at the anchor."""
    g = gb_inputs(case)
    lik = gb_likelihood(case, model)
    theta = g.anchor_whittle if model.startswith("whittle") else g.anchor
    ql = lik.quad_logdet(theta)
    ref = g.npz["quad_logdet"][g.json["quad_logdet_models"].index(model)]
    out = {}
    for c in range(2):
        for j, key in enumerate(("quad", "logdet")):
            out[f"c{c}/{key}"] = (float(ql[c, j]), float(ref[c, j]), abs(float(ql[c, j]) - float(ref[c, j])))
    lnl, want = float(lik(theta[None, :])[0]), to_float(g.json["lnl_at_anchor"][model]["lnl_make_lnlike"])
    out["lnL"] = (lnl, want, abs(lnl - want))
    return out
