r"""The 3-band analysis segments and their +-3-band stores (spec §5.2 ``pipeline/segments.py``;
port of ``toy_models/galactic_binary/build_segments.py`` with the ``wdm_gb.prepare_durations``
generator inlined).

A segment is, on the bins under the Meyer filters of the ``N_BANDS`` WDM bands around the
source at the tiling ``ANALYSIS_NT`` (:func:`window`):

    data = Sangria galactic binaries (sky/{dgb,igb,vgb} TDI, dt * rfft over [0, T))
           + simulated stationary instrument noise (:func:`simulate_instrument`)
           - the duration's resolved binaries (the foreground fit's SNR >= 7 set)
           + the fixed injection (:data:`INJECT_NATIVE`),

with a bandwidth-doubling check: generating every subtracted binary and the injection at
``2 BW`` must change them by less than SNR 0.1 (:func:`validate_bandwidth_error`).  The
store (:func:`build_store`) repeats the recipe on ``STORE_MARGIN`` extra bands per side; its
inner bins are the segment's, verbatim (checked to 1e-12), and its extra bins' instrument
noise draws from ``[INSTRUMENT_SEED, months, OUTER_SEED]``.

Every piece that needs no Sangria file is pinned against the shipped products by the tests
(window and store geometry, the instrument draws bit for bit, the injection and the
resolved-binary selection); :func:`build_segment` / :func:`build_store` themselves run on a
small synthetic HDF5 in the tests and on the real record only through
``scripts/galactic_binary/build_segments.py`` (approval-gated, spec §6).

Differences from the legacy code, none of which changes a shipped number:

* the injection is :data:`INJECT_NATIVE` (the shipped segments' ``inject_*``, pinned), not
  read from ``wdm_paper_2/toy_models/runs/gb/segment_snr5_3mo_gbonfly.npz``;
* the raw-convention check against ``sangria_minimal.npz`` is optional (``verification=``),
  as it needs a second private file;
* the geometry is a :class:`SegmentSpec` (the paper's from :meth:`SegmentSpec.paper`), so a
  small synthetic record can run the same code;
* waveforms come from binary_galactica: ``backend="onfly"`` by default (user ruling), and
  ``backend="fourier"`` -- gbonfly's algorithm, ``pipeline.SHIPPED_BACKEND`` -- reproduces the
  shipped injections to ~1e-16 of their peak.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..prepare import psd_components
from .foreground_fit import read_gb_ae, s_inst

__all__ = ["BW", "CASE", "CLOSE_KEYS", "DT", "EXACT_KEYS", "F_CENTRE", "INJECT_KEYS", "INJECT_NATIVE",
           "INSTRUMENT_SEED", "MAX_ERROR_SNR", "OUTER_SEED", "STORE_MARGIN", "SegmentSpec", "YEAR",
           "add_to_window", "build_segment", "build_store", "compare_npz", "full_support_snr", "generator",
           "load_foreground", "read_ae_crop", "select_reaching", "simulate_instrument", "snr_function",
           "store_window", "subtract_resolved", "validate_bandwidth_error", "verify_raw_convention",
           "window"]

DT = 5.0
YEAR = 31536000.0
INJECT_KEYS = ("f0", "fdot", "amp", "phi0", "colat", "lon", "psi", "cos_iota")
#: The injection's native parameters (``INJECT_KEYS`` order), fixed across durations: the
#: shipped segments' ``inject_*`` (the manuscript's three-month segment, amplitude not retuned).
INJECT_NATIVE = (0.001, 1e-18, 7.192196123700149e-22, 0.5, 1.7806669190576343, 0.0, 0.7, 0.3)
F_CENTRE = 0.001
#: Generator bandwidths: 32768 at 3 and 6 months, 131072 at 12 months (the doubling check
#: failed at 16384 / 65536 on the wider windows).
BW = {3: 32768, 6: 32768, 12: 131072}
CASE = {3: "gb_3mo", 6: "gb_6mo", 12: "gb_12mo"}
#: Seed of the simulated instrument noise; each duration draws from ``[seed, months]``.
INSTRUMENT_SEED = 20260929
#: Extra WDM bands stored on each side of the analysis window, and the seed word of their bins.
STORE_MARGIN = 3
OUTER_SEED = 1
#: The bandwidth-doubling budget [SNR].
MAX_ERROR_SNR = 0.1


@dataclass(frozen=True)
class SegmentSpec:
    """One duration's record and tiling: ``T`` [s], ``NT`` (even, divides ``T/DT``), the
    generator bandwidth ``bw`` and the band count; ``months`` labels the files and seeds."""

    months: int
    T: float
    NT: int
    bw: int
    n_bands: int = config.N_BANDS

    @classmethod
    def paper(cls, months: int) -> "SegmentSpec":
        if months not in CASE:
            raise ValueError(f"months must be one of {sorted(CASE)}; got {months!r}")
        return cls(months, YEAR * months / 12, config.ANALYSIS_NT[CASE[months]], BW[months])

    @property
    def N(self) -> int:
        return int(round(self.T / DT))


# ---------------------------------------------------------------------------
# wdm_gb.prepare_durations, inlined
# ---------------------------------------------------------------------------


def read_ae_crop(h5, *, T: float, dt: float = DT) -> np.ndarray:
    """Read ``obs/tdi`` over [0, T), subtract the MBHB XYZ, convert to AE, return ``dt * rfft``."""
    import h5py
    from scipy.fft import rfft

    n = int(round(T / dt))
    if n < 2 or not np.isclose(n * dt, T, rtol=0, atol=1e-8):
        raise ValueError("T must contain an integral number of samples")
    with h5py.File(h5, "r") as f:
        obs, mbhb = f["obs/tdi"], f["sky/mbhb/tdi"]
        if min(obs.shape[0], mbhb.shape[0]) < n:
            raise ValueError(f"raw record is too short for T={T:g} seconds")
        d, b = obs[:n].reshape(-1), mbhb[:n].reshape(-1)
    expected = np.arange(n) * dt
    if not (np.array_equal(d["t"], expected) and np.array_equal(b["t"], expected)):
        raise ValueError("raw timestamps must start at zero and have uniform cadence")
    x, y, z = (d[k] - b[k] for k in ("X", "Y", "Z"))
    out = np.empty((2, n // 2 + 1), dtype=complex)
    out[0] = dt * rfft((z - x) / np.sqrt(2))
    out[1] = dt * rfft((x - 2 * y + z) / np.sqrt(6))
    return out


def verify_raw_convention(h5, catalog_npz) -> dict:
    """Require the full-year raw transform to reproduce the AE of ``sangria_minimal.npz``."""
    with np.load(catalog_npz, allow_pickle=False) as p:
        T = float(p["Tobs"])
        bins = np.rint(p["freqs"] * T).astype(int)
        reference = p["channels_mbhb_sub_real"] + 1j * p["channels_mbhb_sub_imag"]
    actual = read_ae_crop(h5, T=T)[:, bins]
    relative = np.linalg.norm(actual - reference, axis=1) / np.linalg.norm(reference, axis=1)
    if not np.all(relative < 1e-10):
        raise ValueError(f"raw H5 does not reproduce supplied full-year AE: {relative}")
    return {"full_year_relative_AE_error": relative.tolist(),
            "ae_convention": "A=(Z-X)/sqrt(2); E=(X-2Y+Z)/sqrt(6)",
            "fourier_convention": "dt*rfft(time-domain data), DC indexed at zero"}


def full_support_snr(bins, h, *, T: float, psd: Callable) -> float:
    """SNR on all positive physical bins of a source, before window clipping."""
    bins = np.asarray(bins)
    use = (bins > 0) & (bins < T / (2 * DT))
    f = bins[use] / T
    S = psd(f)
    return float(np.sqrt(4 / T * np.sum(np.abs(h[:, use]) ** 2 / S)))


def add_to_window(window, bins, h, *, bin_lo: int) -> None:
    """Add only overlapping physical bins without wrapping negative indices."""
    idx = np.asarray(bins, dtype=int) - bin_lo
    keep = (idx >= 0) & (idx < window.shape[1])
    window[:, idx[keep]] += h[:, keep]


def validate_bandwidth_error(error, *, max_error: float = MAX_ERROR_SNR) -> None:
    """Require numerical subtraction uncertainty below the noise-scale budget."""
    if not np.isfinite(error) or error > max_error:
        raise ValueError(f"bandwidth convergence error SNR={error:.5g} exceeds {max_error:g}; "
                         "increase the generator bandwidth before generating a scientific segment")


def generator(T: float, bw: int, *, backend: str = "onfly") -> Callable:
    """``native -> (bins, h)``: binary_galactica at ``BW = bw`` with ``orbit_cache(T, 0, bw)``,
    ``h = sqrt(T) (A, E)`` (the ``dt * rfft`` data convention)."""
    import binary_galactica as bg
    from wdm.backend import require_x64

    require_x64()
    if backend not in ("onfly", "fourier"):
        raise ValueError(f"backend must be 'onfly' or 'fourier'; got {backend!r}")
    orbit = bg.orbit_cache(T, 0.0, bw)

    def generate(native):
        p = bg.params_from_native(*native, T=T)
        w = bg.ucb_waveform(p, T=T, BW=bw, positions=orbit, backend=backend)
        bins = np.asarray(w["freq_bin_indices"], dtype=int)
        h = np.sqrt(T) * np.stack([np.asarray(w["A"]), np.asarray(w["E"])])
        return bins, h
    return generate


# ---------------------------------------------------------------------------
# build_segments
# ---------------------------------------------------------------------------


def simulate_instrument(freqs, T: float, seed) -> np.ndarray:
    """Stationary A/E instrument noise on the bins ``freqs``, ``(2, n)`` in the ``dt * rfft``
    convention: independent complex Gaussian bins with ``E|d|^2 = T S/2``, ``S`` the Sangria
    instrument PSD on both channels (``foreground_fit.s_inst``: the legacy
    ``psd_components(freqs)[0]``, the same arrays)."""
    S = np.repeat(np.atleast_2d(s_inst(freqs)), 2, 0)
    rng = np.random.default_rng(seed)
    return np.sqrt(T * S / 4) * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))


def _support(spec: SegmentSpec) -> np.ndarray:
    """Offsets (rfft bins) of the nonzero Meyer filter around a band centre."""
    import wdm.filter

    phi = wdm.filter.phi_centered(spec.N // spec.NT, spec.NT, DT)
    return np.nonzero(phi)[0] - spec.NT // 2


def window(spec: SegmentSpec, f0: float = F_CENTRE) -> dict:
    """Bands and stored bins for ``spec``: ``n_bands`` bands, the source's band
    ``(n_bands - 1) // 2`` from the bottom, and the bins under their Meyer filters."""
    step = spec.NT // 2
    m0 = int(round(f0 * spec.T / step))
    first = m0 - (spec.n_bands - 1) // 2
    last = first + spec.n_bands - 1
    nz = _support(spec)
    lo, hi = first * step + int(nz.min()), last * step + int(nz.max())
    return dict(T=spec.T, NT=spec.NT, source_band=m0, first_band=first, last_band=last,
                bin_lo=lo, n_bins=hi - lo + 1)


def store_window(spec: SegmentSpec, f0: float = F_CENTRE) -> dict:
    """``window`` plus the wider store: ``STORE_MARGIN`` extra bands on each side, the bins
    under all their Meyer filters (``store_lo``, ``store_n``)."""
    w = window(spec, f0)
    step = spec.NT // 2
    nz = _support(spec)
    lo = (w["first_band"] - STORE_MARGIN) * step + int(nz.min())
    hi = (w["last_band"] + STORE_MARGIN) * step + int(nz.max())
    return dict(w, store_lo=lo, store_n=hi - lo + 1)


def load_foreground(path, T: float) -> dict:
    """A duration's resolved binaries (native order, colatitude) and fitted foreground
    (:func:`.foreground_fit.write`'s file; shipped: ``foreground_<m>mo_snr7.npz``)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as p:
        if not np.isclose(float(p["T"]), T, rtol=0, atol=1e-6):
            raise ValueError(f"{path}: T = {float(p['T'])} is not {T}")
        if tuple(p["native_names"]) != INJECT_KEYS:
            raise ValueError(f"{path}: native columns {tuple(p['native_names'])}")
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    resolved=np.array(p["resolved_native"], float),
                    component=np.array(p["resolved_component"]), row=np.array(p["resolved_row"]),
                    fg_params=np.array(p["fg_params"], float), snr_threshold=float(p["snr_threshold"]))


def select_reaching(native, *, T: float, bin_lo: int, n_bins: int, bw: int) -> np.ndarray:
    """Indices of the sources whose generated ``bw`` bins (centred on ``trunc(f0 T)``) can
    reach the bins ``bin_lo .. bin_lo + n_bins - 1`` (one bin of slack each side)."""
    lower = np.trunc(np.asarray(native)[:, 0] * T).astype(int) - bw // 2
    return np.flatnonzero((lower - 1 < bin_lo + n_bins) & (lower + bw + 1 > bin_lo))


def subtract_resolved(native, *, T: float, bin_lo: int, n_bins: int, bw: int,
                      backend: str = "onfly", log: Callable = print):
    """Sum every source of ``native`` whose generated ``bw`` bins reach the window."""
    indices = select_reaching(native, T=T, bin_lo=bin_lo, n_bins=n_bins, bw=bw)
    generate = generator(T, bw, backend=backend)
    summed = np.zeros((2, n_bins), complex)
    for i, idx in enumerate(indices):
        bins, h = generate(native[idx])
        add_to_window(summed, bins, h, bin_lo=bin_lo)
        if (i + 1) % 500 == 0:
            log(f"  resolved {i + 1}/{len(indices)}")
    return summed, indices


def snr_function(T: float, S: np.ndarray) -> Callable:
    """``h -> sqrt(4/T sum |h|^2 / S)`` (both channels; the segments' SNR budget)."""
    return lambda h: float(np.sqrt(4 / T * np.sum(np.abs(h) ** 2 / S)))


def build_segment(h5, spec: SegmentSpec, out: Path, *, foreground: dict, backend: str = "onfly",
                  verification: dict | None = None, injection=INJECT_NATIVE,
                  max_error: float = MAX_ERROR_SNR, log: Callable = print) -> Path:
    """Write ``segment_snr<snr>_<m>mo_<n>bands.npz`` into ``out`` (the shipped schema)."""
    native = np.asarray(injection, float)
    w = window(spec, native[0])
    T, lo, n, bw = spec.T, w["bin_lo"], w["n_bins"], spec.bw
    out = Path(out)
    dest = out / f"segment_snr{foreground['snr_threshold']:g}_{spec.months}mo_{spec.n_bands}bands.npz"
    if dest.exists():
        raise FileExistsError(f"{dest} exists; select a fresh output directory")
    t0 = time.perf_counter()
    log(f"{spec.months} months: NT={w['NT']}, bands {w['first_band']}..{w['last_band']} "
        f"(source {w['source_band']}), bins {lo}..{lo + n - 1} ({n}); "
        f"{len(foreground['resolved'])} resolved binaries")
    freqs = (lo + np.arange(n)) / T
    S = np.sum(psd_components(freqs, foreground["fg_params"]), axis=0)          # (2, n)
    snr = snr_function(T, S)
    galaxy = read_gb_ae(h5, T)[:, lo:lo + n].copy()
    instrument = simulate_instrument(freqs, T, [INSTRUMENT_SEED, spec.months])
    observed = galaxy + instrument
    removed, subtracted = subtract_resolved(foreground["resolved"], T=T, bin_lo=lo, n_bins=n, bw=bw,
                                            backend=backend, log=log)
    residual = observed - removed
    bins, h = generator(T, bw, backend=backend)(native)
    injected = np.zeros_like(residual)
    add_to_window(injected, bins, h, bin_lo=lo)
    doubled, subtracted_check = subtract_resolved(foreground["resolved"], T=T, bin_lo=lo, n_bins=n,
                                                  bw=2 * bw, backend=backend, log=log)
    delta = snr(doubled - removed)
    validate_bandwidth_error(delta, max_error=max_error)
    cbins, ch = generator(T, 2 * bw, backend=backend)(native)
    check_injected = np.zeros_like(injected)
    add_to_window(check_injected, cbins, ch, bin_lo=lo)
    inj_delta = snr(check_injected - injected)
    validate_bandwidth_error(inj_delta, max_error=max_error)
    log(f"  BW {bw} vs {2 * bw}: subtraction {delta:.4g}, injection {inj_delta:.4g} (SNR)")
    provenance = {"raw_h5": str(h5),
                  "foreground_fit": {k: foreground[k] for k in ("path", "sha256", "snr_threshold")},
                  "builder": "wdm_models.galactic_binary.pipeline.segments.build_segment",
                  "waveform_backend": backend,
                  "months": spec.months, "duration_seconds": T, "start_seconds": 0.0,
                  "sample_cadence_seconds": DT, "window": w, "subtraction_bw": bw,
                  "resolved_binaries": len(foreground["resolved"]), "resolved_reaching_window": len(subtracted),
                  "snr_psd": "Sangria instrument + fitted GLASS foreground (prepare.psd_components)",
                  "data": "Sangria galactic binaries (sky/dgb, igb, vgb TDI) - resolved + simulated instrument noise + injection",
                  "instrument_noise": {"model": "nominal Sangria S_inst (prepare.psd_components)",
                                       "simulation": "independent complex Gaussian Fourier bins, E|d|^2 = T S/2",
                                       "seed": [INSTRUMENT_SEED, spec.months]},
                  "bandwidth_check": {"bw": 2 * bw, "residual_difference_snr": delta,
                                      "injection_difference_snr": inj_delta,
                                      "resolved_reaching_window": len(subtracted_check),
                                      "max_error_snr": max_error, "passed": True},
                  "injection_policy": "All eight physical source parameters fixed across durations; amplitude is not retuned.",
                  **({"raw_convention": "not verified (no catalogue file given)"} if verification is None
                     else verification)}
    psd = lambda f: np.sum(psd_components(f, foreground["fg_params"]), axis=0)  # noqa: E731
    payload = {f"inject_{k}": native[i] for i, k in enumerate(INJECT_KEYS)}
    payload.update(data_A=(residual + injected)[0], data_E=(residual + injected)[1],
                   residual_A=residual[0], residual_E=residual[1],
                   injected_A=injected[0], injected_E=injected[1],
                   instrument_A=instrument[0], instrument_E=instrument[1],
                   freqs=freqs, t_obs=T, t_start=0.0, df=1 / T, bin_offset=lo, n_bins=n,
                   f_centre=F_CENTRE, f_lo=lo / T, f_hi=(lo + n) / T,
                   waveform=f"binary_galactica:{backend}", inject_bw=bw, inject_snr=snr(injected),
                   inject_full_support_snr=full_support_snr(bins, h, T=T, psd=psd),
                   fg_params=foreground["fg_params"], fg_model="GLASS (foreground.glass_psd)",
                   snr_cut_snr=foreground["snr_threshold"], snr_cut_tobs=T,
                   resolved_subtracted_indices=subtracted,
                   resolved_subtracted_component=foreground["component"][subtracted],
                   resolved_subtracted_row=foreground["row"][subtracted],
                   n_ucb_subtracted_overlap=len(subtracted),
                   provenance_json=json.dumps(provenance, sort_keys=True, default=int))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **payload)
    log(f"  wrote {dest}: {len(subtracted)} resolved binaries subtracted; injection SNR "
        f"{payload['inject_snr']:.3f}; {time.perf_counter() - t0:.0f} s")
    return dest


def build_store(h5, spec: SegmentSpec, segment, out: Path, *, foreground: dict, backend: str = "onfly",
                injection=INJECT_NATIVE, max_error: float = MAX_ERROR_SNR,
                log: Callable = print) -> Path:
    """The segment's data band-passed to the wider store (``store_snr<snr>_<m>mo_<n>bands_pm3.npz``).

    Built with :func:`build_segment`'s recipe on ``STORE_MARGIN`` extra bands per side; the
    inner bins are the segment's, verbatim (the recomputation must agree to 1e-12), and the
    extra bins' instrument noise draws from ``[INSTRUMENT_SEED, months, OUTER_SEED]``.
    """
    native = np.asarray(injection, float)
    w = store_window(spec, native[0])
    T, bw = spec.T, spec.bw
    lo, n = w["store_lo"], w["store_n"]
    out = Path(out)
    dest = out / f"store_snr{foreground['snr_threshold']:g}_{spec.months}mo_{spec.n_bands}bands_pm{STORE_MARGIN}.npz"
    if dest.exists():
        raise FileExistsError(f"{dest} exists; select a fresh output directory")
    with np.load(segment, allow_pickle=False) as s:
        seg = {k: s[k] for k in s.files}
    if (int(seg["bin_offset"]), int(seg["n_bins"])) != (w["bin_lo"], w["n_bins"]):
        raise ValueError(f"{segment} is not the {spec.months}-month analysis window")
    inner = slice(w["bin_lo"] - lo, w["bin_lo"] - lo + w["n_bins"])
    t0 = time.perf_counter()
    log(f"{spec.months} months store: bins {lo}..{lo + n - 1} ({n}; inner {w['n_bins']} at {inner.start})")
    freqs = (lo + np.arange(n)) / T
    S = np.sum(psd_components(freqs, foreground["fg_params"]), axis=0)
    snr = snr_function(T, S)
    galaxy = read_gb_ae(h5, T)[:, lo:lo + n].copy()
    instrument = simulate_instrument(freqs, T, [INSTRUMENT_SEED, spec.months, OUTER_SEED])
    instrument[:, inner] = np.stack([seg["instrument_A"], seg["instrument_E"]])
    removed, subtracted = subtract_resolved(foreground["resolved"], T=T, bin_lo=lo, n_bins=n, bw=bw,
                                            backend=backend, log=log)
    residual = galaxy + instrument - removed
    bins, h = generator(T, bw, backend=backend)(native)
    injected = np.zeros_like(residual)
    add_to_window(injected, bins, h, bin_lo=lo)
    # the inner bins must reproduce the segment before it is spliced in verbatim
    for key, new in (("residual", residual), ("injected", injected)):
        old = np.stack([seg[f"{key}_A"], seg[f"{key}_E"]])
        err = np.max(np.abs(new[:, inner] - old)) / np.max(np.abs(old))
        if err > 1e-12:
            raise RuntimeError(f"recomputed inner {key} differs from the segment by {err:.2e}")
        new[:, inner] = old
    doubled, _ = subtract_resolved(foreground["resolved"], T=T, bin_lo=lo, n_bins=n, bw=2 * bw,
                                   backend=backend, log=log)
    delta = snr(doubled - removed)
    validate_bandwidth_error(delta, max_error=max_error)
    log(f"  BW {bw} vs {2 * bw}: subtraction {delta:.4g} (SNR)")
    provenance = {"segment": str(segment), "months": spec.months, "window": w,
                  "store_margin_bands": STORE_MARGIN, "subtraction_bw": bw, "waveform_backend": backend,
                  "resolved_reaching_store": len(subtracted),
                  "instrument_seed_extra_bins": [INSTRUMENT_SEED, spec.months, OUTER_SEED],
                  "inner_bins": "the segment's, verbatim (recomputation agrees to 1e-12)",
                  "bandwidth_check": {"bw": 2 * bw, "residual_difference_snr": delta,
                                      "max_error_snr": max_error, "passed": True}}
    payload = dict(data_A=(residual + injected)[0], data_E=(residual + injected)[1],
                   residual_A=residual[0], residual_E=residual[1],
                   injected_A=injected[0], injected_E=injected[1],
                   instrument_A=instrument[0], instrument_E=instrument[1],
                   freqs=freqs, t_obs=T, df=1 / T, bin_offset=lo, n_bins=n,
                   inner_bin_offset=w["bin_lo"], inner_n_bins=w["n_bins"],
                   fg_params=foreground["fg_params"],
                   provenance_json=json.dumps(provenance, sort_keys=True, default=int))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **payload)
    log(f"  wrote {dest}: {len(subtracted)} resolved binaries reach the store; "
        f"{time.perf_counter() - t0:.0f} s")
    return dest


#: Keys every rebuilt segment/store must reproduce exactly (the instrument draws, the bin
#: geometry and the selection) and the keys compared relative to their largest magnitude.
EXACT_KEYS = ("instrument_A", "instrument_E", "freqs", "t_obs", "df", "bin_offset", "n_bins",
              "fg_params", "resolved_subtracted_indices", "resolved_subtracted_row",
              "resolved_subtracted_component", "inner_bin_offset", "inner_n_bins")
CLOSE_KEYS = ("data_A", "data_E", "residual_A", "residual_E", "injected_A", "injected_E",
              "inject_snr", "inject_full_support_snr")


def compare_npz(new: Any, shipped: Any, *, rtol: float = 1e-12) -> dict:
    """A rebuilt segment or store against the shipped file: :data:`EXACT_KEYS` ``array_equal``,
    :data:`CLOSE_KEYS` within ``rtol`` of the shipped key's largest magnitude.  Keys absent from
    both are skipped; a key in only one is a problem."""
    out: dict[str, Any] = {"exact": {}, "close": {}, "problems": []}
    for key in EXACT_KEYS + CLOSE_KEYS:
        has_new, has_old = key in new, key in shipped
        if not (has_new or has_old):
            continue
        if has_new != has_old:
            out["problems"].append(f"{key} is only in the {'rebuilt' if has_new else 'shipped'} file")
            continue
        a, b = np.asarray(new[key]), np.asarray(shipped[key])
        if a.shape != b.shape:
            out["problems"].append(f"{key}: shape {a.shape} vs {b.shape}")
            continue
        if key in EXACT_KEYS:
            ok = bool(np.array_equal(a, b))
            out["exact"][key] = ok
            if not ok:
                out["problems"].append(f"{key} is not the shipped array")
        else:
            scale = float(np.max(np.abs(b))) or 1.0
            rel = float(np.max(np.abs(a - b))) / scale if np.all(np.isfinite(a)) else float("inf")
            out["close"][key] = rel
            if not rel <= rtol:
                out["problems"].append(f"{key} differs by {rel:.3e} relative (gate {rtol:.0e})")
    out["ok"] = not out["problems"]
    return out
