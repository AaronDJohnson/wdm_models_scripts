r"""Resolve the Sangria galactic binaries and fit the foreground for one duration (spec §5.2
``pipeline/foreground_fit.py``; port of ``toy_models/galactic_binary/fit_foreground.py``).

The manuscript's procedure (Sec. VI B), on the Sangria record cropped to the duration:

1. the SNR of every Sangria galactic binary (the 26 M detached, 3 M interacting and 36
   verification binaries of ``sky/{dgb,igb,vgb}/cat``) at its true parameters, against the
   instrument PSD alone (:func:`s_inst`);
2. the power of the unresolved binaries: Sangria's galactic-binary TDI with the resolved
   binaries (SNR at or above the threshold) subtracted, smoothed in frequency
   (:func:`confusion_psd`);
3. the SNR of the resolved binaries against the instrument PSD plus that power;
4. the binaries that fall below the threshold move to the unresolved set; 2-3 repeat until
   no binary moves.

GLASS's foreground model (``foreground.glass_psd``) is then fitted once to the converged
unresolved power: the MAP of a short impulse parallel-tempered run (seed 0) inside GLASS's
prior box, polished by Nelder-Mead (:func:`map_fit_glass`, :func:`map_fit_binned`).

**The SNR** is :math:`\rho^2=\frac4T\sum_k\sum_{c\in\{A,E\}}|h_c(f_k)|^2/S(f_k)` on every
positive bin of the source's generated band (binary_galactica, GLASS's band heuristic
``ucb_bw`` with safety 2).  **Which binaries need a waveform:** only those whose cheap proxy
:math:`A\sqrt T\,x|\sin x|/\sqrt{S_{\rm inst}}` (:func:`proxy_snr`, :math:`x=f/f_*`) times
:data:`PROXY_MARGIN` reaches the threshold; :func:`resolve` checks the cut on the
``check_below`` brightest-proxy binaries below it.

**The fit** is a Whittle likelihood on the binned periodogram of both channels,
:math:`\ln L=-\sum_b 2M_b[\ln\bar S_b+\bar P_b/\bar S_b]`, over :data:`FIT_BAND`.

Differences from the legacy module, none of which changes a number on the paper's path:

* the HDF5 path is an argument (``h5``), never a module constant; the record length ``T``
  may be given (tests run the loop on a small synthetic file);
* waveforms come from binary_galactica (``backend="onfly"`` by default; ``"fourier"`` is
  gbonfly's algorithm and reproduces the shipped fits, ``pipeline.SHIPPED_BACKEND``);
* the Sangria-formula fit (legacy ``fit_foreground``/``s_fg``/``_hessian``, "kept for
  comparison") is not ported; the GLASS MAP is the only fit, and GLASS's names, prior box and
  start are ``foreground``'s (one source);
* the MAP fit is split: :func:`map_fit_binned` fits a binned periodogram and takes the sampler
  class (impulse's ``PTSampler`` by default, imported lazily), :func:`map_fit_glass` bins the
  data first -- so the shipped binned periodograms can be refitted and the wiring tested
  without impulse;
* :func:`write` no longer copies ``foreground_<m>mo.npz`` (the segment builder takes the
  ``foreground_<m>mo_snr7.npz`` path explicitly) and records the waveform backend;
* the batched generator takes the orbit as a ``jit`` argument (spec §3.3: constants are never
  closed over); the spectra equal binary_galactica's single-source call to 1e-12 of the peak.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..foreground import GLASS_NAMES, GLASS_PRIOR, GLASS_START, glass_psd
from ..instrument import SANGRIA_NOISE_PARAMS, tdi_instrumental_psd_AE

__all__ = ["BATCH_BINS", "BLOCK_DLNF", "BLOCK_MIN_BINS", "COMPONENTS", "DT", "FIELDS", "FIT_ATOL_KEYS",
           "FIT_BAND", "FIT_EXACT_KEYS", "FIT_RTOL_KEYS", "GLASS_NAMES", "GLASS_PRIOR", "GLASS_START",
           "Generator", "NATIVE", "PROXY_MARGIN", "SMOOTH_BAND", "SMOOTH_BLOCKS", "YEAR", "accumulate",
           "band_snr2", "binned_power", "block_mean", "blocks", "compare_fit", "confusion_psd",
           "glass_objective", "labels", "map_fit_binned", "map_fit_glass", "native_of", "polish",
           "proxy_snr", "read_column", "read_gb_ae", "resolve", "s_inst", "write"]

DT = 5.0
YEAR = 31536000.0
COMPONENTS = ("dgb", "igb", "vgb")
#: Native parameter columns written out (colatitude, not the catalogue's latitude).
NATIVE = ("f0", "fdot", "amp", "phi0", "colat", "lon", "psi", "cos_iota")
#: A binary is generated if ``proxy * PROXY_MARGIN >= threshold``.
PROXY_MARGIN = 16.0
#: Frequency band of the fit [Hz] and the blocks of the binned periodogram.
FIT_BAND = (3.0e-4, 4.0e-3)
BLOCK_DLNF, BLOCK_MIN_BINS = 0.01, 32
#: Band and smoothing of the confusion-power estimate that drives the iteration.
SMOOTH_BAND = (5.0e-5, 3.0e-2)
SMOOTH_BLOCKS = 5
#: Generator batches hold at most this many output bins.
BATCH_BINS = 1 << 20
# GLASS's foreground coordinates, prior box and start (GLASS_NAMES, GLASS_PRIOR, GLASS_START)
# are ``foreground``'s, re-exported here.
#: Catalogue fields, in :data:`NATIVE` order (latitude and inclination converted).
FIELDS = ("Frequency", "FrequencyDerivative", "Amplitude", "InitialPhase", "EclipticLatitude",
          "EclipticLongitude", "Polarization", "Inclination")


# ---------------------------------------------------------------------------
# noise model
# ---------------------------------------------------------------------------


def s_inst(f) -> np.ndarray:
    """The Sangria A/E instrument PSD (``instrument.tdi_instrumental_psd_AE`` at the A
    parameters, as the legacy ``s_inst``)."""
    P = SANGRIA_NOISE_PARAMS
    return np.asarray(tdi_instrumental_psd_AE(np.asarray(f, float), S_oms=P.S_oms_A, S_a=P.S_a_A))


# ---------------------------------------------------------------------------
# the catalogue and the waveforms
# ---------------------------------------------------------------------------


def read_column(h5, name: str) -> np.ndarray:
    """One catalogue field for every Sangria galactic binary, components concatenated."""
    import h5py

    with h5py.File(h5, "r") as f:
        return np.concatenate([f[f"sky/{comp}/cat"].fields([name])[:].reshape(-1)[name]
                               for comp in COMPONENTS])


def labels(h5) -> np.ndarray:
    """``(n, 2)``: component (index into ``COMPONENTS``) and catalogue row of every binary."""
    import h5py

    with h5py.File(h5, "r") as f:
        sizes = [f[f"sky/{comp}/cat"].shape[0] for comp in COMPONENTS]
    return np.concatenate([np.stack([np.full(n, c), np.arange(n)], axis=1)
                           for c, n in enumerate(sizes)])


def native_of(h5, index) -> np.ndarray:
    """Native parameters (``NATIVE`` order, colatitude, cos inclination) of the binaries ``index``."""
    out = np.empty((len(index), 8))
    for j, name in enumerate(FIELDS):
        out[:, j] = read_column(h5, name)[index]
    out[:, 4] = np.pi / 2 - out[:, 4]
    out[:, 7] = np.cos(out[:, 7])
    return out


def read_gb_ae(h5, T: float, dt: float = DT) -> np.ndarray:
    """``dt * rfft`` of the A and E channels of all Sangria galactic binaries over ``[0, T)``."""
    import h5py
    from scipy.fft import rfft

    n = int(round(T / dt))
    acc = {k: np.zeros(n) for k in "XYZ"}
    with h5py.File(h5, "r") as f:
        for comp in COMPONENTS:
            rec = f[f"sky/{comp}/tdi"][:n].reshape(-1)
            if not np.array_equal(rec["t"], np.arange(n) * dt):
                raise ValueError(f"{comp}: timestamps must start at zero with cadence {dt}")
            for k in "XYZ":
                acc[k] += rec[k]
    x, y, z = acc["X"], acc["Y"], acc["Z"]
    return np.stack([dt * rfft((z - x) / np.sqrt(2)), dt * rfft((x - 2 * y + z) / np.sqrt(6))])


def proxy_snr(amp, f, T: float) -> np.ndarray:
    """Order-of-magnitude SNR against the instrument alone (see the module docstring)."""
    from binary_galactica import FSTAR

    x = f / FSTAR
    return amp * np.sqrt(T) * x * np.abs(np.sin(x)) / np.sqrt(s_inst(f))


class Generator:
    """binary_galactica spectra ``h = sqrt(T) * (A, E)`` of many binaries, batched by bandwidth."""

    def __init__(self, T: float, *, backend: str = "onfly"):
        from wdm.backend import require_x64

        require_x64()                       # binary_galactica's float64 entry points need x64
        if backend not in ("onfly", "fourier"):
            raise ValueError(f"backend must be 'onfly' or 'fourier'; got {backend!r}")
        self.T, self.N, self.backend = T, int(round(T / DT)), backend
        self._fns: dict[int, tuple[Callable, Any]] = {}

    def bandwidth(self, native) -> np.ndarray:
        from binary_galactica import ucb_bw

        return np.array([ucb_bw(p[0], p[1], np.cos(p[4]), p[2], self.T, self.N, safety=2.0)
                         for p in native])

    def _fn(self, BW: int) -> tuple[Callable, Any]:
        """``(jit(vmap(ucb_waveform)), orbit)`` at ``BW``; the orbit is a jit argument."""
        if BW not in self._fns:
            import binary_galactica as bg
            import jax

            orbit = bg.orbit_cache(self.T, 0.0, BW)
            T, backend = self.T, self.backend
            self._fns[BW] = (jax.jit(jax.vmap(
                lambda p, o: bg.ucb_waveform(p, T=T, BW=BW, positions=o, backend=backend),
                in_axes=(0, None))), orbit)
        return self._fns[BW]

    def batches(self, native, bw=None) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(positions, qmin, h)`` with ``h`` of shape ``(n, 2, BW)``."""
        import jax.numpy as jnp

        T = self.T
        bw = self.bandwidth(native) if bw is None else bw
        glass = np.stack([native[:, 0] * T, np.cos(native[:, 4]), native[:, 5],
                          np.log(native[:, 2]), native[:, 7], native[:, 6], native[:, 3],
                          native[:, 1] * T * T], axis=1)
        for BW in np.unique(bw):
            pos = np.flatnonzero(bw == BW)
            step = max(1, BATCH_BINS // int(BW))
            for s in range(0, pos.size, step):
                p = pos[s:s + step]
                fn, orbit = self._fn(int(BW))
                w = fn(jnp.asarray(glass[p]), orbit)
                qmin = np.asarray(w["qmin"]).astype(np.int64)
                h = np.sqrt(T) * np.stack([np.asarray(w["A"]), np.asarray(w["E"])], axis=1)
                yield p, qmin, h


def band_snr2(qmin, power, T: float, S_of_f) -> np.ndarray:
    """``4/T sum_k P_k / S(f_k)`` over each band's positive bins; ``power`` is ``(n, BW)``."""
    BW = power.shape[1]
    k = qmin[:, None] + np.arange(BW)[None, :]
    ok = (k > 0) & (k < int(round(T / DT)) // 2)
    S = S_of_f(np.where(ok, k, 1) / T)
    return 4.0 / T * np.sum(np.where(ok, power / S, 0.0), axis=1)


def accumulate(out, qmin, h, sign: float) -> None:
    """``out[:, k] += sign * h`` for every band, overlaps summed (``out`` is ``(2, nf)``)."""
    nf = out.shape[1]
    k = qmin[:, None] + np.arange(h.shape[2])[None, :]
    ok = (k >= 0) & (k < nf)
    idx = k[ok]
    for c in range(2):
        v = h[:, c, :][ok]
        out[c] += sign * (np.bincount(idx, weights=v.real, minlength=nf)
                          + 1j * np.bincount(idx, weights=v.imag, minlength=nf))


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def blocks(T: float, band=FIT_BAND, dlnf=BLOCK_DLNF, min_bins=BLOCK_MIN_BINS) -> np.ndarray:
    """Contiguous bin blocks covering ``band``: ``dlnf`` wide in log f, at least ``min_bins``."""
    k0, k1 = int(np.ceil(band[0] * T)), int(np.floor(band[1] * T))
    edges, k = [k0], k0
    while k < k1:
        k = min(k1, max(k + min_bins, int(np.ceil(k * np.exp(dlnf)))))
        edges.append(k)
    if edges[-1] - edges[-2] < min_bins and len(edges) > 2:
        edges.pop(-2)
    return np.asarray(edges)


def binned_power(d, T: float, edges):
    """Block means of ``|d_k|^2 / (T/2)`` over both channels, and bins per block."""
    k = np.arange(edges[0], edges[-1])
    p = (np.abs(d[0, k]) ** 2 + np.abs(d[1, k]) ** 2) / (T / 2)
    b = np.searchsorted(edges, k, side="right") - 1
    M = np.diff(edges)
    return np.bincount(b, weights=p) / (2 * M), M


def block_mean(S_k, edges) -> np.ndarray:
    k = np.arange(edges[0], edges[-1])
    b = np.searchsorted(edges, k, side="right") - 1
    return np.bincount(b, weights=S_k) / np.diff(edges)


def glass_objective(Pbar, M, fc) -> tuple[Callable, Callable]:
    """``(lnlike, lnprior)`` of GLASS's foreground model on a binned periodogram:
    :math:`-\\sum_b 2M_b[\\ln S_b + \\bar P_b/S_b]` with ``S = glass_psd(fc, x)``, and GLASS's
    uniform box (:data:`GLASS_PRIOR`)."""
    lo, hi = GLASS_PRIOR[:, 0], GLASS_PRIOR[:, 1]

    def lnprior(x):
        return 0.0 if np.all((x >= lo) & (x <= hi)) else -np.inf

    def lnlike(x):
        S = glass_psd(fc, x)
        if not np.all(np.isfinite(S)) or np.any(S <= 0):
            return -np.inf
        return float(-np.sum(2 * M * (np.log(S) + Pbar / S)))

    return lnlike, lnprior


def polish(x, lnlike: Callable, lnprior: Callable) -> np.ndarray:
    """Nelder-Mead on ``-(lnlike + lnprior)`` from ``x`` (``xatol = fatol = 1e-9``, 20000 iterations)."""
    from scipy.optimize import minimize

    neg = lambda y: -(lnlike(y) + lnprior(y)) if np.isfinite(lnprior(y)) else np.inf  # noqa: E731
    return minimize(neg, x, method="Nelder-Mead",
                    options=dict(maxiter=20000, xatol=1e-9, fatol=1e-9)).x


def map_fit_binned(Pbar, M, fc, x0=GLASS_START, *, iterations: int = 20000, ntemps: int = 4,
                   seed: int = 0, sampler_cls: Any = None):
    """MAP of GLASS's foreground model on a binned periodogram (``Pbar``, bins per block ``M``,
    block centres ``fc``), by impulse PT-MCMC.

    A parallel-tempered run from ``x0`` inside GLASS's prior box (ladder
    ``geomspace(1, 50, ntemps)``, proposal covariance ``diag((0.01 (hi - lo))^2)``, ``seed``);
    the highest-posterior cold sample is then polished with Nelder-Mead (:func:`polish`).
    Returns the MAP, the cold chain's posterior standard deviations (after 30% burn-in) and
    the fit.  ``sampler_cls`` defaults to ``impulse.PTSampler`` (the ``[sampling]`` extra).
    """
    if sampler_cls is None:
        try:
            from impulse import PTSampler as sampler_cls
        except ImportError as exc:
            raise ImportError("map_fit_glass needs impulse (impulse-mcmc v2 964283c, the "
                              "wdm-models[sampling] extra)") from exc
    lo, hi = GLASS_PRIOR[:, 0], GLASS_PRIOR[:, 1]
    lnlike, lnprior = glass_objective(Pbar, M, fc)
    x0 = np.clip(np.asarray(x0, float), lo + 1e-9, hi - 1e-9)
    with tempfile.TemporaryDirectory() as tmp:
        sampler = sampler_cls(5, lnlike, lnprior, ntemps=ntemps, outdir=tmp, seed=seed,
                              ladder=np.geomspace(1.0, 50.0, ntemps), verbose=False,
                              save_freq=iterations, sample_cov=np.diag((0.01 * (hi - lo)) ** 2))
        sampler.sample(x0, num_iterations=iterations)
        chain = sampler.load_chain()
    cold, lnp = chain["samples"][0], chain["lnprob"][0]
    xmap = polish(cold[np.argmax(lnp)], lnlike, lnprior)
    burn = cold[int(0.3 * len(cold)):]
    return xmap, burn.std(axis=0), dict(f=fc, Pbar=Pbar, M=M, S_model=glass_psd(fc, xmap),
                                        lnlike=lnlike(xmap),
                                        acceptance=float(np.mean(chain["accepted"][0])))


def map_fit_glass(d, T: float, x0=GLASS_START, band=FIT_BAND, *, iterations: int = 20000,
                  ntemps: int = 4, seed: int = 0, sampler_cls: Any = None):
    """MAP of GLASS's foreground model for the unresolved binaries ``d`` (``(2, nf)``,
    ``dt * rfft``): :func:`map_fit_binned` on the binned periodogram of ``band``
    (:func:`blocks`, :func:`binned_power`; block centres ``sqrt(k_b k_{b+1}) / T``)."""
    edges = blocks(T, band=band)
    Pbar, M = binned_power(d, T, edges)
    fc = np.sqrt(edges[:-1] * edges[1:]) / T
    xmap, sd, fit = map_fit_binned(Pbar, M, fc, x0, iterations=iterations, ntemps=ntemps, seed=seed,
                                   sampler_cls=sampler_cls)
    return xmap, sd, dict(fit, edges=edges)


def confusion_psd(d, T: float, band=SMOOTH_BAND, width: int = SMOOTH_BLOCKS):
    """The smoothed PSD of the unresolved binaries ``d``, as a function of frequency.

    Block means of ``|d_k|^2 / (T/2)`` over both channels (blocks as :func:`blocks`), a
    running mean over ``width`` blocks, interpolated in ``log f``; constant beyond ``band``.
    """
    edges = blocks(T, band=band)
    P, M = binned_power(d, T, edges)
    w = np.ones(width)
    Ps = np.convolve(P * M, w, "same") / np.convolve(M.astype(float), w, "same")
    fc = np.sqrt(edges[:-1] * edges[1:]) / T
    lf, lP = np.log(fc), np.log(np.maximum(Ps, np.finfo(float).tiny))

    def S(f):
        return np.exp(np.interp(np.log(np.asarray(f, float)), lf, lP))
    return S, dict(f=fc, P=Ps)


# ---------------------------------------------------------------------------
# the iteration
# ---------------------------------------------------------------------------


def resolve(h5, months: int, snr: float, *, T: float | None = None, backend: str = "onfly",
            max_iter: int = 100, check_below: int = 5000, map_iterations: int = 100000,
            map_ntemps: int = 8, sampler_cls: Any = None, log: Callable = print) -> dict:
    """Steps 1-4 of the module docstring for one duration and threshold, then the GLASS MAP.

    ``T`` defaults to ``YEAR * months / 12`` (the paper's 3/6/12-month records)."""
    T = YEAR * months / 12 if T is None else float(T)
    t0 = time.perf_counter()
    amp, f0 = read_column(h5, "Amplitude"), read_column(h5, "Frequency")
    proxy = proxy_snr(amp, f0, T)
    n_catalogue = proxy.size
    del amp, f0
    cand = np.flatnonzero(proxy * PROXY_MARGIN >= snr)
    rest = np.flatnonzero(proxy * PROXY_MARGIN < snr)
    k = min(check_below, rest.size)
    below = rest[np.argpartition(proxy[rest], rest.size - k)[rest.size - k:]] if k else rest[:0]
    del rest
    log(f"[{months} mo] {n_catalogue:,} binaries; {cand.size:,} can reach SNR {snr:g} "
        f"against the instrument ({time.perf_counter() - t0:.0f} s)")
    native = native_of(h5, cand)

    gen = Generator(T, backend=backend)
    bw = gen.bandwidth(native)
    # step 1: SNR against the instrument alone; keep the resolved binaries' power
    rho_inst = np.empty(cand.size)
    kept = []                                   # (positions, qmin, power) of resolved binaries
    for p, qmin, h in gen.batches(native, bw):
        P = np.abs(h[:, 0]) ** 2 + np.abs(h[:, 1]) ** 2
        rho_inst[p] = np.sqrt(band_snr2(qmin, P, T, s_inst))
        ok = rho_inst[p] >= snr
        if ok.any():
            kept.append((p[ok], qmin[ok], P[ok].astype(np.float32)))
    resolved = np.flatnonzero(rho_inst >= snr)
    log(f"[{months} mo] step 1: {resolved.size:,} binaries at SNR >= {snr:g} against the "
        f"instrument ({time.perf_counter() - t0:.0f} s)")

    # the proxy cut: the brightest-proxy binaries below it must stay below the threshold
    rho_below = np.zeros(below.size)
    for p, qmin, h in gen.batches(native_of(h5, below)):
        rho_below[p] = np.sqrt(band_snr2(qmin, np.abs(h[:, 0]) ** 2 + np.abs(h[:, 1]) ** 2, T, s_inst))
    log(f"[{months} mo] proxy check: the {below.size} brightest-proxy binaries below the cut "
        f"reach SNR {rho_below.max(initial=0.0):.2f} at most")
    if rho_below.max(initial=0.0) >= snr:
        raise RuntimeError("the proxy cut drops a binary that can reach the threshold")

    # step 2 data: the galactic binaries alone (Sangria's component TDIs) minus the resolved
    d = read_gb_ae(h5, T)
    for p, qmin, h in gen.batches(native[resolved], bw[resolved]):
        accumulate(d, qmin, h, -1.0)
    log(f"[{months} mo] subtracted {resolved.size:,} binaries ({time.perf_counter() - t0:.0f} s)")

    history = []
    for it in range(1, max_iter + 1):
        S_gb, smooth = confusion_psd(d, T)
        S = lambda f, g=S_gb: s_inst(f) + g(f)  # noqa: E731
        rho_all = np.full(cand.size, np.nan)
        for p, q, P in kept:
            rho_all[p] = np.sqrt(band_snr2(q, P.astype(float), T, S))
        rho = rho_all[resolved]
        drop = resolved[rho < snr]
        level = {f"{c:g} mHz": float(S_gb(c * 1e-3) / s_inst(c * 1e-3)) for c in (1, 2, 3, 4)}
        history.append(dict(iteration=it, n_resolved=int(resolved.size), n_dropped=int(drop.size),
                            confusion_over_instrument=level))
        log(f"[{months} mo] iteration {it}: confusion/instrument "
            + ", ".join(f"{k} {v:.2f}" for k, v in level.items())
            + f" ({time.perf_counter() - t0:.0f} s); {drop.size:,} of {resolved.size:,} fall below SNR {snr:g}")
        if drop.size == 0:
            break
        for p, qmin, h in gen.batches(native[drop], bw[drop]):
            accumulate(d, qmin, h, +1.0)
        resolved = resolved[rho >= snr]
    else:
        raise RuntimeError(f"no convergence in {max_iter} iterations")
    # GLASS's foreground model, fitted once to the converged unresolved power (Table III)
    theta, sd, fit = map_fit_glass(d, T, iterations=map_iterations, ntemps=map_ntemps,
                                   sampler_cls=sampler_cls)
    edge = [n for n, v, (a, b) in zip(GLASS_NAMES, theta, GLASS_PRIOR) if min(v - a, b - v) < 1e-3 * (b - a)]
    log(f"[{months} mo] GLASS-model MAP: " + ", ".join(f"{n} {v:.4f} +- {e:.4f}" for n, v, e in zip(GLASS_NAMES, theta, sd))
        + (f"; AT THE PRIOR EDGE: {edge}" if edge else ""))

    order = np.argsort(native[resolved, 0])
    keep = resolved[order]
    return dict(months=months, T=T, snr_threshold=snr, theta=theta, sd=sd, fit=fit, smooth=smooth,
                history=history, n_catalogue=n_catalogue, n_candidates=cand.size,
                resolved_native=native[keep], resolved_label=labels(h5)[cand[keep]],
                resolved_snr=rho[order], resolved_snr_inst=rho_inst[keep],
                proxy_check_max=float(rho_below.max(initial=0.0)), backend=backend,
                seconds=time.perf_counter() - t0)


def write(res: dict, out: Path) -> Path:
    """``foreground_<m>mo_snr<snr>.npz`` and ``.json`` in ``out`` (the shipped schema, plus the
    waveform backend)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"foreground_{res['months']}mo_snr{res['snr_threshold']:g}"
    fit = res["fit"]
    lat = res["resolved_native"].copy()
    lat[:, 4] = np.pi / 2 - lat[:, 4]
    np.savez(out / f"{stem}.npz", months=res["months"], T=res["T"],
             snr_threshold=res["snr_threshold"], fg_params=res["theta"], fg_names=np.array(GLASS_NAMES),
             fg_model=np.array("foreground.glass_psd: GLASS / Digman & Cornish (2022), A/E PSD"),
             fg_prior=GLASS_PRIOR, fg_sd=res["sd"],
             # sangria_minimal's ucb_params order (ecliptic LATITUDE), for the segment builder
             resolved=lat, resolved_names=np.array(("f0", "fdot", "A", "phase0", "ecl_lat", "lon",
                                                    "psi", "cos_iota")),
             resolved_native=res["resolved_native"], native_names=np.array(NATIVE),
             resolved_component=np.array(COMPONENTS)[res["resolved_label"][:, 0]],
             resolved_row=res["resolved_label"][:, 1], resolved_snr=res["resolved_snr"],
             resolved_snr_inst=res["resolved_snr_inst"], fit_f=fit["f"], fit_Pbar=fit["Pbar"],
             fit_M=fit["M"], fit_S_inst=s_inst(fit["f"]), fit_S_model=fit["S_model"],
             fit_band=np.array(FIT_BAND), confusion_f=res["smooth"]["f"],
             confusion_psd=res["smooth"]["P"], waveform_backend=np.array(res["backend"]))
    summary = dict(months=res["months"], T=res["T"], snr_threshold=res["snr_threshold"],
                   fg_model="GLASS (Digman & Cornish 2022): 1.5 A f^(5/3) exp(-(f/f1)^alpha) [1 + tanh((fk - f)/f2)]",
                   fg_params=dict(zip(GLASS_NAMES, res["theta"].tolist())),
                   fg_sd=dict(zip(GLASS_NAMES, res["sd"].tolist())),
                   fg_prior=dict(zip(GLASS_NAMES, GLASS_PRIOR.tolist())),
                   n_resolved=int(len(res["resolved_snr"])), n_catalogue=int(res["n_catalogue"]),
                   n_candidates=int(res["n_candidates"]), proxy_check_max=res["proxy_check_max"],
                   iterations=res["history"], fit_band_hz=list(FIT_BAND), seconds=round(res["seconds"], 1),
                   waveform_backend=res["backend"])
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out / f"{stem}.npz"


#: :func:`compare_fit`: keys that must be identical, keys within a relative tolerance of their
#: largest magnitude, and the MAP (absolute) -- a rebuilt fit against a shipped one.
FIT_EXACT_KEYS = ("months", "T", "snr_threshold", "fg_names", "fg_prior", "native_names", "resolved_names",
                  "resolved_native", "resolved", "resolved_component", "resolved_row", "fit_f", "fit_M",
                  "fit_band", "fit_S_inst", "confusion_f")
FIT_RTOL_KEYS = {"resolved_snr": 1e-9, "resolved_snr_inst": 1e-9, "fit_Pbar": 1e-9, "confusion_psd": 1e-9,
                 "fit_S_model": 1e-6, "fg_sd": 1e-6}
FIT_ATOL_KEYS = {"fg_params": 1e-6}


def compare_fit(new, shipped) -> dict:
    """A rebuilt ``foreground_<m>mo_snr7.npz`` against the shipped one (:data:`FIT_EXACT_KEYS`
    ``array_equal``; :data:`FIT_RTOL_KEYS`, :data:`FIT_ATOL_KEYS` within their gates)."""
    out: dict[str, Any] = {"exact": {}, "close": {}, "problems": []}
    for key in FIT_EXACT_KEYS + tuple(FIT_RTOL_KEYS) + tuple(FIT_ATOL_KEYS):
        if key not in new or key not in shipped:
            out["problems"].append(f"{key} is missing from the {'rebuilt' if key not in new else 'shipped'} file")
            continue
        a, b = np.asarray(new[key]), np.asarray(shipped[key])
        if a.shape != b.shape:
            out["problems"].append(f"{key}: shape {a.shape} vs {b.shape}")
            continue
        if key in FIT_EXACT_KEYS:
            ok = bool(np.array_equal(a, b))
            out["exact"][key] = ok
            if not ok:
                out["problems"].append(f"{key} is not the shipped array")
            continue
        diff = float(np.max(np.abs(a - b))) if np.all(np.isfinite(a)) else float("inf")
        if key in FIT_RTOL_KEYS:
            diff, gate = diff / (float(np.max(np.abs(b))) or 1.0), FIT_RTOL_KEYS[key]
        else:
            gate = FIT_ATOL_KEYS[key]
        out["close"][key] = diff
        if not diff <= gate:
            out["problems"].append(f"{key} differs by {diff:.3e} (gate {gate:.0e})")
    out["ok"] = not out["problems"]
    return out
