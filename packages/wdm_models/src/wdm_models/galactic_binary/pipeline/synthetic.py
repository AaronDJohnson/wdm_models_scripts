r"""A small Sangria-shaped HDF5 for running the pipeline without the private record, and a
sampler stand-in for the foreground MAP when impulse is absent (tests and demonstrations).

:func:`write_synthetic_sangria` writes the groups and structured dtypes the pipeline reads
(``obs/tdi``, ``sky/mbhb/tdi``, ``sky/{dgb,igb,vgb}/{cat,tdi}``; ``(n, 1)`` records, ``t = j dt``)
for a ``SYN_N = 2**15``-sample record at ``dt = 5`` s (``T = 163840`` s).  The galactic-binary
TDI is the sum of the :data:`SYN_SOURCES` spectra plus white "confusion" noise of standard
deviation :data:`SYN_SIGMA` in the ``dgb`` X, Y, Z (about three times the Sangria instrument PSD
at 1-3 mHz).  The spectra are binary_galactica's at ``backend`` (``"onfly"`` by default, as
everywhere in the release; the tests pass ``"fourier"``, the backend they resolve and subtract
with), ``sqrt(T) (X, Y, Z)`` placed on their bins and inverse-transformed, so ``read_gb_ae``
returns them exactly.  The sources are chosen so that the resolve loop drops a binary in its
first iteration and converges in its second (:data:`SYN_SOURCES`).

:class:`PolishOnlySampler` has impulse ``PTSampler``'s constructor/``sample``/``load_chain``
surface and returns ``x0`` as the only cold sample, so ``map_fit_binned`` reduces to the
Nelder-Mead polish from ``x0``; it records its arguments for inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PolishOnlySampler", "SYN_DT", "SYN_N", "SYN_SIGMA", "SYN_SOURCES", "SYN_T",
           "write_synthetic_sangria"]

SYN_N, SYN_DT = 2 ** 15, 5.0
SYN_T = SYN_N * SYN_DT
#: ``(component, f0, fdot, amp, phi0, colat, lon, psi, cos_iota)`` in the native order.  With
#: the confusion noise, at SNR 7: dgb row 0 (1.05 mHz, inside a 3-band window at NT = 32), igb
#: row 0 and vgb row 0 are resolved and stay; dgb row 1 is resolved against the instrument and
#: dropped by the first iteration; dgb rows 2 and 3 and igb row 1 are below the proxy cut (the
#: cut's check generates them).  The far sources sit beyond the store's reach at ``2 bw``.
SYN_SOURCES = (("dgb", 1.05e-3, 1e-17, 4.0e-21, 0.4, 1.2, 2.0, 0.6, 0.5),
               ("dgb", 2.0e-3, 1e-17, 3.6e-22, 1.1, 0.9, 4.0, 1.2, -0.2),
               ("dgb", 3.0e-3, 1e-17, 3.0e-23, 2.0, 2.1, 1.0, 0.3, 0.7),
               ("dgb", 1.5e-3, 1e-17, 1.0e-25, 0.1, 1.6, 3.0, 2.0, 0.1),
               ("igb", 3.6e-3, 2e-17, 1.0e-21, 5.0, 0.5, 5.5, 2.9, 0.9),
               ("igb", 0.8e-3, 1e-18, 1.0e-24, 3.0, 2.5, 0.2, 1.0, -0.6),
               ("vgb", 3.2e-3, 1e-17, 5.0e-22, 0.7, 1.9, 2.6, 0.8, 0.0))
#: White confusion noise in the dgb X, Y, Z: ``2 dt SYN_SIGMA^2`` is ~3 S_inst at 1-3 mHz.
SYN_SIGMA = 3.95e-22

_TDI = np.dtype([("t", "<f8"), ("X", "<f8"), ("Y", "<f8"), ("Z", "<f8")])
_BASE = [("Amplitude", "<f8"), ("EclipticLatitude", "<f8"), ("EclipticLongitude", "<f8"),
         ("Frequency", "<f8")]
_TAIL = [("Inclination", "<f8"), ("InitialPhase", "<f8"), ("Polarization", "<f8")]
#: Sangria's catalogue dtypes (igb's FrequencyDerivative is float32; vgb carries extra fields).
CAT_DTYPES = {
    "dgb": np.dtype([("Name", "<i8")] + _BASE + [("FrequencyDerivative", "<f8")] + _TAIL),
    "igb": np.dtype([("Name", "S7")] + _BASE + [("FrequencyDerivative", "<f4")] + _TAIL),
    "vgb": np.dtype([("Name", "S9"), ("Amplitude", "<f8"), ("Distance", "<f8"), ("EclipticLatitude", "<f8"),
                     ("EclipticLongitude", "<f8"), ("Frequency", "<f8"), ("FrequencyDerivative", "<f8")]
                    + _TAIL),
}


def _tdi(t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    rec = np.zeros((t.size, 1), _TDI)
    rec["t"][:, 0] = t
    for i, ch in enumerate("XYZ"):
        rec[ch][:, 0] = xyz[i]
    return rec


def write_synthetic_sangria(path, *, seed: int = 20261013, backend: str = "onfly", bw: int = 512) -> Path:
    """Write the synthetic record to ``path`` (see the module docstring); returns ``path``."""
    import binary_galactica as bg
    import h5py
    from wdm.backend import require_x64

    require_x64()
    N, DT, T = SYN_N, SYN_DT, SYN_T
    t = np.arange(N) * DT
    orbit = bg.orbit_cache(T, 0.0, bw)
    spec = {c: np.zeros((3, N // 2 + 1), complex) for c in CAT_DTYPES}
    for comp, *native in SYN_SOURCES:
        w = bg.ucb_waveform(bg.params_from_native(*native, T=T), T=T, BW=bw, positions=orbit,
                            backend=backend)
        k = np.asarray(w["freq_bin_indices"], int)
        for i, ch in enumerate("XYZ"):
            spec[comp][i, k] += np.sqrt(T) * np.asarray(w[ch])
    rng = np.random.default_rng(seed)
    gal = {c: np.fft.irfft(spec[c] / DT, n=N, axis=-1) for c in CAT_DTYPES}
    gal["dgb"] = gal["dgb"] + SYN_SIGMA * rng.standard_normal((3, N))
    mbhb = 1e-21 * rng.standard_normal((3, N))
    obs = gal["dgb"] + gal["igb"] + gal["vgb"] + mbhb + 1e-21 * rng.standard_normal((3, N))
    with h5py.File(path, "w") as f:
        f["obs/tdi"] = _tdi(t, obs)
        f["sky/mbhb/tdi"] = _tdi(t, mbhb)
        for comp, dtype in CAT_DTYPES.items():
            rows = [s[1:] for s in SYN_SOURCES if s[0] == comp]
            cat = np.zeros((len(rows), 1), dtype)
            for r, (f0, fdot, amp, phi0, colat, lon, psi, cos_iota) in enumerate(rows):
                cat["Amplitude"][r], cat["Frequency"][r], cat["FrequencyDerivative"][r] = amp, f0, fdot
                cat["InitialPhase"][r], cat["EclipticLatitude"][r] = phi0, np.pi / 2 - colat
                cat["EclipticLongitude"][r], cat["Polarization"][r] = lon, psi
                cat["Inclination"][r] = np.arccos(cos_iota)
            f[f"sky/{comp}/cat"] = cat
            f[f"sky/{comp}/tdi"] = _tdi(t, gal[comp])
    return Path(path)


class PolishOnlySampler:
    """impulse ``PTSampler``'s surface for ``map_fit_binned``: one cold sample, ``x0``."""

    def __init__(self, ndim: int, lnlike: Any, lnprior: Any, **kwargs: Any):
        self.ndim, self.lnlike, self.lnprior, self.kwargs = ndim, lnlike, lnprior, kwargs
        self.x0: np.ndarray | None = None
        self.num_iterations: int | None = None

    def sample(self, x0, num_iterations: int) -> None:
        self.x0, self.num_iterations = np.array(x0, float), int(num_iterations)

    def load_chain(self) -> dict:
        x = self.x0[None, None, :]
        lp = np.array([[self.lnlike(self.x0) + self.lnprior(self.x0)]])
        return {"samples": x, "lnprob": lp, "accepted": np.ones((1, 1))}
