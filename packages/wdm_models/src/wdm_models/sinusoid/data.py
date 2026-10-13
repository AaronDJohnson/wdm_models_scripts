r"""The shipped sinusoid data (spec §5.1 data.py).

One file per case, ``data/sinusoid/<case>/toy_data.npz`` (WT ``data.save`` layout: ``d``, ``h``,
``n``, ``w_d``, ``snr``, ``seed``, ``noise_source``, ``_cfg_keys``/``_cfg_vals``).  The three
cases share one noise draw (``seed = 1``, WT ``reuse_noise``) and differ in the signal.

:func:`load` is fail-closed: the file's sha256 must be the one pinned here (and in
``data/MANIFEST.json``), the stored configuration must equal :mod:`.config`'s case exactly, the
stored ``h`` must be the release signal bit for bit, ``n`` the shared draw (sha256), ``d = h + n``
bit for bit, and ``wdm.forward_time(d, 64, 256, dt)`` must equal the stored ``w_d``
(``np.array_equal``; spec §2.4: the omega-normalised window is the form that makes it bitwise).

:func:`regenerate` is WT ``data.generate``: ``n = G z`` with the voices in WT order (cosine
``k = 0..N/2``, then sine ``k = 1..N/2-1``; ``domainia ExactTimeCovariance.voice_factor``) and
``z = default_rng(seed).standard_normal(N)``, plus the signal.  It is **not bitwise** at the
manuscript size (the shipped ``n`` is reproduced to ~1e-13 relative) and it is heavy (the dense
factor is 16384 x 16384, 2 GB): run it through ``scripts/sinusoid/regenerate_data.py``, never in
the default tests.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import wdm
from domainia.covariance import ExactTimeCovariance
from domainia.data import TimeSeries
from domainia.domains import TimeDomain

from ..common.validation import resolve_data_root, shipped
from .config import SinusoidConfig, config
from .psd import as_loglinear
from .signal import Sinusoid

__all__ = ["DATA_SHA256", "NOISE_SEED", "NOISE_SHA256", "SinusoidData", "check_payload", "load",
           "regenerate"]

DATA_SHA256 = {
    "manuscript_low": "650cf032a6e30fc1ad3bd40248ea943e21a70f8243a27ff1ef40703f86d530e8",
    "manuscript_mid": "02cdb2595ce1058a063b131abbd1588070892bb0f18260f4cf545936785d222d",
    "manuscript_high": "a6db39e500be548d2d6453d04f1f01d3b891d815ba8c90ce2de2958d972bc05a",
}
#: sha256 of the shared noise draw's float64 bytes (every case).
NOISE_SHA256 = "1fc614025fe15440839e8e75512d6b6d42273c4bb7c16067177e5440ce55da23"
NOISE_SEED = 1
_FIELDS = ("Nf", "Nt", "dt", "A_h", "f_0", "phi_0", "A_n", "alpha_c", "alpha_d", "f_ref", "f_min",
           "N", "T_obs", "fs", "f_nyq", "DF", "DT")
#: WT ToyConfig fields the release does not carry, with the only values it accepts.
_FIXED = {"fdot": 0.0, "fit_fdot": 0.0, "frequency_study": 1.0}


@dataclass(frozen=True, eq=False)
class SinusoidData:
    """One case's shipped data (read-only arrays)."""

    case: str
    cfg: SinusoidConfig
    d: np.ndarray
    h: np.ndarray
    n: np.ndarray
    w_d: np.ndarray
    seed: int
    snr: float
    path: Path
    sha256: str

    def series(self) -> TimeSeries:
        """``d`` as a ``domainia`` ``TimeSeries`` on ``TimeDomain(N, dt)``."""
        return TimeSeries(self.d, TimeDomain(self.cfg.N, self.cfg.dt))

    def truth(self, model: str) -> np.ndarray:
        """The injected parameters in ``model``'s layout (``cfg.truth``; the sanitising anchor)."""
        return self.cfg.truth(model)


def _readonly(a: Any) -> np.ndarray:
    out = np.array(a, dtype=np.float64)
    out.setflags(write=False)
    return out


def check_payload(case: str, arrays: dict[str, np.ndarray]) -> None:
    """Raise ``ValueError`` unless ``arrays`` is ``case``'s data as the release defines it."""
    cfg = config(case)
    missing = {"d", "h", "n", "w_d", "seed", "_cfg_keys", "_cfg_vals"} - set(arrays)
    if missing:
        raise ValueError(f"{case}: toy_data.npz lacks {sorted(missing)}")
    stored = {str(k): float(v) for k, v in zip(arrays["_cfg_keys"], arrays["_cfg_vals"], strict=True)}
    want = {**{k: float(getattr(cfg, k)) for k in _FIELDS}, **_FIXED}
    bad = {k: (stored.get(k), v) for k, v in want.items() if stored.get(k) != v}
    if bad:
        raise ValueError(f"{case}: the stored configuration differs from the release's "
                         f"(field: (stored, release)): {bad}")
    d, h, n = (np.asarray(arrays[k]) for k in ("d", "h", "n"))
    for name, v in (("d", d), ("h", h), ("n", n)):
        if v.dtype != np.float64 or v.shape != (cfg.N,) or not np.all(np.isfinite(v)):
            raise ValueError(f"{case}: {name} must be finite float64 of shape ({cfg.N},)")
    if not np.array_equal(Sinusoid.for_config(cfg)(cfg.truth("band_raw")[:3]), h):
        raise ValueError(f"{case}: the stored h is not the release signal bit for bit")
    if hashlib.sha256(np.ascontiguousarray(n).tobytes()).hexdigest() != NOISE_SHA256:
        raise ValueError(f"{case}: the stored noise is not the shared manuscript draw")
    if not np.array_equal(h + n, d):
        raise ValueError(f"{case}: d is not h + n bit for bit")
    if not np.array_equal(wdm.forward_time(d, cfg.Nf, cfg.Nt, cfg.dt), arrays["w_d"]):
        raise ValueError(f"{case}: forward_time(d) differs from the stored w_d (spec §2.4)")


def load(case: str, *, data_root: str | os.PathLike | None = None) -> SinusoidData:
    """``case``'s shipped data, checked as the module docstring says."""
    cfg = config(case)
    rel = f"sinusoid/{case}/toy_data.npz"
    art = shipped(resolve_data_root(data_root), rel, DATA_SHA256[case])
    arrays = art.load_npz()
    check_payload(case, arrays)
    return SinusoidData(case, cfg, _readonly(arrays["d"]), _readonly(arrays["h"]), _readonly(arrays["n"]),
                        _readonly(arrays["w_d"]), int(arrays["seed"]), float(arrays["snr"]), art.path,
                        DATA_SHA256[case])


def regenerate(case: str | SinusoidConfig, *, seed: int = NOISE_SEED) -> dict[str, Any]:
    """``{"d", "h", "n", "seed"}`` drawn as WT ``data.generate`` (module docstring; heavy and
    not bitwise at the manuscript size)."""
    cfg = config(case) if isinstance(case, str) else case
    exact = ExactTimeCovariance(as_loglinear(cfg), cfg.N, cfg.dt, form="dense")
    n = exact.draw((cfg.alpha_c, cfg.alpha_d), np.random.default_rng(seed), cfg.A_n)
    h = Sinusoid.for_config(cfg)(cfg.truth("band_raw")[:3])
    return {"d": h + n, "h": h, "n": n, "seed": int(seed)}
