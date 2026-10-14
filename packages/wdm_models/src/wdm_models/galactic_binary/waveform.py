r"""The galactic-binary template in prepared units (spec §5.2 waveform.py): binary_galactica's
``template_on_window`` on a fixed window of the case's rfft grid, divided by ``DT sigma0``.

* **Backends** (user ruling 2026-10-11).  ``backend="onfly"`` (TDI on the fly, as WDM II
  states) is the release default.  ``backend="fourier"`` is the legacy fast-slow reduced DFT
  that the paper's data and chains used through gbonfly: binary_galactica >= 0.5.0 reproduces
  gbonfly to ~2e-16 relative, so the golden harness passes it **explicitly**.  At the toys'
  1 mHz the two agree to 1 - FF <= 3.1e-6 on the store window; ``onfly`` is audited (reported),
  never gated.
* **Units**: ``template_on_window(p8, T, bin_lo, n_bins, BW, positions=orbit) / (DT sigma0)``
  -- no factor 2 (binary_galactica 0.5.0 restored the physical amplitude: its native GLASS
  vector now equals gbonfly's).
* **Orbit**: ``orbit_cache(T, 0, BW)``, built once, eagerly, under
  ``jax.ensure_compile_time_eval()`` so it is never a leaked tracer.
* **Pytree** (spec §3.3): the orbit and the ``DT sigma0`` scale are leaves (``jit``
  arguments, never closed over); ``T``, ``bin_lo``, ``n_bins``, ``BW`` and ``backend`` are
  static (registered with JAX on the first construction, which has imported JAX already).  Called with a NumPy ``p8`` (the GB "NumPy" likelihoods) the template is computed on
  CPU-JAX and returned as a NumPy array; called under ``jit`` it stays a JAX value.

**Two-tier check** (spec §5.2).  :class:`GBTemplate` refuses a binary_galactica older
than 0.5.0, then (``check=True``) evaluates the injection on the analysis window and on the
store and compares ``sum |h|^2`` and ``Re <z, h>`` with ``config.TEMPLATE_SUMS`` at relative
``1e-6`` (``fourier``) or ``1e-3`` (``onfly``: measured 2.8e-4 at worst).  A power ratio near
1/4 is the half-amplitude native convention of binary_galactica < 0.5.0 and raises with that
diagnosis; any other miss raises with the deltas.  The fine tier (sums at 1e-10 against
``golden/galactic_binary/products.json``) is the harness's (``tests/golden``).

binary_galactica and JAX are imported when a template is built, never at module import.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .config import DT, TEMPLATE_BW, TEMPLATE_SUMS

__all__ = ["BACKENDS", "BG_MIN_VERSION", "GBTemplate", "HALF_AMPLITUDE_MESSAGE", "GUARD_RTOL",
           "TemplateCheckError", "check_binary_galactica", "template_sums"]

BACKENDS = ("onfly", "fourier")
BG_MIN_VERSION = "0.5.0"
#: Construction guard tolerances on the template sums (relative).
GUARD_RTOL = {"fourier": 1e-6, "onfly": 1e-3}
HALF_AMPLITUDE_MESSAGE = ("binary_galactica < 0.5.0 uses the half-amplitude native convention; "
                          "install >= 0.5.0")
#: Power ratios sum|h|^2 / reference inside this band are diagnosed as the half amplitude.
_QUARTER = (0.2, 0.3)


class TemplateCheckError(ValueError):
    """The construction guard failed: wrong binary_galactica normalisation or a broken template."""


def _release(version: str) -> tuple[int, int, int]:
    """``(major, minor, micro)`` of a version string (numeric comparison: 0.10.0 > 0.5.0)."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(version))
    if m is None:
        raise ImportError(f"binary_galactica reports an unparsable version {version!r}")
    return tuple(int(g) for g in m.groups())


def check_binary_galactica():
    """``binary_galactica`` (imported) when its version is >= :data:`BG_MIN_VERSION`;
    ``ImportError`` with :data:`HALF_AMPLITUDE_MESSAGE` otherwise."""
    import binary_galactica as bg

    if _release(bg.__version__) < _release(BG_MIN_VERSION):
        raise ImportError(f"{HALF_AMPLITUDE_MESSAGE} (found {bg.__version__} at {bg.__file__})")
    return bg


def _template_on_window(p8: Any, **kw: Any) -> Any:
    """``binary_galactica.template_on_window`` (one indirection: tests substitute it)."""
    import binary_galactica as bg

    return bg.template_on_window(p8, **kw)


_PYTREE: list[bool] = []


def _register_pytree() -> None:
    """Register :class:`GBTemplate` with JAX once (on the first construction, which has already
    imported JAX): leaves ``orbit``, ``scale``; static ``T, bin_lo, n_bins, BW, backend``."""
    if not _PYTREE:
        import jax

        jax.tree_util.register_pytree_node(
            GBTemplate, lambda t: ((t.orbit, t.scale), (t.T, t.bin_lo, t.n_bins, t.BW, t.backend)),
            lambda aux, leaves: GBTemplate._make(*leaves, *aux))
        _PYTREE.append(True)


class GBTemplate:
    """``p8 -> (2, n_bins)`` complex A/E template in prepared units on absolute bins
    ``[bin_lo, bin_lo + n_bins)`` of ``prep``'s record (a :class:`prepared.PreparedCase`).

    ``window``: ``"store"`` (the matrix models' data), ``"window"`` (the analysis window),
    ``"nominal"`` (the Fourier models' bins) or explicit ``(bin_lo, n_bins)``.  Immutable; a JAX
    pytree whose leaves are ``orbit`` and ``scale`` (``DT sigma0[:, None]``).
    """

    __slots__ = ("orbit", "scale", "T", "bin_lo", "n_bins", "BW", "backend")

    def __init__(self, prep, *, BW: int = TEMPLATE_BW, backend: str = "onfly",
                 window: str | tuple[int, int] = "store", check: bool = True) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}")
        BW = int(BW)
        if BW <= 0 or BW % 2:
            raise ValueError(f"BW must be a positive even integer; got {BW}")
        bg = check_binary_galactica()
        from wdm.backend import jax_backend

        import jax

        jax_backend()                                  # require_x64()
        _register_pytree()
        lo, n = _window(prep, window)
        T = float(prep.T)
        with jax.ensure_compile_time_eval():
            orbit = bg.orbit_cache(T, 0.0, BW)
        self._fill(orbit, DT * np.asarray(prep.sigma0, float)[:, None], T, lo, n, BW, backend)
        if check:
            self.check(prep)

    def _fill(self, orbit, scale, T, bin_lo, n_bins, BW, backend) -> None:
        for name, value in zip(self.__slots__, (orbit, scale, T, bin_lo, n_bins, BW, backend)):
            object.__setattr__(self, name, value)

    @classmethod
    def _make(cls, orbit, scale, T, bin_lo, n_bins, BW, backend) -> "GBTemplate":
        out = object.__new__(cls)
        out._fill(orbit, scale, T, bin_lo, n_bins, BW, backend)
        return out

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("GBTemplate is immutable")

    def __repr__(self) -> str:
        return (f"GBTemplate(T={self.T}, bins=[{self.bin_lo}, {self.bin_lo + self.n_bins}), "
                f"BW={self.BW}, backend={self.backend!r})")

    def __call__(self, p8: Any) -> Any:
        h = _template_on_window(p8, T=self.T, bin_lo=self.bin_lo, n_bins=self.n_bins, BW=self.BW,
                                positions=self.orbit, backend=self.backend) / self.scale
        return np.asarray(h) if isinstance(p8, np.ndarray) else h

    def on(self, window: str | tuple[int, int], prep) -> "GBTemplate":
        """The same template (orbit, scale, backend) on another window of ``prep``."""
        lo, n = _window(prep, window)
        return GBTemplate._make(self.orbit, self.scale, self.T, lo, n, self.BW, self.backend)

    def check(self, prep) -> dict[str, dict[str, float]]:
        """The construction guard at the injection (module docstring); returns the relative
        deltas ``{window: {"sum_h2", "re_zh"}}`` when it passes."""
        ref = TEMPLATE_SUMS[prep.case]
        rtol = GUARD_RTOL[self.backend]
        out: dict[str, dict[str, float]] = {}
        for name, z in (("window", prep.z), ("store", prep.store_z)):
            s2, zh = template_sums(self.on(name, prep), prep.inject, z)
            r2, rz = ref[name]
            ratio = s2 / r2
            if _QUARTER[0] < ratio < _QUARTER[1]:
                raise TemplateCheckError(f"{HALF_AMPLITUDE_MESSAGE}: sum|h|^2 on the {name} is "
                                         f"{ratio:.4f} of the reference")
            d2, dz = abs(ratio - 1.0), abs(zh / rz - 1.0)
            if not (d2 <= rtol and dz <= rtol):
                raise TemplateCheckError(
                    f"{prep.case} {self.backend} template on the {name}: sum|h|^2 {s2!r} (rel "
                    f"{d2:.3e}), Re<z,h> {zh!r} (rel {dz:.3e}); gate {rtol:g} against {ref[name]}")
            out[name] = {"sum_h2": d2, "re_zh": dz}
        return out


def template_sums(template: GBTemplate, p8, z) -> tuple[float, float]:
    """``(sum |h|^2, Re <z, h>)`` of ``template(p8)`` against the data ``z`` on its window."""
    h = np.asarray(template(np.asarray(p8, float)))
    return float(np.sum(np.abs(h) ** 2)), float(np.real(np.vdot(np.asarray(z), h)))


def _window(prep, window) -> tuple[int, int]:
    if isinstance(window, tuple):
        lo, n = (int(v) for v in window)
    elif window == "store":
        lo, n = prep.store_offset, prep.store_n
    elif window == "window":
        lo, n = prep.bin_offset, prep.n_bins
    elif window == "nominal":
        lo, n = prep.bin_offset + prep.freq.start, prep.freq.stop - prep.freq.start
    else:
        raise ValueError(f"window must be 'store', 'window', 'nominal' or (bin_lo, n_bins); got {window!r}")
    if n <= 0:
        raise ValueError(f"empty window {(lo, n)}")
    return lo, n
