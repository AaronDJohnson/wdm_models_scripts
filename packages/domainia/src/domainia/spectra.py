"""Noise spectra (spec §4.3).  ``A`` (the sinusoid amplitude) is held outside every one.

* :class:`LogLinearPSD` -- ``S/A = (max(f, f_floor)/f_ref)^{alpha(t)}``,
  ``alpha(t) = alpha_c + alpha_d (t/T - 1/2)`` (WT ``noise.log_f_clip`` + ``matrix._alpha``).
  ``T`` is the DATA record's length: on a padded grid ``alpha`` continues linearly.
* :class:`SeparablePSD` -- ``S_c(f, t) = sum_j g_j(t) s_{j,c}(f)`` on tabulated bins
  (the GB ``s_I + g(t) s_G``, additive; at most one modulated component).
* :class:`TabulatedSpectrum` -- ``(C, n)`` values on absolute rfft bins.
* :class:`Envelope` -- the protocol of ``g(t) = exp(mu . e(t))`` (S5 builds the GB one).
* :class:`GenericPSD` -- any elementwise ``Sn(f, t, nu)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable

import numpy as np
from wdm.backend import Backend, get_backend

from .domains import FourierDomain

__all__ = ["Envelope", "GenericPSD", "LogLinearPSD", "SeparablePSD", "TabulatedSpectrum"]


@dataclass(frozen=True)
class LogLinearPSD:
    """The sinusoid's drifting power law, unit amplitude: ``exp(alpha(t, nu) L(f))``."""

    f_floor: float
    f_ref: float
    T: float

    capabilities: ClassVar[frozenset[str]] = frozenset({"gram", "bessel", "operator"})
    nu_names: ClassVar[tuple[str, str]] = ("alpha_c", "alpha_d")

    def L(self, f: Any) -> np.ndarray:
        r""":math:`L(f) = \log(\max(f, f_{floor}) / f_{ref})` (host; WT noise.py:105-107)."""
        return np.log(np.maximum(np.asarray(f, dtype=float), self.f_floor) / self.f_ref)

    def alpha(self, t: Any, nu: Any, be: Backend | None = None) -> Any:
        """``alpha_c + alpha_d (t/T - 1/2)`` (WT matrix.py:252-254); ``nu = (alpha_c, alpha_d)``."""
        be = be or get_backend(t, *(nu if isinstance(nu, (tuple, list)) else (nu,)))
        alpha_c, alpha_d = nu[0], nu[1]
        return alpha_c + alpha_d * (be.asarray(t, dtype=float) / self.T - 0.5)

    def __call__(self, f: Any, t: Any, nu: Any, be: Backend | None = None) -> Any:
        """``S / A`` at broadcast ``(f, t)``, elementwise (the ``wdm.evolutionary`` convention)."""
        be = be or get_backend(t, *(nu if isinstance(nu, (tuple, list)) else (nu,)))
        return be.xp.exp(self.alpha(t, nu, be) * self.L(f))

    def bind(self, nu: Any) -> Callable[[Any, Any], Any]:
        """``Sn(f, t)`` at fixed ``nu`` (for ``wdm.noise`` / ``wdm.evolutionary``)."""
        return lambda f, t: self(f, t, nu)


@dataclass(frozen=True, eq=False)
class TabulatedSpectrum:
    """``values[c, j]`` at absolute rfft bin ``domain.k0 + j`` of ``domain``'s record."""

    domain: FourierDomain
    values: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if v.ndim != 2 or v.shape[1] != self.domain.n:
            raise ValueError(f"values must be (channels, {self.domain.n}); got {v.shape}")
        object.__setattr__(self, "values", v)

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    def _check(self, N: int, dt: float, lo: int, hi: int) -> None:
        d = self.domain
        if (N, dt) != (d.N, d.dt):
            raise ValueError(f"bins of an N={N}, dt={dt} record; this spectrum is on N={d.N}, dt={d.dt}")
        if lo < d.k0 or hi > d.k0 + d.n:
            raise ValueError(f"bins [{lo}, {hi}) are outside the tabulated [{d.k0}, {d.k0 + d.n})")

    def on(self, window: FourierDomain) -> np.ndarray:
        """``(C, window.n)``: the values on ``window``'s bins; raises outside the table."""
        self._check(window.N, window.dt, window.k0, window.k0 + window.n)
        lo = window.k0 - self.domain.k0
        return self.values[:, lo:lo + window.n]

    def at(self, bins: Any) -> np.ndarray:
        """``(C, *bins.shape)``: the values at absolute bins; raises outside the table."""
        bins = np.asarray(bins, dtype=np.int64)
        self._check(self.domain.N, self.domain.dt, int(bins.min()), int(bins.max()) + 1)
        return self.values[:, bins - self.domain.k0]


@runtime_checkable
class Envelope(Protocol):
    """``g(t) = exp(mu . e(t))`` with an orthonormal basis ``e`` (``n_params`` functions).

    ``basis`` is host NumPy (a constant); ``log_g``/``ghat`` take a traced ``mu``.
    """

    n_params: int

    def basis(self, t: Any) -> np.ndarray: ...                            # (n_params, len(t))

    def log_g(self, t: Any, mu: Any, be: Backend | None = None) -> Any: ...   # mu @ basis(t)

    def ghat(self, lags: Any, mu: Any, be: Backend | None = None) -> Any: ...  # (len(lags),) complex


@dataclass(frozen=True, eq=False)
class SeparablePSD:
    """``S_c(f, t) = sum_j g_j(t) s_{j,c}(f)``; ``g_j = 1`` where the envelope is ``None``."""

    components: tuple[tuple[TabulatedSpectrum, Envelope | None], ...]
    channels: tuple[str, ...] = ()

    capabilities: ClassVar[frozenset[str]] = frozenset({"separable", "fourier_toeplitz"})

    def __post_init__(self) -> None:
        comps = tuple((s, e) for s, e in self.components)
        if not comps:
            raise ValueError("a separable PSD needs at least one component")
        dom = comps[0][0].domain
        C = comps[0][0].channels
        for s, _ in comps:
            if s.domain != dom or s.channels != C:
                raise ValueError("every component must share the domain and the channel count")
        if sum(e is not None for _, e in comps) > 1:
            raise ValueError("at most one modulated component (the GB s_I + g s_G form)")
        if self.channels and len(self.channels) != C:
            raise ValueError(f"{len(self.channels)} channel names for {C} channels")
        object.__setattr__(self, "components", comps)

    @property
    def domain(self) -> FourierDomain:
        return self.components[0][0].domain

    @property
    def envelope(self) -> Envelope | None:
        return next((e for _, e in self.components if e is not None), None)


@dataclass(frozen=True, eq=False)
class GenericPSD:
    """Any elementwise, broadcasting ``Sn(f, t, nu)`` (Eq. 45 ``interp``/``direct``, dense
    exact covariances, tests)."""

    fn: Callable[[Any, Any, Any], Any]

    capabilities: ClassVar[frozenset[str]] = frozenset({"interp", "direct"})

    def __call__(self, f: Any, t: Any, nu: Any, be: Backend | None = None) -> Any:
        return self.fn(f, t, nu)

    def bind(self, nu: Any) -> Callable[[Any, Any], Any]:
        return lambda f, t: self.fn(f, t, nu)
