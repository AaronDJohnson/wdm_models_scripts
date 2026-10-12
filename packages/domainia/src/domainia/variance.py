"""Diagonal WDM variances (spec §4.3): the band average (Eq. 36) and the midpoint (Eq. 39).

Each is a pytree (``wdm.backend.pytree_dataclass``): the host-built constants are its
data, the spectrum and the cells its static meta, so a variance is passed to ``jax.jit``
as an argument.  Calling it with the noise parameters gives the ``(P,)`` variances of the
cells (``A = 1`` for the log-linear PSD; one channel at a time for a separable PSD).

Oracles (bitwise on NumPy):

* log-linear band -- WT ``matrix.band_variance`` (wdm_toys/matrix.py:257-270): ``|f|``
  folded at DC, ``2 f_Nyq - f`` reflected above Nyquist, ``S = exp(L_b[m] alpha)``,
  ``0.5 (S phi2).sum / phi2.sum``; ``f_off`` uses the CELLS' grid (``N' dt`` on a padded
  grid) while ``alpha`` uses the PSD's ``T`` (the data record, continued linearly);
* log-linear midpoint -- WT ``matrix.mid_variance`` (L273-277);
* separable band/midpoint -- GB ``band_psd`` (toy_models/galactic_binary/matrix.py:470-483)
  and ``covariance_builder``'s ``approx`` (L642-646): ``0.5 (<s_I>_m + g(t_n) <s_G>_m)``
  with ``g = exp(mu @ e(t_n))``, band means ``s[:, window] @ (phi^2 / sum phi^2)``,
  midpoint at the band-centre bin ``m Nt/2``.
* generic -- ``wdm.noise.evolutionary_band_variance`` / ``0.5 S(f_m, t_n)`` (NumPy only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import wdm
from wdm.backend import Backend, get_backend, pytree_dataclass
from wdm.filter import phi_centered

from .domains import CellSet
from .maps.base import _BUILD, on_backend
from .spectra import GenericPSD, LogLinearPSD, SeparablePSD

__all__ = ["BandAveragedVariance", "MidpointVariance"]


def _cell_arrays(cells: CellSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cells.grid
    m = np.asarray(cells.m, dtype=np.int64)
    n = np.asarray(cells.n, dtype=np.int64)
    return m, n, n * wdm.delta_T(g.Nf, g.dt)                           # t_n = n dT (WT Grid.t)


def _separable_means(psd: SeparablePSD, cells: CellSet, how: str) -> tuple[tuple, np.ndarray]:
    """``(C, P)`` per component: the band (phi^2-weighted) or band-centre mean of each cell's band."""
    g = cells.grid
    if (psd.domain.N, psd.domain.dt) != (g.N, g.dt):
        raise ValueError(f"the spectra are tabulated on an N={psd.domain.N} record; the cells "
                         f"are on an N={g.N} grid")
    m, n, t = _cell_arrays(cells)
    bands = np.unique(m)
    b_of = np.searchsorted(bands, m)
    h = g.Nt // 2
    out = []
    if how == "band":
        phi = phi_centered(g.Nf, g.Nt, g.dt)
        w2 = phi ** 2 / np.sum(phi ** 2)
        for s, _ in psd.components:
            per_band = np.stack([s.at((int(b) - 1) * h + np.arange(g.Nt)) @ w2 for b in bands],
                                axis=1)                                 # (C, nb)
            out.append(per_band[:, b_of])
    else:
        for s, _ in psd.components:
            out.append(s.at(h * bands)[:, b_of])                         # (C, nb) -> (C, P)
    env = psd.envelope
    Bn = None if env is None else np.asarray(env.basis(t), dtype=float)
    return tuple(out), Bn


def _separable_value(means: tuple, Bn: Any, modulated: Any, mu: Any, channel: int,
                     be: Backend) -> Any:
    xp = be.xp
    g = None if Bn is None else xp.exp(mu @ Bn)
    acc = None
    for j, s in enumerate(means):
        term = g * s[channel] if j == modulated else s[channel]
        acc = term if acc is None else acc + term
    return 0.5 * acc


@pytree_dataclass(data=("Lc", "phi2", "t", "means", "Bn"),
                  meta=("psd", "cells", "phi2_sum", "modulated"))
@dataclass(frozen=True, eq=False)
class BandAveragedVariance:
    """Eq. 36: ``0.5 <S(f, t_n)>`` over ``phi~^2`` around each cell's band."""

    psd: Any
    cells: CellSet
    Lc: Any = field(default=_BUILD, kw_only=True)
    phi2: Any = field(default=_BUILD, kw_only=True)
    t: Any = field(default=_BUILD, kw_only=True)
    means: Any = field(default=_BUILD, kw_only=True)
    Bn: Any = field(default=_BUILD, kw_only=True)
    phi2_sum: Any = field(default=_BUILD, kw_only=True)
    modulated: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.Lc is not _BUILD:
            return
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        for k in ("Lc", "phi2", "t", "means", "Bn", "phi2_sum", "modulated"):
            set_(k, None)
        g = self.cells.grid
        m, n, t = _cell_arrays(self.cells)
        if isinstance(self.psd, LogLinearPSD):
            phi2 = phi_centered(g.Nf, g.Nt, g.dt) ** 2
            sup = np.flatnonzero(phi2)
            phi2 = phi2[sup]
            f_off = (np.arange(g.Nt) - g.Nt // 2)[sup] / (g.N * g.dt)
            fb = wdm.band_frequencies(g.Nf, g.dt)[:, None] + f_off[None, :]
            ff = np.abs(fb)
            f_nyq = 0.5 / g.dt
            ff = np.where(ff > f_nyq, 2.0 * f_nyq - ff, ff)
            Lb = self.psd.L(ff)                                          # (Nf+1, |sup|)
            set_("Lc", Lb[m])
            set_("phi2", phi2)
            set_("phi2_sum", float(phi2.sum()))
            set_("t", t)
        elif isinstance(self.psd, SeparablePSD):
            means, Bn = _separable_means(self.psd, self.cells, "band")
            set_("means", means)
            set_("Bn", Bn)
            set_("modulated", next((j for j, (_, e) in enumerate(self.psd.components)
                                    if e is not None), None))
        elif isinstance(self.psd, GenericPSD):
            set_("t", t)
        else:
            raise TypeError(f"no band-averaged variance for {type(self.psd).__name__}")

    def on(self, be: Backend) -> "BandAveragedVariance":
        """The constants moved to ``be`` (pass the result to ``jax.jit`` as an argument)."""
        return on_backend(self, be, ("Lc", "phi2", "t", "means", "Bn"))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        be = be or get_backend(nu, *(a for a in (self.Lc, self.Bn) if a is not None))
        xp = be.xp
        if isinstance(self.psd, LogLinearPSD):
            a = self.psd.alpha(self.t, nu, be)                           # (P,)
            S = xp.exp(self.Lc * a[:, None])                             # (P, |sup|)
            return 0.5 * (S * self.phi2[None, :]).sum(axis=1) / self.phi2_sum
        if isinstance(self.psd, SeparablePSD):
            return _separable_value(self.means, self.Bn, self.modulated, nu, channel, be)
        g = self.cells.grid
        m, n, _ = _cell_arrays(self.cells)
        full = wdm.noise.evolutionary_band_variance(g.Nf, g.Nt, g.dt, self.psd.bind(nu))
        return full[m, n]


@pytree_dataclass(data=("Lm", "t", "means", "Bn"), meta=("psd", "cells", "modulated"))
@dataclass(frozen=True, eq=False)
class MidpointVariance:
    """Eq. 39: ``0.5 S(f_m, t_n)``; the GB takes ``f_m`` at the band-centre bin ``m Nt/2``."""

    psd: Any
    cells: CellSet
    Lm: Any = field(default=_BUILD, kw_only=True)
    t: Any = field(default=_BUILD, kw_only=True)
    means: Any = field(default=_BUILD, kw_only=True)
    Bn: Any = field(default=_BUILD, kw_only=True)
    modulated: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.Lm is not _BUILD:
            return
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        for k in ("Lm", "t", "means", "Bn", "modulated"):
            set_(k, None)
        g = self.cells.grid
        m, n, t = _cell_arrays(self.cells)
        if isinstance(self.psd, LogLinearPSD):
            set_("Lm", self.psd.L(wdm.band_frequencies(g.Nf, g.dt))[m])
            set_("t", t)
        elif isinstance(self.psd, SeparablePSD):
            means, Bn = _separable_means(self.psd, self.cells, "mid")
            set_("means", means)
            set_("Bn", Bn)
            set_("modulated", next((j for j, (_, e) in enumerate(self.psd.components)
                                    if e is not None), None))
        elif isinstance(self.psd, GenericPSD):
            set_("t", t)
        else:
            raise TypeError(f"no midpoint variance for {type(self.psd).__name__}")

    def on(self, be: Backend) -> "MidpointVariance":
        """The constants moved to ``be`` (pass the result to ``jax.jit`` as an argument)."""
        return on_backend(self, be, ("Lm", "t", "means", "Bn"))

    def __call__(self, nu: Any, *, channel: int = 0, be: Backend | None = None) -> Any:
        be = be or get_backend(nu, *(a for a in (self.Lm, self.Bn) if a is not None))
        if isinstance(self.psd, LogLinearPSD):
            return 0.5 * be.xp.exp(self.psd.alpha(self.t, nu, be) * self.Lm)
        if isinstance(self.psd, SeparablePSD):
            return _separable_value(self.means, self.Bn, self.modulated, nu, channel, be)
        g = self.cells.grid
        m, _, t = _cell_arrays(self.cells)
        f = wdm.band_frequencies(g.Nf, g.dt)[m]
        return 0.5 * np.asarray(self.psd(f, t, nu))
