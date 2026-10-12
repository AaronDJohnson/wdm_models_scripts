"""Maps out of the time domain: WDM cells, retained Fourier bins, crops and zero padding.

Every evaluator here is the WT oracle's library call (``wdm_paper_2/toy_models/src/
wdm_toys/matrix.py``: ``Grid.forward``/``forward_many``/``inverse_many``/``atom``,
``Variant.extend``, ``MatrixWhittle``; ``band.py``: ``FourierBand``), so the NumPy path is
bitwise: ``TimeToWDM.apply(d) == forward_time(d)[mask][keep]``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import wdm
from wdm.backend import pytree_dataclass
from wdm.transform import atom_spectrum, base_atom_spectra

from ..domains import FourierDomain, TimeDomain, WDMDomain
from .base import _BUILD, LinearMap

__all__ = ["CropTime", "TimeToFourier", "TimeToWDM", "WDMToTime", "ZeroPadTime"]


def _check_grid(time: TimeDomain, dom: WDMDomain) -> wdm.Grid:
    g = dom.cells.grid
    if time.N != g.N or time.dt != g.dt:
        raise ValueError(f"time domain (N={time.N}, dt={time.dt}) does not match the cells' grid "
                         f"(N={g.N}, dt={g.dt})")
    return g


def _last_axis(x: Any, n: int, what: str) -> None:
    """Refuse a vector whose last axis is not ``n`` long (shapes are static under ``jit``)."""
    if np.shape(x)[-1:] != (n,):
        raise ValueError(f"{what}: last axis must have length {n}; got shape {np.shape(x)}")


def _flat(x: Any, N: int) -> tuple[Any, tuple[int, ...]]:
    if x.shape[-1] != N:
        raise ValueError(f"last axis must have length {N}; got {x.shape}")
    return x.reshape(-1, N), x.shape[:-1]


@pytree_dataclass(data=("m_idx", "n_idx"), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class TimeToWDM(LinearMap):
    """``x -> (W x)[cells]``: the WDM transform of a record, cells in ``dst``'s order.

    ``apply`` is ``wdm.forward_time`` (1-D) or ``wdm.forward_time_many`` (batched; bitwise
    the same per row), then a gather -- WT ``Grid.forward``/``forward_many`` followed by
    ``Variant.map``'s ``[keep]`` (matrix.py:191-197, 711-716).  ``adjoint`` is
    ``W^T`` restricted to the cells, ``dt * W^{-1}`` of the zero-filled grid
    (``W W^T = dt I``), valid on any cell subset.  ``rows`` are the time-domain atoms
    ``irfft(atom_spectrum)`` (WT ``Grid.atom``, matrix.py:225-232).
    """

    src: TimeDomain
    dst: WDMDomain
    m_idx: Any = field(default=_BUILD, kw_only=True)
    n_idx: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.m_idx is _BUILD:
            _check_grid(self.src, self.dst)
            object.__setattr__(self, "m_idx", np.asarray(self.dst.cells.m, dtype=np.int64))
            object.__setattr__(self, "n_idx", np.asarray(self.dst.cells.n, dtype=np.int64))

    @property
    def grid(self) -> wdm.Grid:
        return self.dst.cells.grid

    def apply(self, x, be=None):
        be = self._be(x, be)
        x = be.asarray(x)
        g = self.grid
        if x.ndim == 1:
            w = wdm.forward_time(x, g.Nf, g.Nt, g.dt, be=be)
            return w[self.m_idx, self.n_idx]
        X, lead = _flat(x, g.N)
        w = wdm.forward_time_many(X, g.Nf, g.Nt, g.dt, be=be)
        return w[:, self.m_idx, self.n_idx].reshape(*lead, self.m_idx.size)

    def _scatter(self, y, be):
        g = self.grid
        Y = y.reshape(-1, y.shape[-1])
        full = be.xp.zeros((Y.shape[0], g.Nf + 1, g.Nt), dtype=Y.dtype)
        return be.set(full, (slice(None), self.m_idx, self.n_idx), Y, copy=False)

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        g = self.grid
        x = g.dt * wdm.inverse_time_many(self._scatter(y, be), g.Nf, g.Nt, g.dt, be=be)
        return x.reshape(*y.shape[:-1], g.N)

    def rows(self, idx):
        g = self.grid
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        base = base_atom_spectra(g.Nf, g.Nt, g.dt)
        m, n = np.asarray(self.m_idx), np.asarray(self.n_idx)
        return np.stack([np.fft.irfft(atom_spectrum(int(m[p]), int(n[p]), g.Nf, g.Nt, g.dt,
                                                    base=base), n=g.N) for p in idx])


@pytree_dataclass(data=("m_idx", "n_idx"), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class WDMToTime(LinearMap):
    """``W^{-1} = W^T / dt`` on a complete cell set: ``wdm.inverse_time_many`` of the
    unpacked grid (WT ``Grid.inverse_many``, matrix.py:199-204).  ``adjoint`` is
    ``W^{-T} = W / dt``."""

    src: WDMDomain
    dst: TimeDomain
    m_idx: Any = field(default=_BUILD, kw_only=True)
    n_idx: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.m_idx is _BUILD:
            g = _check_grid(self.dst, self.src)
            m, n = wdm.grid.packed_cells(g.Nf, g.Nt)
            got = np.asarray(self.src.cells.m), np.asarray(self.src.cells.n)
            if len(got[0]) != m.size or set(zip(*got)) != set(zip(m.tolist(), n.tolist())):
                raise ValueError("WDMToTime needs every active cell of the grid (W^-1 = W^T/dt "
                                 "holds on complete cell sets only); use TimeToWDM.adjoint")
            object.__setattr__(self, "m_idx", got[0].astype(np.int64))
            object.__setattr__(self, "n_idx", got[1].astype(np.int64))

    @property
    def grid(self) -> wdm.Grid:
        return self.src.cells.grid

    def apply(self, y, be=None):
        be = self._be(y, be)
        y = be.asarray(y)
        g = self.grid
        Y = y.reshape(-1, y.shape[-1])
        full = be.xp.zeros((Y.shape[0], g.Nf + 1, g.Nt), dtype=Y.dtype)
        full = be.set(full, (slice(None), self.m_idx, self.n_idx), Y, copy=False)
        return wdm.inverse_time_many(full, g.Nf, g.Nt, g.dt, be=be).reshape(*y.shape[:-1], g.N)

    def adjoint(self, x, be=None):
        be = self._be(x, be)
        fwd = TimeToWDM(self.dst, self.src, m_idx=self.m_idx, n_idx=self.n_idx)
        return fwd.apply(x, be) / self.grid.dt


def _keep_bins(N: int, dt: float, fmin: float) -> tuple[int, int]:
    """``fmin <= f < f_Nyq``, DC and Nyquist dropped (WT ``FourierBand``, band.py:17-26, and
    ``MatrixWhittle``, matrix.py:883-884): the contiguous kept bins ``(k0, n)``."""
    f = np.fft.rfftfreq(N, d=dt)
    keep = (f >= max(float(fmin), 0.0)) & (f < 0.5 / dt)
    keep[[0, -1]] = False
    k = np.flatnonzero(keep)
    if k.size == 0:
        raise ValueError(f"fmin={fmin} leaves no interior Fourier frequencies")
    if k[-1] - k[0] + 1 != k.size:
        raise AssertionError("the kept bins are not contiguous")
    return int(k[0]), int(k.size)


@pytree_dataclass(data=(), meta=("src", "dst", "fmin"))
@dataclass(frozen=True, eq=False)
class TimeToFourier(LinearMap):
    """``x -> rfft(x)[keep]``: the retained complex bins, ``fmin <= f < f_Nyq`` with DC and
    Nyquist dropped (kind ``"analysis"``).

    ``apply`` returns the complex bins exactly as the oracle computes them
    (``np.fft.rfft(r)[keep]``, WT likelihood.py:245 and matrix.py:894), because the
    Whittle density's ``2/N |R|^2`` order is bitwise only on them.  Their real orthonormal
    coordinates are ``sqrt(2/N) (Re R_k, -Im R_k)``, bin by bin -- the
    ``FourierBand.discarded_spectra`` convention (band.py:30-41) -- with rows
    :meth:`orthonormal_rows`.  ``dst`` has ``units="rfft"``.
    """

    src: TimeDomain
    fmin: float = 0.0
    dst: Any = field(default=_BUILD, kw_only=True)

    kind = "analysis"

    def __post_init__(self) -> None:
        if self.dst is _BUILD:
            k0, n = _keep_bins(self.src.N, self.src.dt, self.fmin)
            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n, "rfft"))

    def apply(self, x, be=None):
        _last_axis(x, self.src.N, "TimeToFourier.apply")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        return be.fft.rfft(be.asarray(x), axis=-1)[..., k0:k0 + n]

    def apply_complex(self, x, be=None):
        """``F x`` for a complex series (``fft``, the kept bins)."""
        _last_axis(x, self.src.N, "TimeToFourier.apply_complex")
        be = self._be(x, be)
        k0, n = self.dst.k0, self.dst.n
        return be.fft.fft(be.asarray(x), axis=-1)[..., k0:k0 + n]

    def adjoint(self, y, be=None):
        """``Re(F^H y)``: the real-inner-product adjoint, ``(N/2) irfft`` of the kept bins."""
        _last_axis(y, self.dst.n, "TimeToFourier.adjoint")
        be = self._be(y, be)
        y = be.asarray(y)
        N, k0, n = self.src.N, self.dst.k0, self.dst.n
        spec = be.xp.zeros((*y.shape[:-1], N // 2 + 1), dtype=complex)
        spec = be.set(spec, (..., slice(k0, k0 + n)), y, copy=False)
        return (N / 2.0) * be.fft.irfft(spec, n=N, axis=-1)

    def rows(self, idx):
        """``exp(-2 pi i k j / N)`` with ``k j`` reduced mod ``N`` in exact int64 arithmetic
        first: the phase stays in ``[0, 2 pi)``, so the rows are good to ~1e-16 at any ``N``
        (unreduced, ``k j`` reaches 1.3e8 at ``N = 16384`` and the phase error 6e-12)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        k = self.dst.k0 + idx
        j = np.arange(self.src.N, dtype=np.int64)
        return np.exp(-2j * np.pi * ((np.outer(k, j) % self.src.N) / self.src.N))

    def orthonormal_rows(self) -> np.ndarray:
        """``(2 n, N)`` real orthonormal rows, bin by bin ``sqrt(2/N) (cos, sin)``: the
        coordinates ``sqrt(2/N) (Re R_k, -Im R_k)`` of :meth:`apply`'s bins."""
        N, k0, n = self.src.N, self.dst.k0, self.dst.n
        spectra = np.zeros((2 * n, N // 2 + 1), dtype=complex)
        rows = np.arange(n)
        spectra[2 * rows, k0 + rows] = np.sqrt(N / 2)
        spectra[2 * rows + 1, k0 + rows] = -1j * np.sqrt(N / 2)
        return np.fft.irfft(spectra, n=N, axis=-1)


@pytree_dataclass(data=(), meta=("src", "start", "n"))
@dataclass(frozen=True, eq=False)
class CropTime(LinearMap):
    """Samples ``[start, start + n)`` (whittle_cut: ``[6*64, 251*64)``, ``N_c = 15680``;
    WT ``MatrixWhittle``, matrix.py:870)."""

    src: TimeDomain
    start: int
    n: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        object.__setattr__(self, "n", operator.index(self.n))
        if not 0 <= self.start < self.start + self.n <= self.src.N:
            raise ValueError(f"[{self.start}, {self.start + self.n}) is not inside [0, {self.src.N})")

    @property
    def dst(self) -> TimeDomain:
        return TimeDomain(self.n, self.src.dt, self.src.t0 + self.start * self.src.dt)

    def apply(self, x, be=None):
        _last_axis(x, self.src.N, "CropTime.apply")
        return x[..., self.start:self.start + self.n]

    def adjoint(self, y, be=None):
        _last_axis(y, self.n, "CropTime.adjoint")
        be = self._be(y, be)
        out = be.xp.zeros((*y.shape[:-1], self.src.N), dtype=y.dtype)
        return be.set(out, (..., slice(self.start, self.start + self.n)), y, copy=False)


@pytree_dataclass(data=(), meta=("src", "n_extra"))
@dataclass(frozen=True, eq=False)
class ZeroPadTime(LinearMap):
    """``n_extra`` zero samples appended at the end (the sinusoid pad: 16 columns, 1024
    samples; cyclically equivalent to 8 + 8).  WT ``Variant.extend`` (matrix.py:702-709)."""

    src: TimeDomain
    n_extra: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_extra", operator.index(self.n_extra))
        if self.n_extra < 0:
            raise ValueError(f"n_extra must be >= 0; got {self.n_extra}")

    @property
    def dst(self) -> TimeDomain:
        return TimeDomain(self.src.N + self.n_extra, self.src.dt, self.src.t0)

    def apply(self, x, be=None):
        _last_axis(x, self.src.N, "ZeroPadTime.apply")
        be = self._be(x, be)
        x = be.asarray(x)
        out = be.xp.zeros((*x.shape[:-1], self.src.N + self.n_extra), dtype=x.dtype)
        return be.set(out, (..., slice(0, self.src.N)), x, copy=False)

    def adjoint(self, y, be=None):
        _last_axis(y, self.src.N + self.n_extra, "ZeroPadTime.adjoint")
        return y[..., :self.src.N]
