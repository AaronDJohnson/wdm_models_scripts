"""Where a data vector lives (spec section 4.1).

Every domain is a frozen, hashable dataclass, so it can be the static part of a
``jax.jit`` key and the metadata of a data pytree (:mod:`domainia.data`).

* :class:`TimeDomain`: ``N`` samples at spacing ``dt`` from ``t0`` (``t_j = t0 + j dt``).
* :class:`FourierDomain`: the absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample
  record, in one of three unit conventions (``"rfft"``: ``numpy.fft.rfft(x)``;
  ``"dt_rfft"``: ``dt * rfft(x)``; ``"sigma0"``: ``rfft(x) / sigma0``, the GB
  normalisation).
* :class:`CellSet`: a set of active WDM cells of one ``wdm.Grid``, always in **band
  order** (``m`` ascending, then ``n``), i.e. a subsequence of
  ``wdm.grid.packed_cells(Nf, Nt)``.
* :class:`WDMDomain`: a cell set with a constant ``scale`` (the GB cells are
  ``Re(C z)`` with unit-norm atoms on ``z = rfft / sigma0``; the sinusoid uses 1).
* :class:`ChannelAxis`: named channels (``("A", "E")`` for the GB); empty for one
  channel without a channel axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import wdm
from wdm.grid import packed_cells

__all__ = [
    "FOURIER_UNITS",
    "CellSet",
    "ChannelAxis",
    "FourierDomain",
    "TimeDomain",
    "WDMDomain",
]

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

#: The unit conventions of :class:`FourierDomain`.
FOURIER_UNITS = ("rfft", "dt_rfft", "sigma0")


def _int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int; got {type(value).__name__}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _positive_float(name: str, value: object) -> float:
    v = float(value)  # type: ignore[arg-type]
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return v


@dataclass(frozen=True)
class TimeDomain:
    """``N`` real samples ``t_j = t0 + j dt``."""

    N: int
    dt: float
    t0: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "N", _int("N", self.N, minimum=1))
        object.__setattr__(self, "dt", _positive_float("dt", self.dt))
        t0 = float(self.t0)
        if not math.isfinite(t0):
            raise ValueError(f"t0 must be finite; got {self.t0!r}")
        object.__setattr__(self, "t0", t0)

    @property
    def size(self) -> int:
        return self.N

    def times(self) -> NDArray[np.float64]:
        """``t0 + arange(N) * dt`` (``t0 = 0`` gives ``arange(N) * dt`` bit for bit)."""
        return self.t0 + np.arange(self.N) * self.dt


@dataclass(frozen=True)
class FourierDomain:
    """The absolute rfft bins ``[k0, k0 + n)`` of an ``N``-sample record at spacing ``dt``."""

    N: int
    dt: float
    k0: int
    n: int
    units: Literal["rfft", "dt_rfft", "sigma0"]

    def __post_init__(self) -> None:
        N = _int("N", self.N, minimum=1)
        k0 = _int("k0", self.k0, minimum=0)
        n = _int("n", self.n, minimum=1)
        if k0 + n > N // 2 + 1:
            raise ValueError(f"bins [{k0}, {k0 + n}) exceed the {N // 2 + 1} rfft bins of N={N}")
        if self.units not in FOURIER_UNITS:
            raise ValueError(f"units must be one of {FOURIER_UNITS}; got {self.units!r}")
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "dt", _positive_float("dt", self.dt))

    @property
    def size(self) -> int:
        return self.n

    def bins(self) -> IntArray:
        """The absolute rfft bin indices ``k0 + arange(n)``."""
        return self.k0 + np.arange(self.n)

    def frequencies(self) -> NDArray[np.float64]:
        """``bins / (N dt)``."""
        return self.bins() / (self.N * self.dt)


def _cell_index(name: str, value: object) -> IntArray:
    """``value`` as a fresh flat int64 array (a copy: the caller's array is never touched).

    Integer dtypes pass; finite integral floats (``3.0``) are accepted; anything else
    (``2.9999999``, NaN, booleans) raises instead of being truncated by the int64 cast.
    """
    raw = np.asarray(value)
    if raw.size and raw.dtype.kind not in "iu":
        if raw.dtype.kind != "f":
            raise ValueError(f"{name} must hold integer cell indices; got dtype {raw.dtype}")
        bad = ~np.isfinite(raw) | (raw != np.trunc(raw))
        if np.any(bad):
            first = raw.reshape(-1)[np.flatnonzero(bad.reshape(-1))[0]]
            raise ValueError(f"{name} must hold integer cell indices; got {first!r}")
    return np.array(raw, dtype=np.int64).reshape(-1)


def _active(m: IntArray, n: IntArray, Nf: int) -> BoolArray:
    """``wdm.active_mask`` evaluated at the cells ``(m, n)``: interior bands every ``n``,
    DC even ``n``, Nyquist ``(Nf + n)`` even."""
    interior = (m >= 1) & (m <= Nf - 1)
    dc = (m == 0) & (n % 2 == 0)
    nyq = (m == Nf) & ((Nf + n) % 2 == 0)
    return interior | dc | nyq


def _packed_index(m: IntArray, n: IntArray, Nf: int, Nt: int) -> IntArray:
    """Position of active cell ``(m, n)`` in ``wdm.grid.packed_cells(Nf, Nt)`` order:
    ``Nt/2`` DC cells, ``Nt`` cells per interior band, then the Nyquist cells."""
    h = Nt // 2
    out = h + (m - 1) * Nt + n
    out = np.where(m == 0, n // 2, out)
    return np.where(m == Nf, h + (Nf - 1) * Nt + n // 2, out)


@dataclass(frozen=True, eq=False)
class CellSet:
    """Active cells ``(m[i], n[i])`` of ``grid`` in band order (no repeats).

    Equality and hashing use ``(grid, m, n)``; the hash is computed once.  Construct
    with :meth:`all_active`, :meth:`bands` or :meth:`from_mask`, then narrow with
    :meth:`drop_columns`, :meth:`columns` and :meth:`complement`.
    """

    grid: wdm.Grid
    m: tuple[int, ...]
    n: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grid, wdm.Grid):
            raise TypeError(f"grid must be a wdm.Grid; got {type(self.grid).__name__}")
        m = _cell_index("m", self.m)                            # copies: the flags below
        n = _cell_index("n", self.n)                            # must not touch the caller's
        if m.shape != n.shape:
            raise ValueError(f"m and n differ in length: {m.size} != {n.size}")
        Nf, Nt = self.grid.Nf, self.grid.Nt
        if m.size and (m.min() < 0 or m.max() > Nf or n.min() < 0 or n.max() >= Nt):
            raise ValueError(f"cells outside the ({Nf + 1}, {Nt}) grid")
        if not np.all(_active(m, n, Nf)):
            bad = np.flatnonzero(~_active(m, n, Nf))[0]
            raise ValueError(f"cell (m={m[bad]}, n={n[bad]}) is not active on this grid")
        key = m * Nt + n
        if key.size > 1 and not np.all(np.diff(key) > 0):
            raise ValueError("cells must be in band order (m, then n, strictly increasing)")
        m.setflags(write=False)
        n.setflags(write=False)
        object.__setattr__(self, "m", tuple(int(v) for v in m))
        object.__setattr__(self, "n", tuple(int(v) for v in n))
        object.__setattr__(self, "_m", m)
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_hash", hash((self.grid, self.m, self.n)))

    # -- identity -------------------------------------------------------------
    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellSet):
            return NotImplemented
        return (self._hash == other._hash and self.grid == other.grid  # type: ignore[attr-defined]
                and self.m == other.m and self.n == other.n)

    def __repr__(self) -> str:
        bands = sorted(set(self.m))
        span = f"{bands[0]}..{bands[-1]}" if bands else "-"
        return f"CellSet(grid={self.grid!r}, size={self.size}, bands={span})"

    # -- arrays ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.m)

    @property
    def m_array(self) -> IntArray:
        """``m`` as a read-only int64 array."""
        return self._m  # type: ignore[attr-defined]

    @property
    def n_array(self) -> IntArray:
        """``n`` as a read-only int64 array."""
        return self._n  # type: ignore[attr-defined]

    def packed_positions(self) -> IntArray:
        """Each cell's index in ``wdm.grid.packed_cells(Nf, Nt)`` (the full active order)."""
        return _packed_index(self._m, self._n, self.grid.Nf, self.grid.Nt)  # type: ignore[attr-defined]

    def keep_mask(self) -> BoolArray:
        """Boolean mask over the grid's ``N`` packed cells (``wdm.evolutionary.eq45_plan`` keep)."""
        keep = np.zeros(self.grid.N, dtype=bool)
        keep[self.packed_positions()] = True
        return keep

    def band_sizes(self) -> tuple[tuple[int, int], ...]:
        """``((m, count), ...)`` for every band present, ascending."""
        bands, counts = np.unique(self._m, return_counts=True)  # type: ignore[attr-defined]
        return tuple((int(b), int(c)) for b, c in zip(bands, counts, strict=True))

    def index_of(self, sub: CellSet) -> IntArray:
        """Positions of ``sub``'s cells within ``self`` (``sub`` must be a subset)."""
        if sub.grid != self.grid:
            raise ValueError("cell sets live on different grids")
        Nt = self.grid.Nt
        mine = self._m * Nt + self._n  # type: ignore[attr-defined]
        theirs = sub._m * Nt + sub._n  # type: ignore[attr-defined]
        pos = np.searchsorted(mine, theirs)
        if theirs.size and (pos.max() >= mine.size or not np.array_equal(mine[pos], theirs)):
            raise ValueError("sub is not a subset of this cell set")
        return pos.astype(np.int64)

    # -- constructors -----------------------------------------------------------
    @classmethod
    def all_active(cls, grid: wdm.Grid) -> CellSet:
        """Every active cell: ``wdm.grid.packed_cells(Nf, Nt)``."""
        m, n = packed_cells(grid.Nf, grid.Nt)
        return cls(grid, tuple(m.tolist()), tuple(n.tolist()))

    @classmethod
    def bands(cls, grid: wdm.Grid, bands: tuple[int, ...] | list[int]) -> CellSet:
        """Every active cell of the bands ``bands`` (strictly ascending)."""
        b = np.asarray(bands, dtype=np.int64).reshape(-1)
        if b.size == 0 or np.any(np.diff(b) <= 0):
            raise ValueError(f"bands must be non-empty and strictly ascending; got {bands!r}")
        if b.min() < 0 or b.max() > grid.Nf:
            raise ValueError(f"bands must lie in 0..{grid.Nf}; got {bands!r}")
        n_all = np.arange(grid.Nt)
        ms, ns = [], []
        for mm in b:
            nn = n_all[_active(np.full(grid.Nt, mm), n_all, grid.Nf)]
            ms.append(np.full(nn.size, mm))
            ns.append(nn)
        m, n = np.concatenate(ms), np.concatenate(ns)
        return cls(grid, tuple(m.tolist()), tuple(n.tolist()))

    @classmethod
    def from_mask(cls, grid: wdm.Grid, keep: BoolArray) -> CellSet:
        """The cells where ``keep`` (a boolean mask over the ``N`` packed cells) is True."""
        keep = np.asarray(keep)
        if keep.dtype != np.bool_ or keep.shape != (grid.N,):
            raise ValueError(f"keep must be a boolean mask of shape ({grid.N},)")
        m, n = packed_cells(grid.Nf, grid.Nt)
        return cls(grid, tuple(m[keep].tolist()), tuple(n[keep].tolist()))

    # -- narrowing ----------------------------------------------------------------
    def _select(self, sel: BoolArray) -> CellSet:
        return CellSet(self.grid, tuple(self._m[sel].tolist()),  # type: ignore[attr-defined]
                       tuple(self._n[sel].tolist()))  # type: ignore[attr-defined]

    def _edge(self, lead: int, trail: int) -> BoolArray:
        lead = _int("lead", lead, minimum=0)
        trail = _int("trail", trail, minimum=0)
        if lead + trail > self.grid.Nt:
            raise ValueError(f"lead + trail = {lead + trail} exceeds Nt = {self.grid.Nt}")
        cols = np.asarray(wdm.grid.edge_columns(self.grid.Nt, lead, trail), dtype=np.int64)
        return np.isin(self._n, cols)  # type: ignore[attr-defined]

    def drop_columns(self, lead: int, trail: int) -> CellSet:
        """Remove the cells in the first ``lead`` and last ``trail`` time columns (``*_cut``)."""
        return self._select(~self._edge(lead, trail))

    def columns(self, lead: int, trail: int) -> CellSet:
        """Keep only the cells in the first ``lead`` and last ``trail`` time columns."""
        return self._select(self._edge(lead, trail))

    def complement(self, sub: CellSet) -> CellSet:
        """The cells of ``self`` that are not in ``sub`` (``sub`` must be a subset)."""
        pos = self.index_of(sub)
        sel = np.ones(self.size, dtype=bool)
        sel[pos] = False
        return self._select(sel)


@dataclass(frozen=True)
class WDMDomain:
    """A cell set and the constant ``scale`` that relates its cells to the WDM coefficients."""

    cells: CellSet
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.cells, CellSet):
            raise TypeError(f"cells must be a CellSet; got {type(self.cells).__name__}")
        object.__setattr__(self, "scale", _positive_float("scale", self.scale))

    @property
    def size(self) -> int:
        return self.cells.size


@dataclass(frozen=True)
class ChannelAxis:
    """Named channels; ``ChannelAxis()`` means one channel and no channel axis."""

    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not all(isinstance(s, str) and s for s in names):
            raise TypeError(f"channel names must be non-empty strings; got {self.names!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique; got {names!r}")
        object.__setattr__(self, "names", names)

    @property
    def size(self) -> int:
        return len(self.names)
