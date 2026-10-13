"""Linear maps between domains and the lazy congruence they push (spec §4.2).

A map is a frozen dataclass registered as a JAX pytree (``wdm.backend.pytree_dataclass``):
its host-built constants (index tables, atom matrices, Dirichlet kernels) are the pytree
*data*, its domains and sizes the static *meta*.  Pass a map to ``jax.jit`` as an
argument, never close over it, and call :meth:`LinearMap.on` once to move its constants
to the device (spec §3.3, the host-constant rule §0.4).

Every constructor builds its constants in host NumPy float64 when they are not given
(the ``_BUILD`` sentinel); ``jax.tree_util`` rebuilds a map from its leaves without
recomputing anything.

Map kinds (what ``apply`` takes and returns; ``adjoint`` is the adjoint for the real
inner product, ``<z, z'> = Re sum conj(z) z'`` on complex vectors):

==========  ==========================  ================================================
kind        apply                       push(cov) (``PushedCovariance``)
==========  ==========================  ================================================
"real"      real -> real                ``M Sigma M^T``; needs ``cov.matvec``
"analysis"  real -> complex bins        ``F Sigma F^H`` (complex Hermitian)
"cells"     complex bins -> real cells  ``1/2 Re(C M' C^H)``; needs ``cov.dense()``
"complex"   complex -> complex          ``A M' A^H`` for complex-linear maps (BinSelect)
"widely"    complex -> complex          not pushable (A z + B conj z; raises TypeError)
==========  ==========================  ================================================

A covariance without ``matvec`` -- every covariance on WDM cells -- cannot be pushed through
a real or analysis map: ``PushedCovariance.rows``/``matvec`` raise a ``TypeError`` naming
gap G9 (spec section 4.2's WDM -> time ``W^T Lambda W / dt^2`` is not implemented).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from wdm.backend import NUMPY, Backend, get_backend, pytree_dataclass, require_x64

from ..domains import FourierDomain, TimeDomain, WDMDomain

__all__ = [
    "BinSelect",
    "Compose",
    "Identity",
    "LinearMap",
    "PushedCovariance",
    "Restrict",
    "SupportsDense",
    "SupportsMatvec",
    "domain_size",
    "on_backend",
]


class _Build:
    """Sentinel default of every constant field: "build it on the host"."""

    def __repr__(self) -> str:
        return "<build>"


_BUILD: Any = _Build()


def _last_axis(x: Any, n: int, what: str) -> None:
    """Refuse a vector whose last axis is not ``n`` long (shapes are static under ``jit``)."""
    if np.shape(x)[-1:] != (n,):
        raise ValueError(f"{what}: last axis must have length {n}; got shape {np.shape(x)}")


def domain_size(dom: Any) -> int:
    """Length of a vector on ``dom``: samples, bins (complex) or cells."""
    if isinstance(dom, TimeDomain):
        return int(dom.N)
    if isinstance(dom, FourierDomain):
        return int(dom.n)
    if isinstance(dom, WDMDomain):
        return len(dom.cells.m)
    raise TypeError(f"not a domain: {dom!r}")


def on_backend(obj: Any, be: Backend, fields: tuple[str, ...]) -> Any:
    """``obj`` with its constant ``fields`` (NumPy arrays, tuples of them, or nested maps)
    moved to ``be`` by ``be.asarray`` -- the host-constant rule (spec §0.4): build once on
    the host, then ``device_put``.  NumPy returns ``obj`` itself.  Any other backend first
    calls ``require_x64()`` (spec §3.3): a ``Backend`` object kept from earlier would
    otherwise turn the float64 constants into float32 silently when x64 is off."""
    if be.name == "numpy":
        return obj
    require_x64()

    def move(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return be.asarray(v)
        if isinstance(v, tuple):
            return tuple(move(u) for u in v)
        if isinstance(v, LinearMap):
            return v.on(be)
        return v

    return dataclasses.replace(obj, **{k: move(getattr(obj, k)) for k in fields})


class SupportsMatvec(Protocol):
    """A real symmetric covariance on a map's source: ``matvec(X)`` is ``X Sigma`` row by
    row, ``(..., n) -> (..., n)``."""

    def matvec(self, X: Any, be: Backend | None = None) -> Any: ...


class SupportsDense(Protocol):
    """A complex Hermitian covariance ``M' = E[z z^H]`` of a map's source bins."""

    def dense(self, be: Backend | None = None) -> Any: ...


class LinearMap:
    """Base of every map.  Subclasses define ``src``, ``dst``, ``kind`` and ``apply``/``adjoint``."""

    kind: str = "real"

    @property
    def shape(self) -> tuple[int, int]:
        return domain_size(self.dst), domain_size(self.src)

    def _be(self, x: Any, be: Backend | None) -> Backend:
        """``be``, else JAX iff ``x`` or one of the map's constants is a ``jax.Array``."""
        if be is not None:
            return be
        consts = [getattr(self, f.name) for f in dataclasses.fields(self)]
        return get_backend(x, *(v for v in consts
                                if hasattr(v, "shape") and not isinstance(v, LinearMap)))

    def apply(self, x: Any, be: Backend | None = None) -> Any:
        raise NotImplementedError

    def adjoint(self, y: Any, be: Backend | None = None) -> Any:
        raise NotImplementedError

    def rows(self, idx: Any) -> np.ndarray:
        """Host dense rows ``M[idx, :]`` (``(k, n_src)``), from the adjoint of unit vectors."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        E = np.zeros((idx.size, self.shape[0]))
        E[np.arange(idx.size), idx] = 1.0
        out = np.asarray(self.adjoint(E, NUMPY))
        return np.conj(out) if np.iscomplexobj(out) else out

    def dense(self) -> np.ndarray:
        """The whole host matrix (tests and small grids only)."""
        return self.rows(np.arange(self.shape[0]))

    def push(self, cov: Any) -> "PushedCovariance":
        """``M Sigma M^T`` (or its complex counterpart, see the module table), lazily."""
        if self.kind == "widely":
            raise TypeError(
                f"{type(self).__name__} is widely linear (A z + B conj z): a circular bin "
                "covariance does not stay circular, so it cannot be pushed; the GB pad model "
                "uses ZeroPadded('cell_congruence') instead")
        return PushedCovariance(self, cov)

    def on(self, be: Backend) -> "LinearMap":
        """The same map with every array constant (and nested map) moved to ``be``."""
        names = tuple(f.name for f in dataclasses.fields(self)
                      if isinstance(getattr(self, f.name), (np.ndarray, LinearMap)))
        return on_backend(self, be, names)


@dataclass(frozen=True, eq=False)
class PushedCovariance:
    """``M Sigma M^T`` evaluated on demand: ``rows(idx)``, ``diag()``, ``dense()``.

    Rows follow the oracle order of each map kind: ``"real"`` --
    ``M (Sigma M[idx]^T)``, i.e. ``map.apply(cov.matvec(map.rows(idx)))`` (WT
    ``MatrixWdm._exact_rows``: atoms, then the covariance, then the forward transform);
    ``"cells"`` -- ``0.5 * Re((C[idx] @ M') @ conj(C).T)`` (GB ``exact_rows_builder``,
    matrix.py:686-693); ``"analysis"`` -- ``F[idx] Sigma F^H``; ``"complex"`` --
    ``(A M' A^H)[idx]``.
    """

    map: LinearMap
    cov: Any

    @property
    def size(self) -> int:
        return self.map.shape[0]

    def _matvec(self) -> Any:
        """``cov.matvec``, or a TypeError naming gap G9: a covariance on WDM cells
        (``DiagonalCovariance``, ``Eq45Covariance``, ...) has no fixed-noise-state ``matvec``,
        so spec section 4.2's WDM -> time translation ``W^T Lambda W / dt^2`` is not available."""
        fn = getattr(self.cov, "matvec", None)
        if fn is None:
            raise TypeError(f"{type(self.cov).__name__} has no matvec: a push through a "
                            f"{self.map.kind!r} map needs a fixed-noise-state SupportsMatvec "
                            "(X -> X Sigma, e.g. ExactTimeCovariance.at(nu)) "
                            "(gap G9: WDM->time translation not implemented)")
        return fn

    def rows(self, idx: Any, be: Backend | None = None) -> Any:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        kind = self.map.kind
        if kind == "real":
            matvec = self._matvec()
            R = self.map.rows(idx)
            return self.map.apply(matvec(R, be), be)
        if kind == "cells":
            return self.map.push_rows(self.cov.dense(be), idx, be)
        if kind == "complex":
            return self.map.push_rows(self.cov.dense(be), idx, be)
        if kind == "analysis":
            matvec = self._matvec()
            F = self.map.rows(idx)                                   # (k, N) complex
            Y = matvec(F.real, be) + 1j * matvec(F.imag, be)
            return np.conj(self.map.apply_complex(np.conj(Y), be))
        raise TypeError(f"cannot push through a {kind!r} map")

    def dense(self, be: Backend | None = None) -> Any:
        return self.rows(np.arange(self.size), be)

    def diag(self, be: Backend | None = None) -> Any:
        D = self.dense(be)
        be = be or get_backend(D)
        out = be.xp.diagonal(D)
        return be.xp.real(out) if be.xp.iscomplexobj(out) else out

    def matvec(self, X: Any, be: Backend | None = None) -> Any:
        """``X (M Sigma M^T)`` row by row (real kinds), so pushes compose."""
        if self.map.kind != "real":
            raise TypeError("matvec of a pushed covariance is defined for real maps only")
        return self.map.apply(self._matvec()(self.map.adjoint(X, be), be), be)


@pytree_dataclass(data=(), meta=("src",))
@dataclass(frozen=True, eq=False)
class Identity(LinearMap):
    """The identity on ``src`` (the sinusoid pad model's samples, spec §4.8).  A scaled
    ``WDMDomain`` passes through unchanged (nothing to apply); ``GaussianLikelihood``
    refuses it (gap G4)."""

    src: Any

    @property
    def dst(self) -> Any:
        return self.src

    @property
    def kind(self) -> str:  # type: ignore[override]
        return "complex" if isinstance(self.src, FourierDomain) else "real"

    def apply(self, x, be=None):
        return x

    def adjoint(self, y, be=None):
        return y

    def push_rows(self, Mp, idx, be=None):
        return Mp[idx]


@pytree_dataclass(data=("outer", "inner"), meta=())
@dataclass(frozen=True, eq=False)
class Compose(LinearMap):
    """``outer . inner`` (``inner`` first).  ``inner.dst`` must equal ``outer.src``."""

    outer: Any
    inner: Any

    def __post_init__(self) -> None:
        if isinstance(self.outer, LinearMap) and isinstance(self.inner, LinearMap):
            if self.outer.src != self.inner.dst:
                raise ValueError(f"cannot compose: {self.inner.dst!r} != {self.outer.src!r}")

    @property
    def src(self) -> Any:
        return self.inner.src

    @property
    def dst(self) -> Any:
        return self.outer.dst

    @property
    def kind(self) -> str:  # type: ignore[override]
        if self.outer.kind == "real" and self.inner.kind == "real":
            return "real"
        return "composite"

    def apply(self, x, be=None):
        return self.outer.apply(self.inner.apply(x, be), be)

    def adjoint(self, y, be=None):
        return self.inner.adjoint(self.outer.adjoint(y, be), be)

    def rows(self, idx):
        # Row r of outer.inner is outer[r] inner = conj(inner^H conj(outer[r])): the same
        # conj convention as LinearMap.rows.  An all-real composite is untouched.
        R = self.outer.rows(idx)
        out = np.asarray(self.inner.adjoint(np.conj(R) if np.iscomplexobj(R) else R, NUMPY))
        return np.conj(out) if np.iscomplexobj(out) else out

    def push(self, cov):
        if self.kind != "real":
            raise TypeError("Compose.push is defined for real maps only "
                            f"({self.outer.kind} after {self.inner.kind})")
        return PushedCovariance(self, cov)


@pytree_dataclass(data=("pos",), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class Restrict(LinearMap):
    """The cells ``dst.cells`` (a subset, in its own order) of a vector on ``src.cells``."""

    src: WDMDomain
    dst: WDMDomain
    pos: Any = field(default=_BUILD, kw_only=True)

    def __post_init__(self) -> None:
        if self.pos is _BUILD:
            self.src.require_unit_scale("Restrict")
            self.dst.require_unit_scale("Restrict")
            where = {c: i for i, c in enumerate(zip(self.src.cells.m, self.src.cells.n))}
            missing = [c for c in zip(self.dst.cells.m, self.dst.cells.n) if c not in where]
            if missing or self.src.cells.grid != self.dst.cells.grid:
                raise ValueError(f"dst is not a subset of src ({len(missing)} cells missing)")
            object.__setattr__(self, "pos", np.asarray(
                [where[c] for c in zip(self.dst.cells.m, self.dst.cells.n)], dtype=np.int64))

    def apply(self, x, be=None):
        return x[..., self.pos]

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        out = be.xp.zeros((*y.shape[:-1], self.shape[1]), dtype=y.dtype)
        return be.set(out, (..., self.pos), y, copy=False)


@pytree_dataclass(data=(), meta=("src", "dst"))
@dataclass(frozen=True, eq=False)
class BinSelect(LinearMap):
    """The bins of the sub-window ``dst`` of ``src`` (store -> window -> nominal, spec §4.9).

    ``BinSelect(src, (k0, n))`` takes absolute bins ``[k0, k0 + n)`` of ``src``'s record.
    """

    src: FourierDomain
    dst: Any

    kind = "complex"

    def __post_init__(self) -> None:
        if isinstance(self.dst, tuple):
            k0, n = (int(v) for v in self.dst)
            object.__setattr__(self, "dst", FourierDomain(self.src.N, self.src.dt, k0, n,
                                                          self.src.units))
        d, s = self.dst, self.src
        if (d.N, d.dt, d.units) != (s.N, s.dt, s.units) or d.k0 < s.k0 or d.k0 + d.n > s.k0 + s.n:
            raise ValueError(f"bins [{d.k0}, {d.k0 + d.n}) are not inside [{s.k0}, {s.k0 + s.n})")

    @property
    def offset(self) -> int:
        return self.dst.k0 - self.src.k0

    def apply(self, z, be=None):
        return z[..., self.offset:self.offset + self.dst.n]

    def adjoint(self, y, be=None):
        be = self._be(y, be)
        out = be.xp.zeros((*y.shape[:-1], self.src.n), dtype=complex)
        return be.set(out, (..., slice(self.offset, self.offset + self.dst.n)), y, copy=False)

    def push_rows(self, Mp, idx, be=None):
        sel = self.offset + np.arange(self.dst.n)
        return Mp[np.ix_(self.offset + np.asarray(idx), sel)]
