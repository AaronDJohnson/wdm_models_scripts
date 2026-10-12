"""Factorisations and solves of the WDM II covariances (spec sections 3.4 and 4).

Every routine is a line-for-line port of one oracle code path and keeps that oracle's
library call at each call site (spec section 0, rule 1).  The Cholesky implementation
is pinned per site through ``Backend.cholesky(A, impl=...)``:

==========================  ==========  ===============================================
routine                     ``impl``    oracle
==========================  ==========  ===============================================
:func:`blocktri_factor`     ``numpy``   WT ``matrix.BlockTri`` (ragged), WG ``_btri_factor``
                                        (padded)
:func:`cblock_factor`       ``numpy``   GB ``fullcache.block_factor`` (12-month exact)
:func:`chol_factor`         per call    WT ``Solver(dense)`` / ``marginal_lnlike`` (scipy);
                                        GB ``_dense_quad_logdet``, ``MatrixCache``,
                                        ``fullcache`` dense, ``lnl_complex`` (numpy)
:func:`schur_factor`        per call    WT ``Schur`` (scipy); GB ``EdgeSchurCache`` (numpy);
                                        WG ``_factor`` (JAX)
:func:`padded_factor`       per call    WT ``PaddedMarginal`` (scipy); WG ``_factor`` pad
                                        branch (JAX)
==========================  ==========  ===============================================

Layouts of a block-tridiagonal matrix (spec section 3.2):

* ``"ragged"``: Python lists of per-band blocks, a host loop.  Runs eagerly on either
  backend (never under ``jit``); the NumPy golden default, bitwise WT.
* ``"padded"``: ``(nb, B, B)`` arrays with identity dummy slots (:class:`BandLayout`),
  a ``be.scan`` over the bands; jit/vmap-able, the JAX default (WG).

Factors are pytrees (:func:`wdm.backend.pytree_dataclass`): they stack along a rung
axis and pass through ``jax.jit``/``vmap``.  A factor whose Cholesky failed carries
``ok = False`` (NaN entries on the padded layout, a truncated factor on the ragged
one); callers map that to ``-inf`` (spec section 4.7, PD policy).  The JAX Cholesky
reads the lower triangle only (``symmetrize_input=False``, see ``wdm.backend``); WG
called ``jnp.linalg.cholesky`` with JAX's default, which differs by <= 7.4e-14 relative
on inputs that are not bitwise symmetric (JAX is never claimed bitwise).

Constants (:class:`BandLayout`, edge/interior index arrays, ``Wp``) are host NumPy
float64/int64 built once; under ``jax.jit`` pass them as arguments, never close over
them, and move them with :func:`device_constants` (which checks ``jax_enable_x64``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from wdm.backend import Backend, get_backend, is_tracer, pytree_dataclass

from .domains import CellSet

__all__ = [
    "BandLayout",
    "BlockTriFactor",
    "CholFactor",
    "DiagFactor",
    "blocktri_factor",
    "blocktri_solve",
    "blocktri_solve_packed",
    "blocktri_whiten",
    "blocktri_whiten_packed",
    "chol_backward",
    "chol_factor",
    "chol_solve",
    "chol_whiten",
    "diag_factor",
    "diag_solve",
]

Layout = Literal["ragged", "padded"]
CholImpl = Literal["numpy", "scipy"]


def _be(be: Backend | None, *arrays: Any) -> Backend:
    return be or get_backend(*[a for a in arrays if a is not None])


def _logdet_from_chol(xp: Any, L: Any) -> Any:
    """``2 sum log Re diag L`` -- the expression every oracle uses (``np.real`` of a real
    array is the array itself, so the real and complex forms agree bit for bit)."""
    return 2.0 * xp.sum(xp.log(xp.real(xp.diagonal(L, axis1=-2, axis2=-1))))


# ---------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("v", "logdet", "ok"))
@dataclass(frozen=True, eq=False)
class DiagFactor:
    """``C = diag(v)``."""

    v: Any
    logdet: Any
    ok: Any


def diag_factor(v: Any, *, be: Backend | None = None) -> DiagFactor:
    """WT ``Solver(diag=v)`` (matrix.py:579-596); ``logdet = sum(log v)``, ``ok = all(v > 0)``."""
    be = _be(be, v)
    xp = be.xp
    v = be.asarray(v)
    return DiagFactor(v, xp.sum(xp.log(v)), xp.all(v > 0))


def diag_solve(f: DiagFactor, b: Any, *, be: Backend | None = None) -> Any:
    """``C^-1 b`` as WT ``Solver.solve`` / WG ``_approx_solve``: ``b / v`` (column-wise for 2-D ``b``)."""
    return b / (f.v if b.ndim == 1 else f.v[:, None])


# ---------------------------------------------------------------------------
# Dense Cholesky (real symmetric or complex Hermitian)
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("L", "logdet", "ok"), meta=("impl",))
@dataclass(frozen=True, eq=False)
class CholFactor:
    """``C = L L^H``, ``L`` lower triangular (upper triangle zero)."""

    L: Any
    logdet: Any
    ok: Any
    impl: str


def chol_factor(A: Any, *, impl: CholImpl, symmetrize: bool = False,
                be: Backend | None = None) -> CholFactor:
    """Cholesky of ``A`` (``0.5 (A + A^T)`` first when ``symmetrize``).

    ``impl="scipy"``: WT ``Solver(dense)``, ``marginal_lnlike`` (``cho_factor``);
    ``impl="numpy"``: GB ``_dense_quad_logdet`` / ``MatrixCache.state`` (with
    ``symmetrize=True``, matrix.py:928-938, 978-983), GB ``fullcache`` dense and
    ``lnl_complex`` (``symmetrize=False``).
    """
    be = _be(be, A)
    xp = be.xp
    A = be.asarray(A)
    if symmetrize:
        A = 0.5 * (A + xp.swapaxes(A, -1, -2))
    L, ok = be.cholesky(A, impl=impl)
    return CholFactor(L, _logdet_from_chol(xp, L), ok, impl)


def chol_whiten(f: CholFactor, b: Any, *, be: Backend | None = None) -> Any:
    """``L^-1 b`` (GB ``fullcache._solve_lower``, ``matrix._solve_lower``, ``lnl_complex``)."""
    be = _be(be, f.L, b)
    return be.solve_triangular(f.L, b, lower=True)


def chol_backward(f: CholFactor, u: Any, *, be: Backend | None = None) -> Any:
    """``L^-H u``: ``trans="T"`` for a real ``L`` (GB ``matrix._solve_lower(trans=True)``),
    ``trans="C"`` for a complex one (GB ``fullcache._solve_upper_h``).  scipy takes a
    different LAPACK path for the two, so the choice follows the oracle by dtype."""
    be = _be(be, f.L, u)
    trans = 2 if be.xp.iscomplexobj(f.L) else 1
    return be.solve_triangular(f.L, u, lower=True, trans=trans)


def chol_solve(f: CholFactor, b: Any, *, be: Backend | None = None) -> Any:
    """``C^-1 b`` by ``cho_solve`` (WT ``Solver(dense)``, ``Schur``, ``PaddedMarginal``)."""
    be = _be(be, f.L, b)
    return be.cho_solve(f.L, b)


# ---------------------------------------------------------------------------
# Block-tridiagonal (real symmetric): band layouts
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("valid", "src", "back"), meta=("nb", "B", "n"))
@dataclass(frozen=True, eq=False)
class BandLayout:
    """Band-major positions ``0..n-1`` laid out as ``(nb, B)`` slots with dummies.

    WG ``_BandLayout`` (gpu_sinusoid/wdmtoysgpu/matrix.py:148-180) for a vector whose
    bands are contiguous: ``valid[m, i]`` marks real slots, ``src[m, i]`` is the
    position read into slot ``(m, i)`` (0 for a dummy) and ``back[p]`` the flat slot
    of position ``p``.  Host NumPy constants: pass the layout to ``jit`` as an argument.
    """

    nb: int
    B: int
    n: int
    valid: Any
    src: Any
    back: Any

    @classmethod
    def from_sizes(cls, sizes: Sequence[int]) -> BandLayout:
        """The layout of bands with ``sizes[m]`` cells each (concatenated in order)."""
        sizes = [int(s) for s in sizes]
        if not sizes or min(sizes) < 0 or max(sizes) == 0:
            raise ValueError(f"sizes must be non-negative with at least one band of cells; got {sizes}")
        off = np.concatenate([[0], np.cumsum(sizes)])
        nb, B, n = len(sizes), max(sizes), int(off[-1])
        slot = np.full((nb, B), -1)
        for m, s in enumerate(sizes):
            slot[m, :s] = np.arange(off[m], off[m] + s)
        valid = slot >= 0
        src = np.where(valid, slot, 0)
        flat = slot.ravel()
        back = np.empty(n, int)
        back[flat[flat >= 0]] = np.flatnonzero(flat >= 0)
        return cls(nb, B, n, valid, src, back)

    @classmethod
    def from_cells(cls, cells: CellSet) -> BandLayout:
        """One slot row per band present in a band-ordered :class:`~domainia.domains.CellSet`."""
        return cls.from_sizes([s for _, s in cells.band_sizes()])

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(s) for s in np.asarray(self.valid).sum(axis=1))

    def pack(self, b: Any, *, be: Backend | None = None) -> Any:
        """``(n, k) -> (nb, B, k)`` with zero dummies (WG ``_BandLayout.pack``)."""
        be = _be(be, b, self.src)
        xp = be.xp
        g = be.asarray(b)[be.asarray(self.src)]
        return xp.where(be.asarray(self.valid)[..., None], g, 0.0)

    def unpack(self, x: Any, *, be: Backend | None = None) -> Any:
        """``(nb, B, k) -> (n, k)`` (WG ``_BandLayout.unpack``)."""
        be = _be(be, x, self.back)
        return be.asarray(x).reshape(self.nb * self.B, -1)[be.asarray(self.back)]

    def add_identity_dummies(self, diag: Any, *, be: Backend | None = None) -> Any:
        """``diag + diag(1 on dummy slots)`` per band: WG ``_tri_factor``
        (``diag + vmap(jnp.diag)(where(valid, 0, 1))``, matrix.py:428)."""
        be = _be(be, diag, self.valid)
        xp = be.xp
        w = xp.where(be.asarray(self.valid), 0.0, 1.0)
        return be.asarray(diag) + w[:, :, None] * xp.eye(self.B)[None, :, :]

    def pack_blocks(self, diag: Sequence[Any], up: Sequence[Any | None], *,
                    be: Backend | None = None) -> tuple[Any, Any]:
        """Ragged blocks (``diag[m]`` square, ``up[m]`` of bands ``m, m+1`` or ``None``) as
        ``(nb, B, B)`` / ``(nb-1, B, B)`` arrays: zero padding, identity dummies.

        Host-side: the band sizes are read from the concrete ``valid``, so call it outside
        ``jit`` (a layout passed to ``jit`` as an argument is traced; JAX then raises
        ``TracerArrayConversionError``).  Under ``jit`` pass already-packed arrays."""
        if len(diag) != self.nb or len(up) != self.nb - 1:
            raise ValueError(f"need {self.nb} diagonal and {self.nb - 1} upper blocks; "
                             f"got {len(diag)} and {len(up)}")
        be = _be(be, *diag, *up)
        xp = be.xp
        sizes = self.sizes
        D = xp.zeros((self.nb, self.B, self.B))
        U = xp.zeros((max(self.nb - 1, 0), self.B, self.B))
        for m, blk in enumerate(diag):
            s = sizes[m]
            if s:
                D = be.set(D, (m, slice(0, s), slice(0, s)), be.asarray(blk), copy=False)
        for m, blk in enumerate(up):
            if blk is not None and sizes[m] and sizes[m + 1]:
                U = be.set(U, (m, slice(0, sizes[m]), slice(0, sizes[m + 1])), be.asarray(blk),
                           copy=False)
        return self.add_identity_dummies(D, be=be), U


@pytree_dataclass(data=("Ls", "Xs", "logdet", "ok", "band_layout"), meta=("layout",))
@dataclass(frozen=True, eq=False)
class BlockTriFactor:
    """``Lambda = L L^T`` with ``L`` lower block-bidiagonal: ``L_0 = chol(D_0)``,
    ``X_i = L_{i-1}^-1 U_{i-1}``, ``L_i = chol(D_i - X_i^T X_i)``.

    ``"ragged"``: ``Ls``/``Xs`` tuples (``Xs[i]`` ``None`` where there is no coupling,
    always for ``i = 0``); ``logdet`` a Python float.  ``"padded"``: ``Ls (nb, B, B)``,
    ``Xs (nb-1, B, B)``, ``band_layout`` the :class:`BandLayout` for natural-order vectors.
    """

    Ls: Any
    Xs: Any
    logdet: Any
    ok: Any
    band_layout: BandLayout | None
    layout: str


def _ragged_factor(diag: Sequence[Any], up: Sequence[Any | None], be: Backend) -> BlockTriFactor:
    # WT matrix.BlockTri.__init__ (src/wdm_toys/matrix.py:525-541), line for line.
    xp = be.xp
    Ls: list[Any] = []
    Xs: list[Any] = []
    prev = None
    for i, A in enumerate(diag):
        A = be.asarray(A)
        if i and prev is not None and up[i - 1] is not None and A.size and prev.size:
            X = be.solve_triangular(prev, be.asarray(up[i - 1]), lower=True)   # L_{i-1}^{-1} B
            A = A - X.T @ X
            Xs.append(X)
        else:
            Xs.append(None)
        if A.size:
            L, ok = be.cholesky(A, impl="numpy")
            if not bool(ok):
                return BlockTriFactor(tuple(Ls), tuple(Xs), float("nan"), False, None, "ragged")
        else:
            L = A
        Ls.append(L)
        prev = L
    # WT BlockTri.logdet (matrix.py:543-544): Python sum of per-block float sums.
    logdet = 2.0 * sum(float(xp.sum(xp.log(xp.diagonal(L)))) for L in Ls if L.size)
    return BlockTriFactor(tuple(Ls), tuple(Xs), logdet, True, None, "ragged")


def _padded_factor(diag: Any, up: Any, band_layout: BandLayout, be: Backend) -> BlockTriFactor:
    # WG _btri_factor (gpu_sinusoid/wdmtoysgpu/matrix.py:96-109), line for line.
    xp = be.xp
    L0, _ = be.cholesky(diag[0], impl="numpy")
    if diag.shape[0] == 1:
        Ls, Xs = L0[None], xp.zeros((0,) + tuple(diag.shape[1:]))
    else:
        def step(Lprev: Any, xs: Any) -> tuple[Any, Any]:
            A, U = xs
            X = be.solve_triangular(Lprev, U, lower=True)
            L, _ = be.cholesky(A - X.T @ X, impl="numpy")
            return L, (L, X)

        _, (Ls, Xs) = be.scan(step, L0, (diag[1:], up))
        Ls = xp.concatenate([L0[None], Ls], axis=0)
    dg = xp.diagonal(Ls, axis1=1, axis2=2)
    logdet = 2.0 * xp.sum(xp.log(dg))
    ok = xp.all(xp.isfinite(dg))
    return BlockTriFactor(Ls, Xs, logdet, ok, band_layout, "padded")


def blocktri_factor(diag: Any, up: Any, *, layout: Layout,
                    band_layout: BandLayout | None = None,
                    be: Backend | None = None) -> BlockTriFactor:
    """Block Cholesky of a symmetric block-tridiagonal matrix.

    ``layout="ragged"``: ``diag`` a sequence of square blocks (empty allowed), ``up`` a
    sequence of ``len(diag) - 1`` upper blocks or ``None`` (WT ``BlockTri``).  Eager only.

    ``layout="padded"``: ``band_layout`` is required.  ``diag``/``up`` are either ragged
    sequences (packed with :meth:`BandLayout.pack_blocks`) or ``(nb, B, B)`` /
    ``(nb-1, B, B)`` arrays that already carry identity dummies
    (:meth:`BandLayout.add_identity_dummies`, the WG ``_tri_factor`` order).
    """
    if layout == "ragged":
        if not isinstance(diag, (list, tuple)):
            raise TypeError("layout='ragged' needs a sequence of diagonal blocks")
        if len(up) != len(diag) - 1:
            raise ValueError(f"need {len(diag) - 1} upper blocks; got {len(up)}")
        be = _be(be, *diag)
        if any(is_tracer(a) for a in diag):
            raise ValueError("layout='ragged' runs eagerly only; use layout='padded' under jit")
        return _ragged_factor(diag, up, be)
    if layout == "padded":
        if band_layout is None:
            raise ValueError("layout='padded' needs band_layout")
        if isinstance(diag, (list, tuple)):
            diag, up = band_layout.pack_blocks(diag, up, be=be)
        be = _be(be, diag, up)
        diag, up = be.asarray(diag), be.asarray(up)
        if diag.shape != (band_layout.nb, band_layout.B, band_layout.B):
            raise ValueError(f"diag must be ({band_layout.nb}, {band_layout.B}, {band_layout.B}); "
                             f"got {diag.shape}")
        return _padded_factor(diag, up, band_layout, be)
    raise ValueError(f"layout must be 'ragged' or 'padded'; got {layout!r}")


def _failed(f: Any) -> bool:
    """True when ``f.ok`` is a concrete False (never for a tracer)."""
    return not is_tracer(f.ok) and not bool(f.ok)


def _require_complete(f: Any) -> None:
    """A factor that stopped at its first failed block (ragged) or skipped its own
    Cholesky because its base failed cannot be applied: raise.  Array factors with a
    failed Cholesky carry NaN instead, which propagates (the JAX behaviour)."""
    if _failed(f):
        raise ValueError("the factorisation failed (not positive definite): check .ok first "
                         "and map the row to -inf")


def _factor_arrays(f: BlockTriFactor) -> tuple[Any, ...]:
    """The factor's arrays, for picking the backend when ``b`` is a host array."""
    return tuple(f.Ls) if f.layout == "ragged" else (f.Ls,)


def _ragged_forward(f: BlockTriFactor, B: Any, be: Backend) -> list[Any]:
    # The forward half of WT BlockTri.solve (matrix.py:550-556).
    off = np.concatenate([[0], np.cumsum([L.shape[0] for L in f.Ls])])
    z: list[Any] = []
    for i, L in enumerate(f.Ls):
        r = B[off[i]:off[i + 1]]
        if f.Xs[i] is not None:
            r = r - f.Xs[i].T @ z[i - 1]
        z.append(be.solve_triangular(L, r, lower=True) if L.size else r)
    return z


def blocktri_whiten(f: BlockTriFactor, b: Any, *, be: Backend | None = None) -> Any:
    """``L^-1 b`` for ``b`` of shape ``(n,)`` or ``(n, k)`` in natural (band-major) order,
    so ``b^T Lambda^-1 b = |L^-1 b|^2`` (WG ``_forward_only``).

    Ragged: the forward substitution of WT ``BlockTri.solve``.  Padded: WG
    ``_btri_forward`` between :meth:`BandLayout.pack` and :meth:`BandLayout.unpack`
    (the dummy entries of the packed result are exactly zero).
    """
    if f.layout == "ragged":
        _require_complete(f)
    be = _be(be, b, *_factor_arrays(f))
    one = b.ndim == 1
    B = b[:, None] if one else b
    if f.layout == "ragged":
        out = be.xp.concatenate(_ragged_forward(f, be.asarray(B), be), axis=0)
    else:
        lay = f.band_layout
        out = lay.unpack(blocktri_whiten_packed(f, lay.pack(B, be=be), be=be), be=be)
    return out[:, 0] if one else out


def blocktri_solve(f: BlockTriFactor, b: Any, *, be: Backend | None = None) -> Any:
    """``Lambda^-1 b`` for ``b`` of shape ``(n,)`` or ``(n, k)`` in natural order.

    Ragged: WT ``BlockTri.solve`` (matrix.py:546-563) verbatim.  Padded: WG
    ``_approx_solve`` (``unpack(_btri_solve(pack(b)))``, matrix.py:605-612).
    """
    if f.layout == "ragged":
        _require_complete(f)
    be = _be(be, b, *_factor_arrays(f))
    xp = be.xp
    one = b.ndim == 1
    B = be.asarray(b[:, None] if one else b)
    if f.layout == "ragged":
        z = _ragged_forward(f, B, be)
        n = len(f.Ls)
        x: list[Any] = [None] * n
        for i in range(n - 1, -1, -1):
            r = z[i]
            if i + 1 < n and f.Xs[i + 1] is not None:
                r = r - f.Xs[i + 1] @ x[i + 1]
            x[i] = (be.solve_triangular(f.Ls[i], r, lower=True, trans=1) if f.Ls[i].size else r)
        out = xp.concatenate(x, axis=0)
    else:
        lay = f.band_layout
        out = lay.unpack(blocktri_solve_packed(f, lay.pack(B, be=be), be=be), be=be)
    return out[:, 0] if one else out


def blocktri_whiten_packed(f: BlockTriFactor, b: Any, *, be: Backend | None = None) -> Any:
    """Padded layout only: ``L^-1 b`` for ``b (nb, B, k)`` -- WG ``_btri_forward``
    (matrix.py:112-123) verbatim; the forward-only quadratic form is ``sum(z * z)``."""
    if f.layout != "padded":
        raise ValueError("blocktri_whiten_packed needs the padded layout")
    be = _be(be, f.Ls, b)
    xp = be.xp
    Ls, Xs = f.Ls, f.Xs
    z0 = be.solve_triangular(Ls[0], b[0], lower=True)
    if Ls.shape[0] == 1:
        return z0[None]

    def fwd(zprev: Any, xs: Any) -> tuple[Any, Any]:
        L, X, bi = xs
        z = be.solve_triangular(L, bi - X.T @ zprev, lower=True)
        return z, z

    _, zs = be.scan(fwd, z0, (Ls[1:], Xs, b[1:]))
    return xp.concatenate([z0[None], zs], axis=0)


def blocktri_solve_packed(f: BlockTriFactor, b: Any, *, be: Backend | None = None) -> Any:
    """Padded layout only: ``Lambda^-1 b`` for ``b (nb, B, k)`` -- WG ``_btri_solve``
    (matrix.py:126-145) verbatim."""
    if f.layout != "padded":
        raise ValueError("blocktri_solve_packed needs the padded layout")
    be = _be(be, f.Ls, b)
    xp = be.xp
    Ls, Xs = f.Ls, f.Xs
    z = blocktri_whiten_packed(f, b, be=be)
    xl = be.solve_triangular(Ls[-1], z[-1], lower=True, trans=1)
    if Ls.shape[0] == 1:
        return xl[None]

    def bwd(xnext: Any, xs: Any) -> tuple[Any, Any]:
        L, Xn, zi = xs
        x = be.solve_triangular(L, zi - Xn @ xnext, lower=True, trans=1)
        return x, x

    _, xs = be.scan(bwd, xl, (Ls[:-1], Xs, z[:-1]), reverse=True)
    return xp.concatenate([xs, xl[None]], axis=0)
