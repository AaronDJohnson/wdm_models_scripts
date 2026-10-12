"""The dual-backend shim: one implementation per kernel, NumPy or JAX underneath.

Every kernel in ``wdm`` (and later ``domainia``) is written once against a
:class:`Backend` and never branches on the backend's name for arithmetic.  The
backend supplies the array namespace (``xp``), the FFT namespace used by the
transforms (``fft``: ``numpy.fft`` / ``jax.numpy.fft``), the pinned FFT used by
the exact covariance operator (``fft_pinned``: ``scipy.fft(workers=4)`` on
NumPy, as the WT ``CovOperator`` did), functional updates (``set``,
``add_at``), the linear algebra the oracles used at each call site
(``cholesky(impl=...)``, ``solve_triangular``, ``cho_solve``), control flow
(``scan``, ``while_loop``, ``fori_loop``, ``map``, ``vmap``, ``jit``) and the
modified Bessel functions ``I_p`` (``bessel_i``).

Rules
-----
* ``import wdm`` never imports JAX.  :func:`jax_backend` imports it on first
  use; :func:`pytree_dataclass` registers classes lazily (see there).
* Nothing flips ``jax_enable_x64`` on import.  Call :func:`enable_x64`
  explicitly (scripts do it first); :func:`require_x64` raises if it is off,
  and every route to the JAX backend (:func:`jax_backend`, so also
  :func:`get_backend` on a ``jax.Array`` or with ``backend="jax"``) calls it:
  with x64 off a wdm JAX path raises instead of silently computing in float32.
* ``set`` / ``add_at`` are functional (they return the updated array).  On
  NumPy they copy first **unless** the caller passes ``copy=False``, which a
  kernel may do only for a buffer it allocated itself; then the update is the
  in-place NumPy statement the oracle used (``b[idx] = v``, ``b[idx] += v`` for
  basic indices, ``np.add.at`` for fancy indices), so operation order -- and
  therefore every bit -- is the oracle's.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np

__all__ = [
    "NUMPY",
    "Backend",
    "enable_x64",
    "get_backend",
    "is_tracer",
    "jax_backend",
    "pytree_dataclass",
    "require_x64",
]

#: Terms of the JAX Bessel power series (relative error ~2.5e-14 for |a| <= 8).
BESSEL_SERIES_TERMS = 40


def _is_basic_index(idx: Any) -> bool:
    """True when ``idx`` uses only ints, slices, ``...`` and ``None``."""
    parts = idx if isinstance(idx, tuple) else (idx,)
    return all(
        p is Ellipsis or p is None or isinstance(p, (slice, int, np.integer))
        for p in parts
    )


def _is_full_slice(p: Any) -> bool:
    return isinstance(p, slice) and p == slice(None)


def _check_static_fold(a_shape: tuple[int, ...], idx: Any, v_shape: tuple[int, ...],
                       k0: int, n: int) -> int:
    """Validate ``add_at(a, idx, v, static_fold=(k0, n))``; return the fold length.

    The pad-reshape-sum lowering equals ``a[idx] += v`` only when ``idx`` selects
    the whole of ``a`` except the last axis and puts the fold
    ``(k0 + arange(len)) % n`` on that last axis, which has length ``n``.  So
    every leading component of ``idx`` must be ``:`` or ``...`` (and, without
    ``...``, there must be exactly ``a.ndim - 1`` of them), ``a.shape[-1]`` must
    be ``n``, ``v.shape[-1]`` must be the fold length and the result must keep
    ``a``'s shape.  Anything else raises ``ValueError``: the lowering would
    otherwise broadcast over the wrong axes and return wrong numbers silently.
    """
    parts = idx if isinstance(idx, tuple) else (idx,)
    if not parts:
        raise ValueError("static_fold needs a non-empty idx ending in the fold")
    lead, last = parts[:-1], parts[-1]
    if not all(p is Ellipsis or _is_full_slice(p) for p in lead):
        raise ValueError(
            "static_fold=(k0, n) requires every leading component of idx to be ':' or "
            f"'...'; got {lead!r}")
    n_ellipsis = sum(p is Ellipsis for p in lead)
    n_slices = len(lead) - n_ellipsis
    ndim = len(a_shape)
    if n_ellipsis > 1 or ndim == 0 or n_slices > ndim - 1 or (
            n_ellipsis == 0 and n_slices != ndim - 1):
        raise ValueError(
            "static_fold=(k0, n) requires idx to put the fold on the last axis of a; "
            f"got {len(parts)} index components for a.ndim={ndim}")
    if a_shape[-1] != n:
        raise ValueError(f"static_fold=(k0, n={n}) requires a.shape[-1] == n; got {a_shape}")
    fold = np.asarray(last)
    length = fold.shape[0] if fold.ndim == 1 else -1
    if length < 0 or not np.array_equal(fold, (k0 + np.arange(length)) % n):
        raise ValueError("static_fold=(k0, n) requires idx[-1] == (k0 + arange(len)) % n")
    if len(v_shape) == 0 or v_shape[-1] != length:
        raise ValueError(
            f"static_fold requires v.shape[-1] == len(idx[-1]) == {length}; got v.shape={v_shape}")
    try:
        out_shape = np.broadcast_shapes(tuple(a_shape), (*v_shape[:-1], n))
    except ValueError as err:
        raise ValueError(f"static_fold: v.shape={v_shape} does not fit a.shape={a_shape}") from err
    if out_shape != tuple(a_shape):
        raise ValueError(f"static_fold: v.shape={v_shape} would change a.shape={a_shape}")
    return length


def _tree_index(xs: Any, i: int) -> Any:
    if isinstance(xs, (tuple, list)):
        return type(xs)(_tree_index(x, i) for x in xs)
    return xs[i]


def _tree_len(xs: Any) -> int:
    if isinstance(xs, (tuple, list)):
        return _tree_len(xs[0])
    return len(xs)


def _tree_stack(items: list[Any]) -> Any:
    first = items[0]
    if first is None:
        return None
    if isinstance(first, (tuple, list)):
        return type(first)(_tree_stack([it[j] for it in items]) for j in range(len(first)))
    return np.stack([np.asarray(it) for it in items])


@dataclass(frozen=True)
class Backend:
    """Array backend.  Use :data:`NUMPY` or :func:`jax_backend`; never construct."""

    name: Literal["numpy", "jax"]
    xp: ModuleType
    fft: ModuleType

    # -- arrays ---------------------------------------------------------------
    def asarray(self, a: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def to_host(self, a: Any) -> np.ndarray:
        return np.asarray(a)

    def set(self, a: Any, idx: Any, v: Any, *, copy: bool = True) -> Any:
        raise NotImplementedError

    def add_at(self, a: Any, idx: Any, v: Any, *, static_fold: tuple[int, int] | None = None,
               copy: bool = True) -> Any:
        """Functional ``a[idx] += v`` (repeated indices accumulate).

        ``static_fold=(k0, n)`` declares that ``idx`` is ``(:, ..., :, k)`` or
        ``(..., k)`` with ``k = (k0 + arange(len)) % n`` on the last axis and
        ``a.shape[-1] == n``; JAX then lowers the update to a pad-reshape-sum
        and raises ``ValueError`` if the whole ``idx`` is not that fold, and the
        result keeps ``a``'s dtype under the cast rule of ``a.at[idx].add(v)``.
        NumPy ignores it.
        """
        raise NotImplementedError

    def fft_pinned(self, op: str, x: Any, *, axis: int = -1, n: int | None = None) -> Any:
        raise NotImplementedError

    # -- linear algebra -------------------------------------------------------
    def cholesky(self, A: Any, *, impl: Literal["numpy", "scipy"]) -> tuple[Any, Any]:
        """``(L, ok)`` from the lower triangle of ``A`` only (on both backends)."""
        raise NotImplementedError

    def solve_triangular(self, L: Any, b: Any, *, lower: bool = True, trans: int = 0) -> Any:
        raise NotImplementedError

    def cho_solve(self, L: Any, b: Any) -> Any:
        raise NotImplementedError

    # -- control flow ---------------------------------------------------------
    def scan(self, f: Callable, carry: Any, xs: Any, *, reverse: bool = False) -> tuple[Any, Any]:
        raise NotImplementedError

    def while_loop(self, cond: Callable, body: Callable, init: Any) -> Any:
        raise NotImplementedError

    def fori_loop(self, lo: int, hi: int, body: Callable, init: Any) -> Any:
        raise NotImplementedError

    def map(self, f: Callable, xs: Any) -> Any:
        raise NotImplementedError

    def vmap(self, f: Callable, in_axes: int | Sequence[int | None] = 0) -> Callable:
        raise NotImplementedError

    def jit(self, f: Callable, **kw: Any) -> Callable:
        raise NotImplementedError

    # -- special functions ----------------------------------------------------
    def bessel_i(self, P: int, a: Any) -> Any:
        """``(P, len(a))``: modified Bessel functions ``I_p(a)``, ``p = 0..P-1``."""
        raise NotImplementedError


@dataclass(frozen=True)
class _NumpyBackend(Backend):
    def asarray(self, a: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(a, dtype=dtype)

    def set(self, a: Any, idx: Any, v: Any, *, copy: bool = True) -> np.ndarray:
        b = a.copy() if copy else a
        b[idx] = v
        return b

    def add_at(self, a: Any, idx: Any, v: Any, *, static_fold: tuple[int, int] | None = None,
               copy: bool = True) -> np.ndarray:
        # ``static_fold`` only changes the JAX lowering; NumPy keeps the oracle's
        # statement, whose order is sequential per index.
        b = a.copy() if copy else a
        if _is_basic_index(idx):
            b[idx] += v
        else:
            np.add.at(b, idx, v)
        return b

    def fft_pinned(self, op: str, x: Any, *, axis: int = -1, n: int | None = None) -> Any:
        import scipy.fft

        return getattr(scipy.fft, op)(x, n=n, axis=axis, workers=4)

    def cholesky(self, A: Any, *, impl: Literal["numpy", "scipy"]) -> tuple[np.ndarray, bool]:
        """``(L, ok)``.  ``impl="numpy"``: ``np.linalg.cholesky``; ``impl="scipy"``:
        ``scipy.linalg.cho_factor(lower=True, check_finite=False)`` with the junk
        upper triangle zeroed (``np.tril``; LAPACK never reads it, so solves are
        bit-identical to ``cho_solve`` on the raw factor).  A matrix that is not
        positive definite gives a NaN-filled ``L`` and ``ok=False``."""
        A = np.asarray(A)
        try:
            if impl == "numpy":
                return np.linalg.cholesky(A), True
            if impl == "scipy":
                from scipy.linalg import cho_factor

                c, _ = cho_factor(A, lower=True, check_finite=False)
                return np.tril(c), True
        except np.linalg.LinAlgError:
            return np.full(A.shape, np.nan), False
        raise ValueError(f"impl must be 'numpy' or 'scipy'; got {impl!r}")

    def solve_triangular(self, L: Any, b: Any, *, lower: bool = True, trans: int = 0) -> np.ndarray:
        from scipy.linalg import solve_triangular

        return solve_triangular(L, b, lower=lower, trans=trans, check_finite=False)

    def cho_solve(self, L: Any, b: Any) -> np.ndarray:
        from scipy.linalg import cho_solve

        return cho_solve((L, True), b, check_finite=False)

    def scan(self, f: Callable, carry: Any, xs: Any, *, reverse: bool = False) -> tuple[Any, Any]:
        n = _tree_len(xs)
        order = range(n - 1, -1, -1) if reverse else range(n)
        ys: list[Any] = [None] * n
        for i in order:
            carry, ys[i] = f(carry, _tree_index(xs, i))
        return carry, _tree_stack(ys) if n else None

    def while_loop(self, cond: Callable, body: Callable, init: Any) -> Any:
        val = init
        while bool(cond(val)):
            val = body(val)
        return val

    def fori_loop(self, lo: int, hi: int, body: Callable, init: Any) -> Any:
        val = init
        for i in range(lo, hi):
            val = body(i, val)
        return val

    def map(self, f: Callable, xs: Any) -> Any:
        return _tree_stack([f(_tree_index(xs, i)) for i in range(_tree_len(xs))])

    def vmap(self, f: Callable, in_axes: int | Sequence[int | None] = 0) -> Callable:
        def looped(*args: Any) -> Any:
            axes = (in_axes,) * len(args) if isinstance(in_axes, int) else tuple(in_axes)
            if len(axes) != len(args):
                raise ValueError("in_axes must have one entry per argument")
            mapped = [a for a, ax in zip(args, axes, strict=True) if ax is not None]
            if not mapped:
                raise ValueError("vmap needs at least one mapped argument")
            sizes = {np.shape(a)[ax] for a, ax in zip(args, axes, strict=True) if ax is not None}
            if len(sizes) != 1:
                raise ValueError(f"mapped axes have different sizes: {sorted(sizes)}")
            n = sizes.pop()
            out = []
            for i in range(n):
                call = [a if ax is None else np.take(a, i, axis=ax)
                        for a, ax in zip(args, axes, strict=True)]
                out.append(f(*call))
            return _tree_stack(out)

        return looped

    def jit(self, f: Callable, **kw: Any) -> Callable:
        return f

    def bessel_i(self, P: int, a: Any) -> np.ndarray:
        # Exactly the WT ``TriCovariance._cheb_weights`` expression (before its x2).
        from scipy.special import ive

        a = np.asarray(a)
        p = np.arange(P)[:, None]
        return ive(p, a[None, :]) * np.exp(np.abs(a))[None, :]


@dataclass(frozen=True)
class _JaxBackend(Backend):
    def asarray(self, a: Any, dtype: Any = None) -> Any:
        return self.xp.asarray(a, dtype=dtype)

    def set(self, a: Any, idx: Any, v: Any, *, copy: bool = True) -> Any:
        return a.at[idx].set(v)

    def add_at(self, a: Any, idx: Any, v: Any, *, static_fold: tuple[int, int] | None = None,
               copy: bool = True) -> Any:
        if static_fold is None:
            return a.at[idx].add(v)
        # ``idx`` must be the fold (k0 + arange(len)) % n on the last axis of ``a``
        # with only full slices / an Ellipsis before it, and a.shape[-1] == n
        # (``_check_static_fold`` checks all of it and raises otherwise).  Then
        # pad, reshape to (..., rows, n) and sum -- no scatter, so no GPU atomics.
        from jax import lax

        k0, n = static_fold
        xp = self.xp
        length = _check_static_fold(tuple(a.shape), idx, tuple(np.shape(v)), k0, n)
        # The dtype rule of ``a.at[idx].add(v)`` (jax ``_scatter_impl``): warn
        # when v cannot be cast to a.dtype safely, accumulate in the promoted
        # dtype, then convert the result back to a.dtype.  Without the final
        # convert, ``a + folded`` would silently return float64 for a float32
        # ``a``, complex for a real ``a``, float for an integer ``a``.
        if xp.result_type(v, a.dtype) != a.dtype:
            warnings.warn(
                "static_fold add_at: cannot safely cast v from "
                f"dtype={xp.result_type(v)} to dtype={a.dtype}; the result keeps a's dtype, "
                "as a.at[idx].add(v) does", FutureWarning, stacklevel=2)
        v = xp.asarray(v, dtype=xp.result_type(a, v))
        lead = k0 % n
        total = -(-(lead + length) // n) * n
        pad = [(0, 0)] * (v.ndim - 1) + [(lead, total - lead - length)]
        folded = xp.pad(v, pad).reshape(*v.shape[:-1], total // n, n).sum(axis=-2)
        return lax.convert_element_type(a + folded, a.dtype)

    def fft_pinned(self, op: str, x: Any, *, axis: int = -1, n: int | None = None) -> Any:
        return getattr(self.fft, op)(x, n=n, axis=axis)

    def cholesky(self, A: Any, *, impl: Literal["numpy", "scipy"]) -> tuple[Any, Any]:
        if impl not in ("numpy", "scipy"):
            raise ValueError(f"impl must be 'numpy' or 'scipy'; got {impl!r}")
        xp = self.xp
        # Read only the lower triangle, as LAPACK does for both NumPy impls.
        # JAX's default (symmetrize_input=True) factors (A + A^T)/2 instead, which
        # differs for any A that is not bitwise symmetric (e.g. assembled
        # lower-only); for a bitwise-symmetric A the two are identical.
        L = xp.linalg.cholesky(A, symmetrize_input=False)
        ok = xp.all(xp.isfinite(xp.diagonal(L, axis1=-2, axis2=-1)))
        return L, ok

    def solve_triangular(self, L: Any, b: Any, *, lower: bool = True, trans: int = 0) -> Any:
        from jax.scipy.linalg import solve_triangular

        return solve_triangular(L, b, lower=lower, trans=trans)

    def cho_solve(self, L: Any, b: Any) -> Any:
        from jax.scipy.linalg import cho_solve

        return cho_solve((L, True), b)

    def scan(self, f: Callable, carry: Any, xs: Any, *, reverse: bool = False) -> tuple[Any, Any]:
        from jax import lax

        return lax.scan(f, carry, xs, reverse=reverse)

    def while_loop(self, cond: Callable, body: Callable, init: Any) -> Any:
        from jax import lax

        return lax.while_loop(cond, body, init)

    def fori_loop(self, lo: int, hi: int, body: Callable, init: Any) -> Any:
        from jax import lax

        return lax.fori_loop(lo, hi, body, init)

    def map(self, f: Callable, xs: Any) -> Any:
        from jax import lax

        return lax.map(f, xs)

    def vmap(self, f: Callable, in_axes: int | Sequence[int | None] = 0) -> Callable:
        import jax

        return jax.vmap(f, in_axes=in_axes if isinstance(in_axes, int) else tuple(in_axes))

    def jit(self, f: Callable, **kw: Any) -> Callable:
        import jax

        return jax.jit(f, **kw)

    def bessel_i(self, P: int, a: Any) -> Any:
        # Verbatim ``wdmtoysgpu.jaxlike.bessel_i`` (40-term power series).  Every
        # term is positive, so there is no cancellation; 40 terms cover |a| <= 8.
        from jax import lax

        jnp = self.xp
        a = jnp.asarray(a)
        half = 0.5 * jnp.abs(a)
        x = half * half
        p = jnp.arange(P, dtype=a.dtype)
        q = jnp.arange(1, P, dtype=a.dtype)
        ratios = jnp.concatenate([jnp.ones((1, a.shape[0]), a.dtype),
                                  half[None, :] / q[:, None]], axis=0)
        t0 = jnp.cumprod(ratios, axis=0)                                  # (P, K)

        def body(m: Any, carry: Any) -> Any:
            t, acc = carry
            t = t * x[None, :] / ((m + 1.0) * (m + 1.0 + p))[:, None]
            return t, acc + t

        _, acc = lax.fori_loop(0, BESSEL_SERIES_TERMS, body, (t0, t0))
        parity = jnp.where(p % 2 == 1, -1.0, 1.0)
        return jnp.where(a[None, :] < 0.0, parity[:, None], 1.0) * acc


#: The NumPy backend (the default whenever no JAX array is involved).
NUMPY: Backend = _NumpyBackend("numpy", np, np.fft)

_PENDING_PYTREES: list[tuple[type, tuple[str, ...], tuple[str, ...]]] = []
_REGISTERED: set[type] = set()


def _flush_pytrees() -> None:
    """Register every :func:`pytree_dataclass` class with JAX (JAX is imported)."""
    import jax

    while _PENDING_PYTREES:
        cls, data, meta = _PENDING_PYTREES.pop(0)
        if cls in _REGISTERED:
            continue
        jax.tree_util.register_dataclass(cls, data_fields=list(data), meta_fields=list(meta))
        _REGISTERED.add(cls)


@functools.cache
def _make_jax_backend() -> Backend:
    import jax.numpy as jnp

    return _JaxBackend("jax", jnp, jnp.fft)


def jax_backend() -> Backend:
    """The JAX backend.  Imports JAX (never at ``import wdm`` time), checks
    :func:`require_x64` and registers any pending :func:`pytree_dataclass` classes.

    Raises ``RuntimeError`` when ``jax_enable_x64`` is off: wdm needs float64,
    and with x64 off JAX would silently compute every kernel in float32
    (``forward_time``: relative error ~2e-7, no warning).
    """
    require_x64()
    be = _make_jax_backend()
    _flush_pytrees()
    return be


def is_tracer(x: Any) -> bool:
    """True when ``x`` is a JAX tracer (host-side validation must be skipped)."""
    if "jax" not in sys.modules:
        return False
    import jax

    return isinstance(x, jax.core.Tracer)


def get_backend(*arrays: Any, backend: str | Backend | None = None) -> Backend:
    """An explicit ``backend`` wins.  Otherwise JAX iff ``"jax"`` is already in
    ``sys.modules`` and any argument is a ``jax.Array`` (tracers included).

    The JAX branch goes through :func:`jax_backend`, so it raises
    ``RuntimeError`` (:func:`require_x64`) when ``jax_enable_x64`` is off.
    """
    if isinstance(backend, Backend):
        return backend
    if backend == "numpy":
        return NUMPY
    if backend == "jax":
        return jax_backend()
    if backend is not None:
        raise ValueError(f"backend must be 'numpy', 'jax' or a Backend; got {backend!r}")
    if "jax" in sys.modules:
        import jax

        if any(isinstance(a, jax.Array) for a in arrays):
            return jax_backend()
    return NUMPY


def require_x64() -> None:
    """Raise ``RuntimeError`` unless ``jax_enable_x64`` is on.

    Called by :func:`jax_backend` (hence by every wdm JAX path); callers that
    build JAX constants without going through wdm should call it themselves.
    """
    import jax

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "wdm needs float64 under JAX: call wdm.backend.enable_x64() (or set "
            "JAX_ENABLE_X64=true) before building any JAX constants"
        )


def enable_x64() -> None:
    """Turn on ``jax_enable_x64``.  Explicit; never called on import."""
    import jax

    jax.config.update("jax_enable_x64", True)
    _flush_pytrees()


def pytree_dataclass(cls: type | None = None, *, data: Sequence[str] = (),
                     meta: Sequence[str] = ()) -> Any:
    """Make ``cls`` a frozen dataclass and a JAX pytree with leaves ``data`` and
    static fields ``meta`` (together exactly the dataclass fields).

    Registration is lazy so that ``import wdm`` never imports JAX: if JAX is
    already imported the class is registered immediately, otherwise it is queued
    and registered by the first :func:`jax_backend` / :func:`enable_x64` /
    :func:`get_backend` call that returns the JAX backend.
    """

    def wrap(c: type) -> type:
        if not dataclasses.is_dataclass(c):
            c = dataclass(frozen=True)(c)
        names = {f.name for f in dataclasses.fields(c)}
        declared = set(data) | set(meta)
        if declared != names or set(data) & set(meta):
            raise ValueError(
                f"{c.__name__}: data {tuple(data)} and meta {tuple(meta)} must partition "
                f"the fields {tuple(sorted(names))}"
            )
        _PENDING_PYTREES.append((c, tuple(data), tuple(meta)))
        if "jax" in sys.modules:
            _flush_pytrees()
        return c

    return wrap if cls is None else wrap(cls)
