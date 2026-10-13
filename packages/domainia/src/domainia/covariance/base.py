r"""The covariance protocol and the factor pytrees shared by every covariance class.

A covariance is built once from host constants and factorised once per noise state:

.. code-block:: python

    cov = DiagonalCovariance(variance)
    F = cov.factor(nu)                    # NumPy: eager, oracle operation order
    F = cov.factor(jnp.asarray(nu))       # JAX: jax.jit(cov._factor)(constants, nu)

``F`` is a :class:`Factor`: ``logdet`` (:math:`\ln|C|`), ``ok`` (False when ``C`` is not
positive definite: the likelihood is then :math:`-\infty`), ``solve(b)`` (:math:`C^{-1}b`),
``whiten(b)`` (:math:`L^{-1}b` with :math:`|L^{-1}b|^2 = b^{\mathsf T}C^{-1}b`),
``quad(b, form=...)`` and ``n_u`` (directions a marginal removed; 0 for a covariance).

**A factor holds per-state arrays only** (spec section 3.3, S4 deferral (b)).  Every array
that depends on the grid alone -- edge/interior positions, ``W_p``, the nuisance ``U``,
band layouts, the Chebyshev basis -- stays in the covariance's constants ``C`` and is passed
to the factor's methods: ``F.solve(b, C=C)``, ``F.whiten(b, C=C)``, ``F.quad(b, C=C)``, with
``C = cov.constants(be)`` (a combinator passes its child's sub-dict, e.g. ``C["interior"]``).
So a cached or stacked noise state carries no constants (the LRU of spec section 3.5 holds
16 states on a GPU).  :meth:`Covariance.factor` attaches ``C`` to the factor it returns
(outside every pytree: not a leaf, dropped by ``tree_map``/``jit``), so ``cov.factor(nu).quad(b)``
works without it; factors produced inside a jitted function or unstacked from a cache get
``C`` explicitly (``GaussianLikelihood`` passes ``parts.covs[c]``).

Structure / Constants / Factor (spec §3.3).  The configuration a subclass keeps on itself
is static (ints, tuples, strings).  Every array a factorisation reads comes from
``constants(be)`` and reaches the jitted function as an ARGUMENT, never closed over:
``host_constants()`` builds it once in host NumPy float64 (spec §0 rule 4) and
``constants(be)`` caches it per backend (JAX: ``require_x64()``, then one
``jax.device_put``).  Choosing jit or eager is dispatch, not arithmetic: the arithmetic is
one code path per named strategy.

Quadratic-form orders (each is the operation order of an oracle, kept per call site):

* ``"solve"``: ``b @ solve(b)`` -- WT ``marginal_lnlike`` (``float(y @ Ciy)``);
* ``"whiten"``: ``w @ w`` with ``w = whiten(b)`` -- GB ``_dense_quad_logdet``, WG forward-only;
* ``"elementwise"``: ``sum(b**2 / v)`` -- GB diagonal models (:class:`DiagFactor` only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from wdm.backend import NUMPY, Backend, get_backend, is_tracer, pytree_dataclass, require_x64

__all__ = [
    "CholFactor",
    "Covariance",
    "DiagFactor",
    "Evaluator",
    "Factor",
    "QuadForm",
    "as_evaluator",
    "attach_constants",
    "factor_constants",
    "quad_form",
]

QuadForm = Literal["solve", "whiten", "elementwise"]


@runtime_checkable
class Factor(Protocol):
    """What a likelihood reads from a factorised covariance.

    ``quad`` is universal; ``solve`` and ``whiten`` are not: ``MarginalFactor`` and
    ``ProjectedFactor`` (``domainia.nuisance``) expose ``quad`` and ``logdet`` only, and
    ``OperatorFactor`` / ``PaddedFactor`` have no whitening (``form="solve"``).  Those
    methods raise ``NotImplementedError``; ``tests/test_dom_cov_api.py`` pins the table.
    """

    logdet: Any
    ok: Any

    @property
    def n_u(self) -> int: ...

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any: ...

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any: ...

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any: ...


class Evaluator(Protocol):
    """A parameter-dependent array (variances, blocks, a matrix) built from host constants.

    ``host_constants()`` is called once (host NumPy float64); ``evaluate(C, nu, be)`` is
    pure in ``(C, nu)`` and jit-able with ``C`` passed as an argument.
    """

    def host_constants(self) -> Any: ...

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any: ...


@dataclass(frozen=True, eq=False)
class _CallableEvaluator:
    """A plain ``fn(nu, be) -> array`` as an :class:`Evaluator` without constants.

    Arrays ``fn`` closes over become program constants under ``jax.jit``; production code
    passes real evaluators, tests may pass callables."""

    fn: Callable[[Any, Backend], Any]

    def host_constants(self) -> tuple[()]:
        return ()

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        return self.fn(nu, be)


@dataclass(frozen=True, eq=False)
class _FixedEvaluator:
    """A parameter-independent array (a constant covariance or variance)."""

    value: Any

    def host_constants(self) -> np.ndarray:
        return np.asarray(self.value)

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        return C


@dataclass(frozen=True, eq=False)
class _PytreeCallableEvaluator:
    """A self-contained pytree callable ``obj(nu, *, be, **kwargs)`` -- the ``domainia.variance``
    classes (D-B).  The object is its own constants: ``host_constants`` returns it, so JAX
    device-puts its array leaves once and the jitted factorisation receives it as an
    argument (never closed over)."""

    obj: Any
    kwargs: tuple[tuple[str, Any], ...] = ()

    def host_constants(self) -> Any:
        return self.obj

    def evaluate(self, C: Any, nu: Any, be: Backend) -> Any:
        return C(nu, be=be, **dict(self.kwargs))


def as_evaluator(obj: Any, **kwargs: Any) -> Evaluator:
    """``obj`` itself when it has ``host_constants``/``evaluate``; a ``domainia.variance``-style
    pytree callable (callable with an ``on(be)`` method: evaluated as ``obj(nu, be=be,
    **kwargs)``, itself the constants); a callable ``fn(nu, be)``; or a fixed array.
    ``kwargs`` (``channel``) apply to the pytree callables only: anything else given
    ``kwargs`` raises ``ValueError`` (an ``Evaluator`` too -- it would drop them silently)."""
    is_evaluator = hasattr(obj, "evaluate") and hasattr(obj, "host_constants")
    if not is_evaluator and callable(obj) and hasattr(obj, "on"):
        return _PytreeCallableEvaluator(obj, tuple(sorted(kwargs.items())))
    if kwargs:
        raise ValueError(f"{sorted(kwargs)} apply to pytree callables (domainia.variance) only")
    if is_evaluator:
        return obj
    if callable(obj):
        return _CallableEvaluator(obj)
    return _FixedEvaluator(obj)


def quad_form(factor: Any, b: Any, form: QuadForm, be: Backend, C: Any = None) -> Any:
    """``b^T C^-1 b`` in the operation order ``form`` (module docstring); ``C`` the factor's
    constants (passed through to ``solve`` / ``whiten``)."""
    if form == "solve":
        return b @ factor.solve(b, C=C, be=be)
    if form == "whiten":
        w = factor.whiten(b, C=C, be=be)
        return w @ w
    raise ValueError(f"{type(factor).__name__} has no quadratic form {form!r}")


def attach_constants(F: Any, C: Any) -> Any:
    """``F`` with ``C`` attached as a plain attribute (never a pytree leaf), so its methods
    run without an explicit ``C=`` (:meth:`Covariance.factor` calls this)."""
    object.__setattr__(F, "_C", C)
    return F


def factor_constants(F: Any, C: Any, what: str) -> Any:
    """The constants a factor method reads: ``C`` when given, else the attached ones; a
    clear error when neither exists (a factor from a jitted function or an unstacked cache
    entry has none attached)."""
    if C is None:
        C = F.__dict__.get("_C")
    if C is None:
        raise ValueError(f"{type(F).__name__}.{what} needs its covariance's constants: pass "
                         "C=cov.constants(be) (cov.factor(nu) attaches them)")
    return C


def concrete_failure(ok: Any) -> bool:
    """True when ``ok`` is a concrete False: an eager factorisation failed (a ragged block, a
    NumPy Cholesky) and the factor must not be applied -- ``domainia.linalg`` raises on a
    failed ragged factor's solve, where WT raised ``LinAlgError`` before anything else ran.
    Combinators then return a failed factor (``ok = False``, ``logdet = nan``) without
    their own algebra.  Never True under ``jit``: a traced ``ok`` lets NaN propagate."""
    return not is_tracer(ok) and not bool(ok)


@pytree_dataclass(data=("v", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class DiagFactor:
    """``C = diag(v)``: WT ``Solver(diag=...)``, GB diagonal models."""

    v: Any
    logdet: Any
    ok: Any

    n_u = 0

    @classmethod
    def build(cls, v: Any, be: Backend) -> DiagFactor:
        xp = be.xp
        v = be.asarray(v)
        return cls(v, xp.sum(xp.log(v)), xp.all(v > 0) & xp.all(xp.isfinite(v)))

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        return b / (self.v if b.ndim == 1 else self.v[:, None])

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        s = be.xp.sqrt(self.v)
        return b / (s if b.ndim == 1 else s[:, None])

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        be = be or get_backend(self.v, b)
        if form == "elementwise":
            return be.xp.sum(b ** 2 / self.v)
        return quad_form(self, b, form, be)


@pytree_dataclass(data=("L", "logdet", "ok"), meta=("impl",))
@dataclass(frozen=True, eq=False)
class CholFactor:
    """Dense real Cholesky ``C = L L^T``.

    ``impl="scipy"`` (WT ``Solver(dense)``, ``Schur``, ``PaddedMarginal``, the marginal's
    ``H``): ``scipy.linalg.cho_factor`` and ``solve = cho_solve``.  ``impl="numpy"`` (GB):
    ``np.linalg.cholesky`` and ``solve`` = forward then backward triangular solve (GB
    ``MatrixCache.state``).  ``whiten`` is the forward solve for both.
    """

    L: Any
    logdet: Any
    ok: Any
    impl: str

    n_u = 0

    @classmethod
    def build(cls, S: Any, *, impl: Literal["numpy", "scipy"], be: Backend) -> CholFactor:
        L, ok = be.cholesky(S, impl=impl)
        xp = be.xp
        return cls(L, 2.0 * xp.sum(xp.log(xp.diagonal(L))), ok, impl)

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        if self.impl == "scipy":
            return be.cho_solve(self.L, b)
        u = be.solve_triangular(self.L, b, lower=True)
        return be.solve_triangular(self.L, u, lower=True, trans=1)

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = be or get_backend(self.L, b)
        return be.solve_triangular(self.L, b, lower=True)

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "solve", be: Backend | None = None) -> Any:
        return quad_form(self, b, form, be or get_backend(self.L, b))


class Covariance:
    """Base class: constants caching and the eager / jitted factorisation.

    Subclasses set ``n`` (the dimension) and implement ``host_constants()`` (a pytree of
    host arrays; a dict when the class has children) and ``_factor(C, nu, be)`` (pure;
    reads arrays only from ``C``).  A combinator lists the covariances it wraps in
    ``children()``; their constants appear in ``C`` under the same keys and keep their
    own caches.  Covariances with a dense form implement ``_matrix(C, nu, be)``;
    :meth:`dense` is its host value.  ``jittable = False`` marks a strategy that runs
    eagerly on both backends (the ragged layout, ``on_fail="raise"``).
    """

    n: int
    jittable: bool = True
    n_u: int = 0

    def host_constants(self) -> Any:
        return {}

    def children(self) -> dict[str, Covariance]:
        return {}

    def state_key(self, nu: Any) -> Any:
        """A hashable static key of the factorisation at the concrete ``nu`` (``None``: no
        host-side dispatch).  The default forwards the children's keys at the same ``nu``;
        ``SchurEdge`` with adaptive rows adds the Chebyshev order, ``ChannelStack`` splits
        ``nu`` per channel.  A likelihood groups cache misses by it (spec §3.5) and never
        traces a factorisation whose key is not ``None`` inside its own ``jit``."""
        keys = tuple(c.state_key(nu) for c in self.children().values())
        return None if all(k is None for k in keys) else keys

    def _factor(self, C: Any, nu: Any, be: Backend) -> Any:
        raise NotImplementedError

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no dense assembly")

    def constants(self, be: Backend | None = None) -> Any:
        """The constants on ``be``: built once in host NumPy, cached per backend (JAX:
        ``require_x64()``, then one ``jax.device_put``).  They are cached for the object's
        lifetime: build a new covariance for a new configuration, never mutate one."""
        be = be or NUMPY
        cache = self.__dict__.setdefault("_constants_cache", {})
        if be.name not in cache:
            own = self.host_constants()
            if be.name != "numpy":
                require_x64()
                import jax

                own = jax.device_put(own)
            kids = {k: c.constants(be) for k, c in self.children().items()}
            cache[be.name] = {**own, **kids} if kids else own
        return cache[be.name]

    def factor(self, nu: Any, be: Backend | None = None, *, jit: bool | None = None) -> Any:
        """Factorise at the noise state ``nu`` (``A_n`` is never inside ``nu``).

        JAX and ``jittable``: one compiled ``jax.jit(lambda C, nu: self._factor(C, nu, be))``
        per covariance, the constants passed as its argument.  A covariance whose
        :meth:`state_key` is not ``None`` (adaptive rows below a combinator: the static
        order needs a concrete ``nu``) runs eagerly unless ``jit`` is given -- the reference
        path; ``SchurEdge.factor`` itself dispatches per order.
        """
        be = be or get_backend(nu)
        C = self.constants(be)
        nu = be.asarray(nu)
        if jit is None:
            jit = self.jittable and self.state_key(np.asarray(nu, dtype=float)) is None
        if be.name == "jax" and bool(jit):
            fns = self.__dict__.setdefault("_jit_cache", {})
            if "factor" not in fns:
                fns["factor"] = be.jit(lambda C, nu: self._factor(C, nu, be))
            return attach_constants(fns["factor"](C, nu), C)
        return attach_constants(self._factor(C, nu, be), C)

    def dense(self, nu: Any) -> np.ndarray:
        """The covariance matrix on the host (tests, dense oracles)."""
        return np.asarray(self._matrix(self.constants(NUMPY), np.asarray(nu, float), NUMPY))
