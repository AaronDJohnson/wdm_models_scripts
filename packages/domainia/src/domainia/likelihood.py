r"""Gaussian likelihoods assembled from data, a map, a covariance and a density
(spec sections 3.3 and 4.7).

``GaussianLikelihood`` evaluates rows ``theta`` (shape ``(R, ndim)``) of

.. math::  \ln L(\theta) = \text{density}\bigl(r^T C(\nu)^{-1} r,\ \ln|C(\nu)|\bigr),
           \qquad r = M d - M s(\theta_{\rm source}),

with ``M`` the map (``map.apply``), ``s`` the signal and ``nu`` the noise slice of
``theta``.  Two strategies (spec section 4.7):

* ``"residual"`` -- the NumPy oracles: the residual cells ``r = y_data - M s`` (in
  that order: WT ``MatrixWdm.lnlike``, ``wdm_toys/matrix.py:851-853``; GB
  ``matrix.make_lnlike``, ``galactic_binary/matrix.py:911-925``) go through
  ``F.quad(r, form=quad_form)``;
* ``"expanded"`` -- the GB sampler caches: the noise state holds ``q0 = d^T C^-1 d``
  and ``y = C^-1 d`` and a row costs ``q0 - 2 h.y + |L^-1 h|^2``
  (``MatrixCache``, ``EdgeSchurCache``, ``fullcache.FullCache``;
  ``galactic_binary/matrix.py:978-994, 1043-1073``, ``fullcache.py:128-144``).

**The covariance protocol** (section D-C, ``domainia.covariance.base``; this module calls
nothing else): a covariance is static structure with ``n``, ``n_u`` (directions a flat
marginal removes; ``n_eff = n - n_u``), ``jittable``, ``constants(be)`` (host-built
arrays, device-put once), the pure ``_factor(C, nu, be)``, the user-facing
``factor(nu, be)`` (its own jit/eager dispatch) and ``state_key(nu)`` (``None`` unless
the factorisation dispatches on a static key, e.g. ``SchurEdge``'s adaptive Chebyshev
order).  A factor ``F`` has ``ok``, ``logdet``, ``n_u``, ``quad(b, form=...)``,
``whiten(b)`` (``L^-1 b``) and ``solve(b)`` (``C^-1 b``; for the GB factors --
``CholFactor(impl="numpy")``, ``HermitianFactor`` -- ``solve`` is ``L^-H (L^-1 b)``, the
``MatrixCache``/``FullCache`` order).  Optional ``F.expanded_terms(d) -> (q0, y)`` and
``F.signal_quad(h)`` replace the whiten-based expanded order where an oracle differs
(``EdgeSchurCache``).

**JAX** (spec section 3.3): the map and the signal must be JAX pytrees
(``wdm.backend.pytree_dataclass``); the covariances stay static and their constants
travel in :attr:`GaussianLikelihood.parts` (``parts.covs[c] = cov.constants(be)``), a
pytree passed to ``jax.jit`` as an argument and never closed over.  The data are mapped
once on the host in NumPy float64 (host-constant rule).  When every covariance is
``jittable`` and has no static key, the whole noise state is one compiled program
(``_noise_state``); otherwise each covariance's own ``factor`` runs (its staged rows, or
eagerly for the ``reference`` strategies) and only the state's remaining arithmetic is
compiled.  Building a JAX likelihood calls ``wdm.backend.jax_backend()``, which runs
``require_x64()``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from wdm.backend import NUMPY, Backend, get_backend, pytree_dataclass

from .covariance.base import concrete_failure
from .covariance.stack import ChannelStack
from .density import ComplexGaussian, RealGaussian

__all__ = ["COUNTERS", "STRATEGIES", "FourierWhittle", "GaussianLikelihood", "NoiseState",
           "Params", "StaticSignal"]

#: What every likelihood counts (``chain_stats.COUNTERS``, ``galactic_binary/fullcache.py:189``).
COUNTERS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations")
STRATEGIES = ("residual", "expanded")
QUAD_FORMS = ("solve", "whiten", "elementwise")


# ---------------------------------------------------------------------------
# parameters and small pytrees
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Params:
    """Where each entry of a row goes: ``source`` indices feed the signal, ``noise``
    indices the covariance (in that order); any other index is read by name (the
    density's amplitude)."""

    names: tuple[str, ...]
    source: tuple[int, ...]
    noise: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        object.__setattr__(self, "source", tuple(int(i) for i in self.source))
        object.__setattr__(self, "noise", tuple(int(i) for i in self.noise))
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"parameter names must be unique; got {self.names}")
        used = self.source + self.noise
        if any(not 0 <= i < len(self.names) for i in used):
            raise ValueError(f"indices {used} out of range for {len(self.names)} parameters")
        if len(set(used)) != len(used):
            raise ValueError("source and noise indices must be distinct")

    @property
    def ndim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no parameter {name!r} in {self.names}") from None


@pytree_dataclass(data=(), meta=("fn",))
class StaticSignal:
    """A plain function ``source -> signal`` as a pytree with no leaves (the function is
    static), so it can sit in :attr:`GaussianLikelihood.parts` under ``jax.jit``."""

    fn: Callable

    def __call__(self, source: Any) -> Any:
        return self.fn(source)


@pytree_dataclass(data=("factors", "y", "q0", "logdet", "ok"), meta=())
class NoiseState:
    """Everything that depends on the noise slice alone, one entry per channel.
    ``y``/``q0`` are set for the expanded strategy only; ``logdet`` holds
    ``logdet + n_eff ln 2 pi`` there for a :class:`RealGaussian` (the GB order)."""

    factors: tuple
    y: tuple | None
    q0: tuple | None
    logdet: tuple
    ok: Any


@pytree_dataclass(data=("y_data", "map", "covs", "signal"), meta=())
class _Parts:
    y_data: Any
    map: Any
    covs: tuple          # cov.constants(be) per channel
    signal: Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _inner(a: Any, b: Any, xp: Any) -> Any:
    """``a.b`` for real vectors (``xp.dot``, GB ``MatrixCache``) or ``Re(a^H b)`` for complex
    ones (``xp.real(xp.vdot(...))``, GB ``FullCache``); the choice is by dtype, not backend."""
    if xp.iscomplexobj(a) or xp.iscomplexobj(b):
        return xp.real(xp.vdot(a, b))
    return xp.dot(a, b)


def _expanded_terms(F: Any, d: Any, xp: Any) -> tuple[Any, Any]:
    """``(q0, y)`` = ``(d^T C^-1 d, C^-1 d)``: ``u = L^-1 d``, ``q0 = u.u``, ``y = F.solve(d)``
    (``L^-H L^-1 d`` for the GB factors: ``MatrixCache.state``, ``fullcache.noise_state``)
    unless the factor brings its own."""
    own = getattr(F, "expanded_terms", None)
    if own is not None:
        return own(d)
    u = F.whiten(d)
    return _inner(u, u, xp), F.solve(d)


def _signal_quad(F: Any, h: Any, xp: Any) -> Any:
    """``|L^-1 h|^2`` in the factor's expanded order (``t = L^-1 h``, ``t.t``)."""
    own = getattr(F, "signal_quad", None)
    if own is not None:
        return own(h)
    t = F.whiten(h)
    return _inner(t, t, xp)


def _rows(Z: Any, ndim: int) -> np.ndarray:
    X = np.asarray(Z)
    if np.iscomplexobj(X):
        raise ValueError("rows must be real")
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.ndim != 2 or X.shape[1] != ndim:
        raise ValueError(f"expected rows of shape (R, {ndim}); got {np.shape(Z)}")
    return X


def _check_pytree(parts: Any, what: str) -> None:
    """On JAX every leaf of ``parts`` must be an array: a map or signal that is not a
    registered pytree would otherwise be closed over or rejected by ``jit``."""
    import jax

    for leaf in jax.tree_util.tree_leaves(parts):
        if not isinstance(leaf, (jax.Array, np.ndarray, np.generic, bool, int, float, complex)):
            raise TypeError(f"{what}: leaf of type {type(leaf).__name__} is not an array; "
                            "maps and signals must be JAX pytrees "
                            "(wdm.backend.pytree_dataclass) on the JAX backend")


def _resolve_backend(values: Any, backend: str | Backend | None) -> Backend:
    be = get_backend(values, backend=backend)
    if be.name == "jax":
        from wdm.backend import jax_backend

        be = jax_backend()                 # require_x64() + pending pytree registrations
    return be


class _CountersMixin:
    """The counters of ``chain_stats.COUNTERS`` plus ``cache_hits`` and ``last_failed``."""

    def reset_counters(self) -> None:
        for k in COUNTERS:
            setattr(self, k, 0)
        self.cache_hits = 0
        self.last_failed = np.zeros(0, dtype=bool)

    def counters(self) -> dict[str, int]:
        out = {k: int(getattr(self, k)) for k in COUNTERS}
        out["cache_hits"] = int(self.cache_hits)
        return out


# ---------------------------------------------------------------------------
# the Gaussian likelihood
# ---------------------------------------------------------------------------


class GaussianLikelihood(_CountersMixin):
    """``(R, ndim) -> (R,)`` log-likelihoods: ``-inf`` where the noise factorisation is not
    positive definite (``state.ok`` False; ``last_failed`` flags those rows), where
    ``row_ok`` rejects the row (the GB ``source_ok``), or where the value is not finite.

    ``data``: a ``TimeSeries`` / ``FourierBins`` / ``WDMCells`` (anything with ``.values``)
    or an array.  ``cov``: one ``domainia.covariance`` covariance, or a ``ChannelStack``
    whose channel ``c`` reads row ``c`` of the mapped data and the noise entries
    ``[c k, (c + 1) k)`` (``k = params_per_channel``; lnL summed in channel order from
    ``0.0``, as the GB loops).  ``nuisance``: cell-domain directions ``U``
    (``(n_cells, n_u)``); the covariance becomes ``domainia.nuisance.FlatMarginal(cov, U)``
    (WT ``form="solve"``; build ``FlatMarginal(cov, U, form="whiten")`` yourself for the WG
    order).  ``quad_form``: the residual quadratic form's order, passed to ``F.quad``
    (``None``: the factor's own default -- ``"solve"``, WT; GB diagonal models
    ``"elementwise"``, GB dense models ``"whiten"``; spec §0 rule 1, section D-C note 3).
    ``noise_clamp``: ``{name: c}`` clips that noise entry to ``[-c, c]`` before the
    factorisation and in the cache key (WG ``ALPHA_D_CLAMP``,
    ``gpu_sinusoid/wdmtoysgpu/matrix.py:637``).  ``row_ok``: ``theta -> bool`` (jit-able).
    ``backend``: ``"numpy"``/``"jax"``/a ``Backend``; default from ``data``.
    """

    def __init__(self, data: Any, map: Any, cov: Any, density: RealGaussian | ComplexGaussian, *,
                 signal: Callable, params: Params, nuisance: Any = None,
                 strategy: Literal["residual", "expanded"] = "residual",
                 quad_form: Literal["solve", "whiten", "elementwise"] | None = None,
                 noise_clamp: Mapping[str, float] | None = None,
                 row_ok: Callable | None = None, backend: str | Backend | None = None) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}; got {strategy!r}")
        if quad_form is not None and quad_form not in QUAD_FORMS:
            raise ValueError(f"quad_form must be one of {QUAD_FORMS} or None; got {quad_form!r}")
        if not isinstance(density, (RealGaussian, ComplexGaussian)):
            raise TypeError(f"density must be RealGaussian or ComplexGaussian; got {type(density)}")
        if not isinstance(params, Params):
            raise TypeError("params must be a domainia.likelihood.Params")
        if strategy == "expanded" and isinstance(density, RealGaussian) and density.amplitude:
            raise ValueError("strategy='expanded' has no amplitude form; use 'residual'")
        if nuisance is not None:
            if isinstance(cov, ChannelStack):
                raise ValueError("nuisance applies to a single-channel covariance")
            from .nuisance import FlatMarginal

            cov = FlatMarginal(cov, nuisance)
        raw = getattr(data, "values", data)
        self.be = be = _resolve_backend(raw, backend)
        values = np.asarray(raw)
        self.params, self.density, self.strategy, self.quad_form = params, density, strategy, quad_form
        if isinstance(cov, ChannelStack):
            covs, k = tuple(cov.covs), int(cov.k)
            slices = tuple(tuple(range(c * k, (c + 1) * k)) for c in range(len(covs)))
            if k * len(covs) != len(params.noise):
                raise ValueError(f"ChannelStack reads {k * len(covs)} noise entries; params has "
                                 f"{len(params.noise)}")
            self.stacked = True
        else:
            covs, slices = (cov,), (tuple(range(len(params.noise))),)
            self.stacked = False
        self.covs = covs
        self.n_channels = len(covs)
        self.n_eff = tuple(int(c.n) - int(c.n_u) for c in covs)
        self._slices = tuple(np.asarray(s, dtype=np.int64) for s in slices)
        self._src = np.asarray(params.source, dtype=np.int64)
        self._noise = np.asarray(params.noise, dtype=np.int64)
        self._amp = (params.index(density.amplitude)
                     if isinstance(density, RealGaussian) and density.amplitude else None)
        clamp = dict(noise_clamp or {})
        for name in clamp:                 # else tuple.index's "x not in tuple"
            if params.index(name) not in params.noise:
                raise ValueError(f"noise_clamp clips noise parameters only; {name!r} is not one")
        self._clamp = tuple((params.noise.index(params.index(name)), float(c))
                            for name, c in sorted(clamp.items()))
        if any(not c > 0 for _, c in self._clamp):
            raise ValueError("noise_clamp values must be positive")
        self.row_ok = row_ok
        #: every covariance's arithmetic is traceable (its factors are array pytrees)
        self.jittable = all(bool(c.jittable) for c in covs)
        #: ... and none dispatches on a static key: the whole noise state is one program
        probe = np.zeros(len(params.noise))
        self.traceable = self.jittable and all(c.state_key(probe[s]) is None
                                               for c, s in zip(covs, self._slices, strict=True))
        # host-constant rule: the data are mapped once in NumPy float64
        y_data = np.asarray(map.apply(np.asarray(values), be=NUMPY))
        if self.stacked and (y_data.ndim < 2 or y_data.shape[0] != self.n_channels):
            raise ValueError(f"stacked data must lead with {self.n_channels} channels; got {y_data.shape}")
        consts = tuple(c.constants(be) for c in covs)       # device-put once per covariance
        if be.name == "jax":
            import jax

            parts = _Parts(be.asarray(y_data), map, consts, signal)
            _check_pytree(parts, "GaussianLikelihood")
            self.parts = jax.device_put(parts)
            self._state_fn = jax.jit(self._noise_state) if self.traceable else None
            self._finish_fn = jax.jit(self._finish) if self.jittable else self._finish
            self._eval_fn = jax.jit(self._evaluate) if self.jittable else self._evaluate
        else:
            self.parts = _Parts(y_data, map, consts, signal)
            self._state_fn, self._finish_fn, self._eval_fn = self._noise_state, self._finish, self._evaluate
        self.reset_counters()

    # -- parameters ---------------------------------------------------------
    @property
    def ndim(self) -> int:
        return self.params.ndim

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.params.names

    def noise_slice(self, theta: Any) -> Any:
        """``theta[noise]`` with the clamps applied (traceable)."""
        be = self.be if not isinstance(theta, np.ndarray) else NUMPY
        nu = theta[self._noise]
        for pos, c in self._clamp:
            nu = be.set(nu, pos, be.xp.clip(nu[pos], -c, c))
        return nu

    def noise_slice_host(self, theta: Any) -> np.ndarray:
        """The same on the host, float64 (the cache key's source)."""
        return np.asarray(self.noise_slice(np.asarray(theta, dtype=np.float64)), dtype=np.float64)

    def state_key(self, nu: Any) -> tuple:
        """Per channel ``cov.state_key(nu_c)`` (``None``: no static dispatch)."""
        nu = np.asarray(nu, dtype=np.float64)
        return tuple(c.state_key(nu[s]) for c, s in zip(self.covs, self._slices, strict=True))

    def _chan(self, x: Any, c: int) -> Any:
        return x[c] if self.stacked else x

    def _quad(self, F: Any, r: Any) -> Any:
        return F.quad(r) if self.quad_form is None else F.quad(r, form=self.quad_form)

    # -- pure kernels -------------------------------------------------------
    def _factors(self, parts: _Parts, nu: Any) -> tuple:
        """Each channel's ``cov._factor(constants, nu_c, be)`` (traceable when ``traceable``)."""
        return tuple(cov._factor(parts.covs[c], nu[self._slices[c]], self.be)
                     for c, cov in enumerate(self.covs))

    def _finish(self, parts: _Parts, factors: tuple) -> NoiseState:
        """The state's arithmetic after the factorisations (``ok``, the GB ``logdet + n ln 2 pi``,
        the expanded ``q0``/``y``); pure in ``(parts, factors)``."""
        xp = self.be.xp
        ys, q0s, logdets = [], [], []
        ok = True
        for c, F in enumerate(factors):
            logdet = F.logdet
            ok = ok & F.ok & xp.isfinite(logdet)
            if self.strategy == "expanded":
                q0, y = _expanded_terms(F, self._chan(parts.y_data, c), xp)
                ok = ok & xp.isfinite(q0)
                ys.append(y)
                q0s.append(q0)
                if isinstance(self.density, RealGaussian):
                    logdet = logdet + self.density.log_norm(self.n_eff[c])
            logdets.append(logdet)
        expanded = self.strategy == "expanded"
        return NoiseState(tuple(factors), tuple(ys) if expanded else None,
                          tuple(q0s) if expanded else None, tuple(logdets), ok)

    def _noise_state(self, parts: _Parts, nu: Any) -> NoiseState:
        return self._finish(parts, self._factors(parts, nu))

    def _evaluate(self, parts: _Parts, theta: Any, state: NoiseState) -> Any:
        be, xp = self.be, self.be.xp
        hc = parts.map.apply(parts.signal(theta[self._src]), be=be)
        if hc.shape != parts.y_data.shape:            # static shapes: a Python check under jit
            raise ValueError(f"the mapped signal has shape {hc.shape}; the mapped data {parts.y_data.shape}")
        log10A = None if self._amp is None else theta[self._amp]
        total = 0.0
        for c in range(self.n_channels):
            F, h = state.factors[c], self._chan(hc, c)
            if self.strategy == "residual":
                q = self._quad(F, self._chan(parts.y_data, c) - h)
                if isinstance(self.density, RealGaussian):
                    val = self.density.lnl(q, state.logdet[c], self.n_eff[c], log10A, be=be)
                else:
                    val = self.density.lnl(q, state.logdet[c], self.n_eff[c])
            else:
                hy = _inner(h, state.y[c], xp)
                tt = _signal_quad(F, h, xp)
                if isinstance(self.density, RealGaussian):
                    val = self.density.lnl_expanded(state.q0[c], hy, tt, state.logdet[c])
                else:
                    val = self.density.lnl_expanded(state.q0[c], hy, tt, state.logdet[c],
                                                    self.n_eff[c])
            total = total + val
        ok = state.ok & xp.isfinite(total)
        if self.row_ok is not None:
            ok = ok & self.row_ok(theta)
        return xp.where(ok, total, -xp.inf)

    def lnl_row(self, parts: _Parts, theta: Any) -> tuple[Any, Any]:
        """``(lnL, failed)`` of one row, noise state included (jit/vmap-able on JAX when
        ``traceable``).  A concretely failed state (outside ``jit``: NumPy, eager JAX) is
        never applied -- a combinator's factor (``FlatMarginal``, ``SchurEdge``, ragged
        block-tridiagonal, zero padding) raises "check ok first" -- and gives
        ``(-inf, True)``; under tracing the guard is inert (``ok`` is a tracer) and
        ``_evaluate``'s ``where(ok, ...)`` maps the row to ``-inf``."""
        state = self._noise_state(parts, self.noise_slice(theta))
        if concrete_failure(state.ok):
            return -self.be.xp.inf, True
        return self._evaluate(parts, theta, state), ~state.ok

    # -- user-facing --------------------------------------------------------
    def noise_state(self, nu: Any) -> NoiseState:
        """The factorisation(s) at the noise slice ``nu``: one compiled program on JAX when
        ``traceable``; otherwise each covariance's own ``factor(nu_c, be)`` (staged rows,
        or eager for the reference strategies), then the compiled rest."""
        nu = self.be.asarray(np.asarray(nu, dtype=np.float64))
        if self._state_fn is not None:
            return self._state_fn(self.parts, nu)
        factors = tuple(cov.factor(nu[self._slices[c]], self.be) for c, cov in enumerate(self.covs))
        return self._finish_fn(self.parts, factors)

    def evaluate(self, theta: Any, state: NoiseState) -> Any:
        """One row's lnL against a noise state (compiled on JAX when ``jittable``); ``-inf``
        without applying the factors when the state failed (``state.ok`` concretely False)."""
        if concrete_failure(state.ok):
            return -self.be.xp.inf
        return self._eval_fn(self.parts, self.be.asarray(np.asarray(theta, dtype=np.float64)), state)

    def quad_logdet(self, theta: Any) -> np.ndarray:
        """``(n_channels, 2)``: each channel's ``(r^T C^-1 r, ln|C|)`` at ``theta`` (``A = 1``,
        marginal included; the G6 "parts").  The expanded strategy gives
        ``q0 - 2 h.y + |L^-1 h|^2`` as the oracles' re-summation did
        (``scripts/golden/workers/gbw_intermediates.py:69-70``)."""
        theta = np.asarray(theta, dtype=np.float64)
        state = self.noise_state(self.noise_slice_host(theta))
        if not bool(state.ok):
            raise np.linalg.LinAlgError("the noise state at theta is not positive definite")
        be, xp, parts = self.be, self.be.xp, self.parts
        th = be.asarray(theta)
        hc = parts.map.apply(parts.signal(th[self._src]), be=be)
        out = np.empty((self.n_channels, 2))
        for c in range(self.n_channels):
            F, h = state.factors[c], self._chan(hc, c)
            if self.strategy == "residual":
                q = self._quad(F, self._chan(parts.y_data, c) - h)
            else:
                q = state.q0[c] - 2.0 * _inner(h, state.y[c], xp) + _signal_quad(F, h, xp)
            out[c] = float(q), float(F.logdet)
        return out

    def __call__(self, Z: Any) -> np.ndarray:
        """Uncached reference: every row factorises its own noise state."""
        X = _rows(Z, self.ndim)
        out = np.full(X.shape[0], -np.inf)
        failed = np.zeros(X.shape[0], dtype=bool)
        self.evaluations += X.shape[0]
        with np.errstate(all="ignore"):
            for r, row in enumerate(X):
                state = self.noise_state(self.noise_slice_host(row))
                self.noise_states += 1
                if not bool(state.ok):
                    self.pd_failed_states += 1
                    self.pd_failed_rows += 1
                    failed[r] = True
                    continue
                out[r] = float(self.evaluate(row, state))
        self.last_failed = failed
        return out

    lnlike = __call__


# ---------------------------------------------------------------------------
# stationary Whittle forms
# ---------------------------------------------------------------------------


@pytree_dataclass(data=("data", "L", "sI", "sG", "map", "signal"), meta=())
class _WhittleParts:
    data: Any
    L: Any
    sI: Any
    sG: Any
    map: Any
    signal: Any


class FourierWhittle(_CountersMixin):
    """Stationary Whittle likelihoods in the two oracle forms (spec section 4.7).

    ``form="real-orthonormal"`` (WT ``FreqWhittle`` with ``analysis_fmin > 0`` and
    ``MatrixWhittle("whittle_cut")``; ``wdm_toys/likelihood.py:240-250``,
    ``matrix.py:862-898``): ``data`` a ``TimeSeries``; ``crop=(start, stop)`` selects the
    samples (``None``: the whole record) and the rfft grid is the crop's own; bins
    ``fmin <= f < f_Nyq`` without DC/Nyquist are kept; ``psd.L(f)`` gives the log
    frequency (``LogLinearPSD``); ``params.noise = (log10 A, alpha)``.  The rfft is taken
    here, not through ``TimeToFourier``, because the oracle sums ``|R|^2`` over complex
    bins (``2/N_c sum |R|^2 / lambda``) and the orthonormal real coordinates would round
    differently.

    ``form="complex"`` (GB ``lnl_frequency`` through ``likelihoods.make_lnlike`` /
    ``matrix.make_lnlike``; ``galactic_binary/likelihoods.py:160-162``, ``matrix.py:895-902``):
    ``data`` the store ``FourierBins``; ``map`` takes store bins to the analysed bins
    (``BinSelect`` to the nominal bins, or ``FourierCrop``); ``sI``/``sG`` ``(C, n)`` on the
    analysed bins, already times ``N_c / N`` for the crop; ``params.noise`` holds one
    ``mu_c`` per channel; ``m = s_I + e^{mu_c} s_G``, lnL ``= sum_c (-sum|r|^2/m - sum ln m)``
    from ``0.0``.
    """

    FORMS = ("real-orthonormal", "complex")

    def __init__(self, form: str, data: Any, *, signal: Callable, params: Params, psd: Any = None,
                 fmin: float | None = None, crop: tuple[int, int] | None = None, map: Any = None,
                 sI: Any = None, sG: Any = None, row_ok: Callable | None = None,
                 backend: str | Backend | None = None) -> None:
        if form not in self.FORMS:
            raise ValueError(f"form must be one of {self.FORMS}; got {form!r}")
        if not isinstance(params, Params):
            raise TypeError("params must be a domainia.likelihood.Params")
        self.form, self.params, self.row_ok = form, params, row_ok
        values = getattr(data, "values", data)
        self.be = be = _resolve_backend(values, backend)
        values = np.asarray(values)
        self._src = np.asarray(params.source, dtype=np.int64)
        self._noise = np.asarray(params.noise, dtype=np.int64)
        self._density_real = RealGaussian(0.0)
        self._density_complex = ComplexGaussian()
        if form == "real-orthonormal":
            if psd is None or fmin is None or map is not None or sI is not None or sG is not None:
                raise ValueError("form='real-orthonormal' takes psd and fmin (and crop), not map/sI/sG")
            if not (np.isfinite(fmin) and fmin > 0):
                raise ValueError(f"fmin must be a positive analysis cut; got {fmin!r}")
            if len(params.noise) != 2:
                raise ValueError("form='real-orthonormal' needs params.noise = (log10 A, alpha)")
            dom = data.domain
            N, dt = int(dom.N), float(dom.dt)
            start, stop = (0, N) if crop is None else (int(crop[0]), int(crop[1]))
            if not 0 <= start < stop <= N:
                raise ValueError(f"crop {crop} is not inside the {N}-sample record")
            self.crop, self.Nc, self.dt = (start, stop), stop - start, dt
            f = np.fft.rfftfreq(self.Nc, d=dt)
            keep = (f >= max(float(fmin), 0.0)) & (f < 0.5 / dt)
            keep[[0, -1]] = False
            if not keep.any():
                raise ValueError("fmin leaves no interior Fourier bin")
            self.keep = np.flatnonzero(keep)
            self.n_keep = 2 * int(self.keep.size)
            L = np.asarray(psd.L(f[self.keep]), dtype=np.float64)
            parts = _WhittleParts(values, L, None, None, None, signal)
        else:
            if map is None or sI is None or sG is None or psd is not None or fmin is not None \
                    or crop is not None:
                raise ValueError("form='complex' takes map, sI and sG, not psd/fmin/crop")
            d = np.asarray(map.apply(values, be=NUMPY))
            sI, sG = np.asarray(sI, dtype=np.float64), np.asarray(sG, dtype=np.float64)
            if d.ndim != 2 or sI.shape != d.shape or sG.shape != d.shape:
                raise ValueError(f"data bins {d.shape}, sI {sI.shape} and sG {sG.shape} must agree "
                                 "as (channels, bins)")
            if len(params.noise) != d.shape[0]:
                raise ValueError("form='complex' needs one noise entry per channel")
            self.n_keep = int(d.shape[1])
            parts = _WhittleParts(d, None, sI, sG, map, signal)
        if be.name == "jax":
            import jax

            parts = jax.tree_util.tree_map(be.asarray, parts)
            _check_pytree(parts, "FourierWhittle")
            self.parts = jax.device_put(parts)
            self._row_fn = jax.jit(self.lnl_row)
        else:
            self.parts, self._row_fn = parts, self.lnl_row
        self.reset_counters()

    @property
    def ndim(self) -> int:
        return self.params.ndim

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.params.names

    def _terms(self, parts: _WhittleParts, theta: Any) -> list[tuple[Any, Any]]:
        """``[(quad, logdet)]`` per channel (one entry for the real form)."""
        be, xp = self.be, self.be.xp
        if self.form == "real-orthonormal":
            r = parts.data - parts.signal(theta[self._src])
            x = r[self.crop[0]:self.crop[1]]
            R = be.fft.rfft(x)[self.keep]
            log10A, alpha = theta[self._noise[0]], theta[self._noise[1]]
            lam = 10.0 ** log10A * xp.exp(alpha * parts.L) / (2.0 * self.dt)
            return [(2.0 / self.Nc * xp.sum(xp.abs(R) ** 2 / lam), 2.0 * xp.sum(xp.log(lam)))]
        hc = parts.map.apply(parts.signal(theta[self._src]), be=be)
        out = []
        for c in range(parts.data.shape[0]):
            r = parts.data[c] - hc[c]
            m = parts.sI[c] + xp.exp(theta[self._noise[c]]) * parts.sG[c]
            out.append((xp.sum(xp.abs(r) ** 2 / m), xp.sum(xp.log(m))))
        return out

    def lnl_row(self, parts: _WhittleParts, theta: Any) -> tuple[Any, Any]:
        xp = self.be.xp
        terms = self._terms(parts, theta)
        if self.form == "real-orthonormal":
            val = self._density_real.lnl(terms[0][0], terms[0][1], self.n_keep)
        else:
            val = 0.0
            for quad, logdet in terms:
                val = val + self._density_complex.lnl(quad, logdet)
        ok = xp.isfinite(val)
        if self.row_ok is not None:
            ok = ok & self.row_ok(theta)
        return xp.where(ok, val, -xp.inf), xp.zeros((), dtype=bool)

    def quad_logdet(self, theta: Any) -> np.ndarray:
        """``(channels, 2)``: ``(quad, logdet)`` per channel at ``theta`` (the G6 "parts")."""
        terms = self._terms(self.parts, self.be.asarray(np.asarray(theta, dtype=np.float64)))
        return np.array([[float(q), float(l)] for q, l in terms])

    def __call__(self, Z: Any) -> np.ndarray:
        X = _rows(Z, self.ndim)
        self.evaluations += X.shape[0]
        out = np.empty(X.shape[0])
        with np.errstate(all="ignore"):
            for r, row in enumerate(X):
                out[r] = float(self._row_fn(self.parts, self.be.asarray(row))[0])
        self.last_failed = np.zeros(X.shape[0], dtype=bool)
        return out

    lnlike = __call__
