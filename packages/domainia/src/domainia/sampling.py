"""Host-side sampling glue: the noise-state LRU, batched rows and prior sanitising
(spec section 3.5).  Caches live here, never inside a covariance.

* :class:`NoiseStateCache` -- GB ``fullcache.NoiseStateCache`` / ``MatrixCache`` /
  ``EdgeSchurCache`` / ``FullCache`` (``galactic_binary/fullcache.py:192-293``) and the
  WG ``MatrixGPU._cached_rows`` (``gpu_sinusoid/wdmtoysgpu/matrix.py:726-795``): an LRU
  of noise states keyed by the float64 bytes of the (clamped) noise slice; failed
  states cached as ``None``; the counters of ``chain_stats.COUNTERS`` plus
  ``cache_hits`` and ``last_failed``; on JAX the rows of one call are evaluated in one
  ``vmap`` over their stacked states (GB ``batch_signal``).
* :class:`BatchedLikelihood` -- GB ``gpu/jax_models.BatchedLikelihood`` (``:21-36``):
  ``jit(vmap(row))`` for the models without a cached factorisation.
* :class:`PriorSanitized` -- ``toy_models/sinusoid/gpu/run.py:135-178``: rows outside the
  prior are replaced by an anchor before the device call and masked back to ``-inf``.

Nothing here imports JAX at module import.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .likelihood import COUNTERS, FourierWhittle, GaussianLikelihood, NoiseState, _CountersMixin, _rows

__all__ = ["COUNTERS", "BatchedLikelihood", "NoiseStateCache", "PriorSanitized", "group_misses"]

MISSES = ("serial", "batched")


def group_misses(group_keys: Sequence[Hashable]) -> list[tuple[Hashable, list[int]]]:
    """Positions grouped by their static key, groups in order of first appearance and
    positions ascending: a deterministic grouping that never depends on hashing (spec
    section 3.5, misses grouped by the adaptive order P)."""
    groups: dict[Hashable, list[int]] = {}
    order: list[Hashable] = []
    for i, k in enumerate(group_keys):
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    return [(k, groups[k]) for k in order]


def _accelerator() -> bool:
    """WG ``batch_rungs``: ``jax.default_backend() != "cpu"`` (stacking costs more than it
    saves on a CPU)."""
    import jax

    return jax.default_backend() != "cpu"


def _stack(states: list[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *states)


def _unstack(stacked: Any, n: int) -> list[Any]:
    import jax

    return [jax.tree_util.tree_map(lambda a, i=i: a[i], stacked) for i in range(n)]


class NoiseStateCache(_CountersMixin):
    """LRU of noise states for a :class:`GaussianLikelihood`.

    ``capacity``: states kept (sinusoid 16 on GPU / 32 on CPU, GB 24; spec section 3.5).
    ``batch_rows``: evaluate the rows of one call in one ``vmap`` over their stacked
    states (JAX with ``lik.jittable`` only; NumPy loops, which is its oracle).  Default
    (``None``): the WG policy (``MatrixGPU.batch_rungs``, ``_cached_rows``) -- on for a
    ``lik.batchable`` JAX likelihood on an accelerator, off on CPU and never for a
    covariance that is not ``batchable`` (the sinusoid edge-exact states, ~120 MiB each);
    ``True`` there is refused.  GB ``batch_signal`` batched on every device: the model layer
    (S7, ``wdm_models.options``) passes ``batch_rows=True`` for the cached GB models on JAX except
    ``exact`` (``batch_rows=False``: lineage C's ``FullCache`` looped over the rows), and
    ``False`` on NumPy.  ``misses``:
    ``"serial"`` (golden default, one factorisation per new state in row order) or
    ``"batched"`` (opt-in, gated in S8: new states grouped by ``lik.state_key`` -- the
    covariances' ``state_key``, e.g. the adaptive Chebyshev order -- with
    :func:`group_misses`, each group one vmapped factorisation when ``lik.traceable``,
    else serial).

    Counters (GB ``NoiseStateCache``): ``noise_states`` factorisations attempted (a state
    evicted and proposed again counts again); ``pd_failed_states`` those not positive
    definite; ``pd_failed_rows`` rows that got ``-inf`` from a failed state (cached
    repeats included); ``evaluations`` rows evaluated; ``cache_hits`` (WG); and
    ``last_failed``, the failed-row mask of the last call.
    """

    def __init__(self, lik: GaussianLikelihood, capacity: int, *, batch_rows: bool | None = None,
                 misses: Literal["serial", "batched"] = "serial") -> None:
        if not isinstance(lik, GaussianLikelihood):
            raise TypeError("NoiseStateCache needs a GaussianLikelihood")
        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        if misses not in MISSES:
            raise ValueError(f"misses must be one of {MISSES}; got {misses!r}")
        self.lik, self.capacity, self.misses = lik, int(capacity), misses
        jax_ok = lik.be.name == "jax" and lik.jittable
        if batch_rows is None:
            batch_rows = jax_ok and lik.batchable and _accelerator()
        self.batch_rows = bool(batch_rows)
        if self.batch_rows and not jax_ok:
            raise ValueError("batch_rows needs the JAX backend and jittable covariances (NumPy loops the rows)")
        if self.batch_rows and not lik.batchable:
            raise ValueError("batch_rows: this likelihood's noise states are not batchable (edge-exact time rows: "
                             "WG evaluated them one row at a time)")
        self.cache: OrderedDict[bytes, NoiseState | None] = OrderedDict()
        self._batch_fn = None
        self._miss_fn = None
        self.reset_counters()

    @property
    def ndim(self) -> int:
        return self.lik.ndim

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.lik.param_names

    def reset_counters(self) -> None:
        """Zero the counters and empty the cache (GB ``reset_counters``: the runner
        excludes its compile and timing calls this way)."""
        if hasattr(self, "cache"):
            self.cache.clear()
        super().reset_counters()

    @staticmethod
    def key(nu: Any) -> bytes:
        return np.asarray(nu, dtype=np.float64).tobytes()

    # -- states ---------------------------------------------------------------
    def _compute(self, nu: np.ndarray) -> NoiseState | None:
        st = self.lik.noise_state(nu)
        return st if bool(st.ok) else None

    def _compute_group(self, nus: list[np.ndarray]) -> list[NoiseState | None]:
        lik = self.lik
        if lik.be.name != "jax" or not lik.traceable or len(nus) == 1:
            return [self._compute(nu) for nu in nus]
        import jax

        if self._miss_fn is None:
            self._miss_fn = jax.jit(jax.vmap(lik._noise_state, in_axes=(None, 0)))
        stacked = self._miss_fn(lik.parts, lik.be.asarray(np.stack(nus)))
        return [st if bool(st.ok) else None for st in _unstack(stacked, len(nus))]

    def _insert(self, key: bytes, st: NoiseState | None) -> None:
        self.noise_states += 1
        self.pd_failed_states += int(st is None)
        self.cache[key] = st
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def state(self, nu: Any) -> NoiseState | None:
        """The cached state at ``nu`` (clamped noise slice), computing it on a miss."""
        nu = np.asarray(nu, dtype=np.float64)
        k = self.key(nu)
        if k in self.cache:
            self.cache.move_to_end(k)
            self.cache_hits += 1
            return self.cache[k]
        st = self._compute(nu)
        self._insert(k, st)
        return st

    def _states(self, nus: list[np.ndarray]) -> list[NoiseState | None]:
        keys = [self.key(nu) for nu in nus]
        pre: dict[bytes, NoiseState | None] = {}
        if self.misses == "batched":
            new, seen = [], set()
            for k, nu in zip(keys, nus, strict=True):
                if k not in self.cache and k not in seen:
                    seen.add(k)
                    new.append((k, nu))
            for _, pos in group_misses([self.lik.state_key(nu) for _, nu in new]):
                got = self._compute_group([new[i][1] for i in pos])
                pre.update({new[i][0]: s for i, s in zip(pos, got, strict=True)})
        out = []
        for k, nu in zip(keys, nus, strict=True):
            if k in self.cache:
                self.cache.move_to_end(k)
                self.cache_hits += 1
                out.append(self.cache[k])
                continue
            st = pre[k] if k in pre else self._compute(nu)
            self._insert(k, st)
            out.append(st)
        return out

    # -- rows -----------------------------------------------------------------
    def __call__(self, Z: Any) -> np.ndarray:
        lik = self.lik
        X = _rows(Z, lik.ndim)
        R = X.shape[0]
        self.evaluations += R
        states = self._states([lik.noise_slice_host(row) for row in X])
        good = np.array([s is not None for s in states], dtype=bool)
        self.pd_failed_rows += int(np.sum(~good))
        out = np.full(R, -np.inf)
        if good.any():
            if self.batch_rows:
                fill = states[int(np.flatnonzero(good)[0])]
                vals = self._batch(X, [s if s is not None else fill for s in states])
                out[good] = vals[good]
            else:
                with np.errstate(all="ignore"):
                    for r in np.flatnonzero(good):
                        out[r] = float(lik.evaluate(X[r], states[r]))
        self.last_failed = ~good
        return out

    lnlike = __call__

    def _batch(self, X: np.ndarray, states: list[NoiseState]) -> np.ndarray:
        import jax

        lik = self.lik
        if self._batch_fn is None:
            self._batch_fn = jax.jit(jax.vmap(lik._evaluate, in_axes=(None, 0, 0)))
        vals = self._batch_fn(lik.parts, lik.be.asarray(X), _stack(states))
        return np.asarray(vals, dtype=np.float64)


class BatchedLikelihood(_CountersMixin):
    """``(R, ndim) -> (R,)`` in one ``jit(vmap(row))`` on JAX, a row loop on NumPy; no
    cache.  For the models without a cached factorisation (diagonals, Whittle): GB
    ``gpu/jax_models.BatchedLikelihood``, WG ``MatrixGPU._fn``.  On JAX a
    :class:`GaussianLikelihood` must be ``traceable`` (every covariance jittable, no
    static key); otherwise use :class:`NoiseStateCache`.

    ``count_states``: count one noise state per row of a :class:`GaussianLikelihood`
    (``noise_states``, ``pd_failed_states``; default).  ``False`` keeps both at 0, the WG
    sinusoid's uncached diagonal models (``wdmtoysgpu/matrix.py:776-786``: positive by
    construction, "nothing to factor"); ``evaluations`` and ``pd_failed_rows`` count either
    way."""

    def __init__(self, lik: GaussianLikelihood | FourierWhittle, *, count_states: bool = True) -> None:
        if not isinstance(lik, (GaussianLikelihood, FourierWhittle)):
            raise TypeError("BatchedLikelihood needs a GaussianLikelihood or a FourierWhittle")
        if isinstance(lik, GaussianLikelihood) and lik.be.name == "jax" and not lik.traceable:
            raise TypeError("BatchedLikelihood on JAX needs a traceable likelihood (jittable covariances "
                            "without a static key); use NoiseStateCache")
        self.lik = lik
        self.count_states = bool(count_states)
        self._fn = None
        self.reset_counters()

    @property
    def ndim(self) -> int:
        return self.lik.ndim

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.lik.param_names

    def __call__(self, Z: Any) -> np.ndarray:
        lik = self.lik
        X = _rows(Z, lik.ndim)
        if lik.be.name == "jax":
            import jax

            if self._fn is None:
                self._fn = jax.jit(jax.vmap(lik.lnl_row, in_axes=(None, 0)))
            vals, failed = self._fn(lik.parts, lik.be.asarray(X))
            vals, failed = np.asarray(vals, dtype=np.float64), np.asarray(failed, dtype=bool)
        else:
            with np.errstate(all="ignore"):
                res = [lik.lnl_row(lik.parts, row) for row in X]
            vals = np.array([float(v) for v, _ in res], dtype=np.float64)
            failed = np.array([bool(f) for _, f in res], dtype=bool)
        self.evaluations += X.shape[0]
        if isinstance(lik, GaussianLikelihood):
            if self.count_states:
                self.noise_states += X.shape[0]
                self.pd_failed_states += int(failed.sum())
            self.pd_failed_rows += int(failed.sum())
        self.last_failed = failed
        return vals

    lnlike = __call__


class PriorSanitized:
    """Keep rows the prior rejects away from the numerical backend.

    The sampler evaluates fixed-size batches even when the prior rejected rows.  Rows
    outside ``bounds`` (``(ndim, 2)``, after wrapping the ``periodic`` entries
    ``{index: (lo, hi)}``) or non-finite are replaced by ``anchor`` before the call and
    masked back to ``-inf`` after it.  A ``-inf`` the likelihood flags in
    ``last_failed`` (not positive definite: a deliberate prior truncation,
    LIKELIHOOD_MATRIX.md) passes; any other non-finite value raises
    ``FloatingPointError``.  The likelihood sees the wrapped rows.
    """

    def __init__(self, lik: Any, bounds: Any, anchor: Any, *,
                 periodic: Mapping[int, tuple[float, float]] | None = None) -> None:
        self.lik = lik
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.anchor = np.asarray(anchor, dtype=np.float64)
        ndim = int(lik.ndim)
        if self.bounds.shape != (ndim, 2) or self.anchor.shape != (ndim,):
            raise ValueError(f"bounds must be ({ndim}, 2) and anchor ({ndim},)")
        self.periodic: dict[int, tuple[float, float]] = {}
        for i, (lo, hi) in (periodic or {}).items():
            i, lo, hi = int(i), float(lo), float(hi)
            if not 0 <= i < ndim:            # a negative index would wrap the wrong column silently
                raise ValueError(f"periodic index {i} is out of range for {ndim} parameters")
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(f"periodic range ({lo}, {hi}) of index {i} must be finite with hi > lo")
            self.periodic[i] = (lo, hi)
        if not self.valid(self.wrap(self.anchor[None, :]))[0]:
            raise ValueError("the anchor is outside the prior")

    @property
    def ndim(self) -> int:
        return int(self.lik.ndim)

    def wrap(self, X: np.ndarray) -> np.ndarray:
        out = np.array(X, dtype=np.float64, copy=True)
        for i, (lo, hi) in self.periodic.items():
            out[:, i] = lo + np.mod(out[:, i] - lo, hi - lo)
        return out

    def valid(self, X: np.ndarray) -> np.ndarray:
        return (np.isfinite(X).all(axis=1) & (X >= self.bounds[:, 0]).all(axis=1)
                & (X <= self.bounds[:, 1]).all(axis=1))

    def __call__(self, theta: Any) -> float | np.ndarray:
        x = np.asarray(theta)
        if np.iscomplexobj(x):
            raise ValueError("theta must be real")
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 1
        X = x[None, :] if scalar else x
        if X.ndim != 2 or X.shape[1] != self.ndim:
            raise ValueError(f"theta must have shape ({self.ndim},) or (batch, {self.ndim}); got {x.shape}")
        wrapped = self.wrap(X)
        valid = self.valid(wrapped)
        safe = np.where(valid[:, None], wrapped, self.anchor[None, :])
        values = np.asarray(self.lik(safe), dtype=np.float64)
        if values.shape != (len(safe),):
            raise ValueError(f"likelihood returned shape {values.shape}; expected {(len(safe),)}")
        failed = getattr(self.lik, "last_failed", None)
        failed = (np.zeros(len(safe), dtype=bool) if failed is None
                  else np.asarray(failed, dtype=bool) & (values == -np.inf))
        if not (np.isfinite(values) | failed).all():
            raise FloatingPointError("the likelihood returned a non-finite value for a sanitized batch")
        values = np.where(valid, values, -np.inf)
        return float(values[0]) if scalar else values

    lnlike = __call__

    def counters(self) -> dict[str, int]:
        return self.lik.counters()
