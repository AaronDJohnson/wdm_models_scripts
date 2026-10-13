"""D-D4: NoiseStateCache, BatchedLikelihood and PriorSanitized against their oracles
(GB fullcache.NoiseStateCache, gpu/jax_models.BatchedLikelihood, sinusoid run.py
PriorSanitizedLikelihood)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dom_toys import ToyMap, ToyReMap, ToySignal, dense_cov, diag_cov, spd
from domainia.covariance.stack import ChannelStack
from domainia.density import RealGaussian
from domainia.likelihood import COUNTERS, FourierWhittle, GaussianLikelihood, Params, StaticSignal
from domainia.sampling import BatchedLikelihood, NoiseStateCache, PriorSanitized, group_misses

P, NB = 16, 20
PARAMS = Params(("s0", "s1", "a0", "b0", "a1", "b1"), (0, 1), (2, 3, 4, 5))


def two_channel(seed=0, backend=None, strategy="residual"):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((P, NB)) + 1j * rng.standard_normal((P, NB))
    B = rng.standard_normal((2, 2 * NB)) + 1j * rng.standard_normal((2, 2 * NB))
    z = rng.standard_normal((2, NB)) + 1j * rng.standard_normal((2, NB))
    cov = ChannelStack((dense_cov(spd(P, rng), P), dense_cov(spd(P, rng), P)), 2)
    return GaussianLikelihood(z, ToyReMap(C), cov, RealGaussian(), signal=ToySignal(B, (2, NB)),
                              params=PARAMS, strategy=strategy, quad_form="whiten", backend=backend)


def noise_rows(n_states, reps, seed=1):
    """``reps`` source moves per noise state, states interleaved like PT rungs."""
    rng = np.random.default_rng(seed)
    states = np.column_stack([rng.uniform(-0.3, 0.3, n_states), rng.uniform(0.0, 0.5, n_states),
                              rng.uniform(-0.3, 0.3, n_states), rng.uniform(0.0, 0.5, n_states)])
    X = np.repeat(states, reps, axis=0)
    return np.column_stack([rng.standard_normal((len(X), 2)), X])


def test_group_misses_is_first_appearance_order():
    assert group_misses([10, 8, 10, None, 8, 12]) == [(10, [0, 2]), (8, [1, 4]), (None, [3]), (12, [5])]
    assert group_misses([]) == []


def test_cache_values_equal_the_uncached_reference_bitwise():
    lik = two_channel()
    X = noise_rows(3, 4)
    want = lik(X)
    cache = NoiseStateCache(two_channel(), 24)
    assert np.array_equal(cache(X), want)
    assert cache.counters() == {"noise_states": 3, "pd_failed_states": 0, "pd_failed_rows": 0,
                                "evaluations": 12, "cache_hits": 9}
    assert tuple(cache.counters())[:4] == COUNTERS


def test_lru_eviction_and_failed_states_follow_fullcache():
    cache = NoiseStateCache(two_channel(), 2)
    X = noise_rows(3, 1)
    X[2, 3] = -60.0                                        # third state: channel 0 not PD
    out = cache(np.concatenate([X, X[[2, 0]]]))           # 5 rows: s0 s1 s2(F) s2(F, hit) s0(evicted)
    assert np.isneginf(out[[2, 3]]).all() and np.isfinite(out[[0, 1, 4]]).all()
    assert cache.last_failed.tolist() == [False, False, True, True, False]
    assert cache.counters() == {"noise_states": 4, "pd_failed_states": 1, "pd_failed_rows": 2,
                                "evaluations": 5, "cache_hits": 1}
    assert len(cache.cache) == 2 and list(cache.cache.values())[0] is None   # failed cached as None
    cache.reset_counters()
    assert len(cache.cache) == 0 and cache.counters()["evaluations"] == 0


def test_key_is_the_float64_bytes_of_the_clamped_noise_slice():
    assert NoiseStateCache.key(np.array([1.0, -2.0])) == np.array([1.0, -2.0]).tobytes()
    n = 6
    lik = GaussianLikelihood(np.zeros(n), ToyMap(np.eye(n)), diag_cov(np.ones(n), n), RealGaussian(),
                             signal=ToySignal(np.ones((1, n))), params=Params(("s", "a"), (0,), (1,)),
                             noise_clamp={"a": 4.0})
    cache = NoiseStateCache(lik, 4)
    cache(np.array([[0.0, 5.0], [0.0, 9.0]]))                # both clamp to 4.0: one state
    assert cache.counters()["noise_states"] == 1 and cache.counters()["cache_hits"] == 1


def test_batched_misses_on_numpy_equal_serial():
    X = noise_rows(4, 2)
    a = NoiseStateCache(two_channel(), 8)(X)
    b = NoiseStateCache(two_channel(), 8, misses="batched")(X)
    assert np.array_equal(a, b)


def test_constructor_checks():
    lik = two_channel()
    with pytest.raises(ValueError, match="capacity"):
        NoiseStateCache(lik, 0)
    with pytest.raises(ValueError, match="misses"):
        NoiseStateCache(lik, 4, misses="parallel")
    with pytest.raises(ValueError, match="needs the JAX backend"):
        NoiseStateCache(lik, 4, batch_rows=True)
    with pytest.raises(TypeError, match="GaussianLikelihood"):
        NoiseStateCache(object(), 4)


def test_batched_likelihood_on_numpy_equals_the_likelihood():
    X = noise_rows(3, 2)
    X[1, 3] = -60.0
    lik = two_channel()
    want = lik(X)
    bl = BatchedLikelihood(two_channel())
    got = bl(X)
    assert np.array_equal(got, want)
    assert bl.last_failed.tolist() == lik.last_failed.tolist()
    assert bl.counters()["pd_failed_rows"] == 1 and bl.counters()["evaluations"] == 6


def test_batched_count_states_false_counts_rows_but_no_states():
    """E-D1: ``count_states=False`` is the WG sinusoid's uncached diagonal models
    (``wdmtoysgpu/matrix.py:776-786``): values, ``last_failed``, ``evaluations`` and
    ``pd_failed_rows`` as the default; ``noise_states`` and ``pd_failed_states`` stay 0."""
    X = noise_rows(3, 2)
    X[1, 3] = -60.0
    counted, plain = BatchedLikelihood(two_channel()), BatchedLikelihood(two_channel(), count_states=False)
    assert np.array_equal(plain(X), counted(X))
    assert plain.last_failed.tolist() == counted.last_failed.tolist()
    assert counted.counters() == {"noise_states": 6, "pd_failed_states": 1, "pd_failed_rows": 1,
                                  "evaluations": 6, "cache_hits": 0}
    assert plain.counters() == {"noise_states": 0, "pd_failed_states": 0, "pd_failed_rows": 1,
                                "evaluations": 6, "cache_hits": 0}
    assert plain.count_states is False and counted.count_states is True


def sinusoid_whittle(backend=None):
    N, dt = 512, 2.0 ** -9
    rng = np.random.default_rng(5)
    ts = SimpleNamespace(values=rng.standard_normal(N), domain=SimpleNamespace(N=N, dt=dt))

    class LogF:
        def L(self, f):
            return np.log(np.maximum(np.asarray(f, dtype=float), 4.0))

    if backend == "jax":
        import jax.numpy as xp
    else:
        xp = np
    t = xp.asarray(np.arange(N) * dt)
    return FourierWhittle("real-orthonormal", ts, psd=LogF(), fmin=4.0,
                          signal=StaticSignal(lambda s: s[0] * xp.cos(2.0 * np.pi * s[1] * t + s[2])),
                          params=Params(("A", "f", "p", "lA", "al"), (0, 1, 2), (3, 4)), backend=backend)


THETA_W = np.array([[0.5, 37.0, 0.1, 0.0, -1.0], [0.2, 80.0, 1.0, 0.3, -2.0]])


def test_batched_likelihood_wraps_fourier_whittle():
    lik = sinusoid_whittle()
    assert np.array_equal(BatchedLikelihood(lik)(THETA_W), lik(THETA_W))


class _Recorder:
    """A likelihood stub that records what the backend was asked."""

    ndim = 3

    def __init__(self, values=None, failed=None):
        self.seen, self.values, self.failed = None, values, failed

    def __call__(self, X):
        self.seen = X.copy()
        self.last_failed = (np.zeros(len(X), bool) if self.failed is None else np.asarray(self.failed))
        return np.zeros(len(X)) if self.values is None else np.asarray(self.values, float)

    def counters(self):
        return {"evaluations": len(self.seen)}


BOUNDS = np.array([[0.0, 3.0], [-np.pi, np.pi], [-2.0, 2.0]])
ANCHOR = np.array([1.0, 0.6, 1.2])


def test_prior_sanitized_replaces_and_masks_out_of_prior_rows():
    rec = _Recorder()
    ps = PriorSanitized(rec, BOUNDS, ANCHOR, periodic={1: (-np.pi, np.pi)})
    X = np.array([[1.0, 0.5, 0.0], [5.0, 0.5, 0.0], [1.0, 4.0, 0.0], [np.nan, 0.0, 0.0]])
    out = ps(X)
    assert out.tolist()[:3] == [0.0, -np.inf, 0.0] and np.isneginf(out[3])
    assert np.array_equal(rec.seen[1], ANCHOR) and np.array_equal(rec.seen[3], ANCHOR)
    assert rec.seen[2, 1] == pytest.approx(4.0 - 2 * np.pi)          # wrapped, then seen
    assert ps(np.array([1.0, 0.5, 0.0])) == 0.0                         # scalar row -> float
    assert ps.counters() == {"evaluations": 1}


def test_prior_sanitized_passes_flagged_failures_and_raises_on_others():
    ps = PriorSanitized(_Recorder(values=[-np.inf, 1.0], failed=[True, False]), BOUNDS, ANCHOR)
    assert ps(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])).tolist() == [-np.inf, 1.0]
    ps = PriorSanitized(_Recorder(values=[np.nan, 1.0]), BOUNDS, ANCHOR)
    with pytest.raises(FloatingPointError, match="non-finite"):
        ps(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    ps = PriorSanitized(_Recorder(values=[-np.inf, 1.0]), BOUNDS, ANCHOR)       # -inf not flagged
    with pytest.raises(FloatingPointError):
        ps(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_prior_sanitized_checks():
    with pytest.raises(ValueError, match="anchor is outside"):
        PriorSanitized(_Recorder(), BOUNDS, np.array([9.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="bounds must be"):
        PriorSanitized(_Recorder(), BOUNDS[:2], ANCHOR)
    with pytest.raises(ValueError, match="out of range"):
        PriorSanitized(_Recorder(), BOUNDS, ANCHOR, periodic={-1: (-np.pi, np.pi)})
    with pytest.raises(ValueError, match="hi > lo"):
        PriorSanitized(_Recorder(), BOUNDS, ANCHOR, periodic={1: (np.pi, -np.pi)})
    ps = PriorSanitized(_Recorder(), BOUNDS, ANCHOR)
    with pytest.raises(ValueError, match="real"):
        ps(np.array([1.0 + 1j, 0.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        ps(np.zeros((2, 4)))


def test_prior_sanitized_over_a_cache_end_to_end():
    lik = NoiseStateCache(two_channel(), 8)
    bounds = np.array([[-5, 5], [-5, 5], [-1, 1], [-1, 1], [-1, 1], [-1, 1]], float)
    anchor = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.2])
    X = noise_rows(2, 2)
    X[3, 0] = 9.0                                            # outside the box
    out = PriorSanitized(lik, bounds, anchor)(X)
    assert np.isneginf(out[3]) and np.isfinite(out[:3]).all()


# --------------------------------------------------------------------------- JAX


def test_jax_cache_batches_hits_and_matches_numpy():
    pytest.importorskip("jax")
    X = noise_rows(3, 3)
    X[3:6, 3] = -60.0                                        # the second state fails: 3 rows, 1 state
    want = two_channel()(X)
    for strategy in ("residual", "expanded"):
        cache = NoiseStateCache(two_channel(backend="jax", strategy=strategy), 16, batch_rows=True)
        assert cache.batch_rows                              # the default on CPU is off (WG, E-C4)
        got = cache(X)
        assert np.array_equal(np.isneginf(got), np.isneginf(want))
        ok = np.isfinite(want)
        assert np.max(np.abs(got[ok] - want[ok]) / np.abs(want[ok])) <= 1e-12
        assert cache.counters() == {"noise_states": 3, "pd_failed_states": 1, "pd_failed_rows": 3,
                                    "evaluations": 9, "cache_hits": 6}
        looped = NoiseStateCache(two_channel(backend="jax", strategy=strategy), 16, batch_rows=False)
        assert np.max(np.abs(looped(X)[ok] - got[ok]) / np.abs(got[ok])) <= 1e-12


def test_jax_batched_misses_group_and_match_serial():
    pytest.importorskip("jax")
    X = noise_rows(4, 2)
    serial = NoiseStateCache(two_channel(backend="jax"), 8)(X)
    batched = NoiseStateCache(two_channel(backend="jax"), 8, misses="batched")(X)
    assert np.max(np.abs(serial - batched) / np.abs(serial)) <= 1e-12


def test_jax_batched_likelihood_matches_numpy():
    pytest.importorskip("jax")
    X = noise_rows(3, 2)
    X[1, 3] = -60.0
    want = two_channel()(X)
    bl = BatchedLikelihood(two_channel(backend="jax"))
    got = bl(X)
    assert bl.last_failed.tolist() == [False, True, False, False, False, False]
    ok = np.isfinite(want)
    assert np.array_equal(np.isfinite(got), ok)
    assert np.max(np.abs(got[ok] - want[ok]) / np.abs(want[ok])) <= 1e-12
    a = BatchedLikelihood(sinusoid_whittle("jax"))(THETA_W)
    b = sinusoid_whittle()(THETA_W)
    assert np.max(np.abs(a - b) / np.abs(b)) <= 1e-12


def test_importing_sampling_does_not_import_jax():
    src = Path(__file__).resolve().parents[1] / "src"
    wdm_src = Path(__file__).resolve().parents[2] / "wdm" / "src"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(src), str(wdm_src)]))
    out = subprocess.run([sys.executable, "-P", "-c", "import sys, domainia.sampling; print('jax' in sys.modules)"],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
