"""S4 deferrals (b) and (c) at the production sinusoid grid (64 x 256, manuscript_mid).

* Per-state memory: the JAX noise state (sum of its leaves' ``nbytes``: what one LRU slot
  holds) of band_pad, band_exact and tri_raw is within 1.5x of WG's (9.3 / 120 / 81 MiB);
  before S4 it was 297 / 249 / 79-97 MiB, and WT's state forms still fail the gate (the
  negative control).  A floor of 0.5x WG makes a measure that misses the state's leaves
  fail instead of passing.  The value at the truth stays inside the approved
  JAX reference gate (``golden/sinusoid/tolerances_measured.json``).
* Keyed jit: band_exact (``FlatMarginal(SchurEdge(adaptive staged rows))``) compiles one
  program per Chebyshev order, and a new state on a warm program is faster than the eager
  reference (``jit=False``) measured in the same test (reference CPU: 0.65-0.73 s per state
  against 1.5-4.6 s eager; the times are printed, the absolute values are not gated).

``slow``: about 30 s and 3 GB of device memory (the staged rows' Phi).  Run with
``-m slow`` (the S4 section gate does).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scripts.golden.goldens_io import read_json, to_float
from scripts.golden.paths import GOLDEN

pytestmark = [pytest.mark.jax, pytest.mark.golden, pytest.mark.slow]

jax = pytest.importorskip("jax")


def _gate(model):
    g = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")["approved"]["jax"]["references"][model]
    return to_float(g["atol"]), to_float(g["rtol"])


@pytest.mark.parametrize("model", ["band_pad", "band_exact", "tri_raw"])
def test_noise_state_memory_is_within_1p5x_of_wg(model):
    from scripts.golden.s2_cases import sinusoid_inputs
    from scripts.golden.s4_cases import WG_STATE_MIB, sinusoid_jax_likelihood, state_bytes

    truth = sinusoid_inputs().truth
    lik = sinusoid_jax_likelihood(model)
    state = lik.noise_state(lik.noise_slice_host(truth))
    mib = state_bytes(state) / 2**20
    assert mib <= 1.5 * WG_STATE_MIB[model], f"{model}: {mib:.2f} MiB per state (WG {WG_STATE_MIB[model]})"
    assert mib >= 0.5 * WG_STATE_MIB[model], \
        f"{model}: {mib:.2f} MiB is implausibly small (WG {WG_STATE_MIB[model]}): the measure missed leaves"
    ref = np.load(GOLDEN / "sinusoid" / "references" / f"manuscript_mid_{model}_reference.npz")
    assert np.array_equal(ref["theta"][0], truth)
    want = float(ref["loglike"][0])
    atol, rtol = _gate(model)
    got = float(lik.evaluate(truth, state))
    assert abs(got - want) <= atol + rtol * abs(want), (got, want)


def test_band_exact_compiles_one_program_per_order_and_beats_eager():
    import jax.numpy as jnp

    from domainia.covariance import cheb_order

    from scripts.golden.s4_cases import sinusoid_jax_likelihood

    lik = sinusoid_jax_likelihood("band_exact")
    cov = lik.covs[0]
    nus = [np.array(v) for v in ([-2.0, 1.2], [-2.01, 1.21], [-1.99, 1.19], [-2.0, 0.3], [-2.02, 0.31])]
    orders = [cheb_order(nu[1], cov.cov.rows.L_max) for nu in nus]
    assert len(set(orders)) == 2
    seconds = []
    for nu in nus:
        t0 = time.perf_counter()
        state = lik.noise_state(nu)
        jax.block_until_ready(state)
        seconds.append(time.perf_counter() - t0)
        assert bool(state.ok)
    keys = sorted({((p, None),) for p in orders})
    assert sorted(cov.compiled_keys()) == keys
    warm = [s for s, (i, p) in zip(seconds, enumerate(orders), strict=True) if orders.index(p) != i]
    assert len(warm) == 3
    t0 = time.perf_counter()                                  # the eager reference (the S2 path)
    eager = cov.factor(jnp.asarray(nus[1]), lik.be, jit=False)
    jax.block_until_ready(eager)
    t_eager = time.perf_counter() - t0
    assert bool(eager.ok) and sorted(cov.compiled_keys()) == keys   # jit=False compiled nothing
    print(f"band_exact s per new state: compiled {[round(s, 2) for s in seconds]}, eager {t_eager:.2f}")
    assert float(np.median(warm)) < t_eager, (seconds, t_eager)


@pytest.mark.parametrize("model", ["band_pad", "band_exact"])
def test_the_memory_gate_bites(model):
    """Negative control: WT's state forms (``LiWp``; ``T^-1 B^T``) on JAX exceed the gate."""
    from scripts.golden.s2_cases import sinusoid_inputs
    from scripts.golden.s4_cases import WG_STATE_MIB, sinusoid_jax_likelihood, state_bytes

    lik = sinusoid_jax_likelihood(model, forms="wt")
    mib = state_bytes(lik.noise_state(lik.noise_slice_host(sinusoid_inputs().truth))) / 2**20
    assert mib > 1.5 * WG_STATE_MIB[model], mib
