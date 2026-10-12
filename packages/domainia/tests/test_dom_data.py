"""D-A2: data vectors are pytrees with static domains (spec section 4.1)."""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from domainia.data import FourierBins, TimeSeries, WDMCells
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain

TD = TimeDomain(32, 0.5)
FD = FourierDomain(32, 0.5, 2, 5, "rfft")
WD = WDMDomain(CellSet.all_active(wdm.Grid(4, 8, 0.5)))
AE = ChannelAxis(("A", "E"))


def test_shapes_are_checked():
    TimeSeries(np.zeros(32), TD)
    TimeSeries(np.zeros((3, 32)), TD)                                   # batch axis
    FourierBins(np.zeros((2, 5), complex), FD, AE)
    WDMCells(np.zeros((7, 2, 32)), WD, AE)
    with pytest.raises(ValueError, match="domain size"):
        TimeSeries(np.zeros(31), TD)
    with pytest.raises(ValueError, match="channels"):
        FourierBins(np.zeros((3, 5), complex), FD, AE)
    with pytest.raises(ValueError, match="at least 2"):
        WDMCells(np.zeros(32), WD, AE)
    with pytest.raises(TypeError, match="TimeDomain"):
        TimeSeries(np.zeros(5), FD)


def test_replace_keeps_the_domain():
    x = TimeSeries(np.zeros(32), TD)
    y = x.replace(values=np.ones(32))
    assert y.domain is TD and np.array_equal(y.values, np.ones(32))


def test_pytree_leaves_and_static_metadata(jax_be):
    import jax

    x = WDMCells(np.arange(64.0).reshape(2, 32), WD, AE)
    leaves, tree = jax.tree_util.tree_flatten(x)
    assert len(leaves) == 1 and leaves[0] is x.values
    y = jax.tree_util.tree_unflatten(tree, [np.ones((2, 32))])
    assert y.domain == WD and y.channels == AE
    # placeholder leaves (vmap axis specs, tree_map to ints) are never shape-checked
    spec = jax.tree_util.tree_map(lambda _: 0, x)
    assert spec.values == 0


def test_jit_and_vmap_over_a_batch(jax_be):
    import jax
    import jax.numpy as jnp

    batch = TimeSeries(jnp.asarray(np.arange(96.0).reshape(3, 32)), TD)
    total = jax.jit(jax.vmap(lambda s: jnp.sum(s.values * s.domain.dt)))(batch)
    assert isinstance(total, jax.Array)
    assert np.allclose(np.asarray(total), np.arange(96.0).reshape(3, 32).sum(1) * 0.5)
    traces = []

    @jax.jit
    def f(s):
        traces.append(s.domain)
        return s.values.sum()

    f(TimeSeries(jnp.zeros(32), TD))
    f(TimeSeries(jnp.ones(32), TD))
    f(TimeSeries(jnp.ones(32), TimeDomain(32, 0.25)))                   # new static domain
    assert traces == [TD, TimeDomain(32, 0.25)]
