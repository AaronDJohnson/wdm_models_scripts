"""S4 deferral (c): keyed combinators run compiled on JAX, one program per static key.

``FlatMarginal(SchurEdge(adaptive rows))`` -- the sinusoid JAX edge-exact default -- has a
static key (the rows' Chebyshev order ``P``, a host function of ``alpha_d``).  ``Covariance.
factor`` compiles one program per key, threads the key to ``SchurEdge`` through every
combinator, and runs staged rows (``rows.staged``) beforehand as their own compiled stages
(``prepare``), passing them in.  Before S4 the combinator ran eagerly (3.75 s per state at the
sinusoid grid against 0.5-0.7 s compiled).
"""

from __future__ import annotations

import numpy as np
import pytest
import wdm
from wdm.backend import NUMPY, is_tracer

from domainia.covariance import (ChannelStack, DiagonalCovariance, ExactTimeCovariance, PushedRows, SchurEdge,
                                 cheb_order)
from domainia.domains import CellSet
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance

pytestmark = pytest.mark.jax

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
PSD = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * DT)
CELLS = CellSet.all_active(wdm.Grid(NF, NT, DT))
EDGE = CELLS.columns(3, 1)
INTERIOR = CELLS.complement(EDGE)
U = FourierBandNuisance(N, DT, 1.0).push_to_cells(CELLS)
#: three noise states, two Chebyshev orders (the first two share one)
NUS = [np.array([-1.7, 0.9]), np.array([-1.5, 1.2]), np.array([-1.7, 0.3])]


def _edge(**kw):
    rows = PushedRows(ExactTimeCovariance(PSD, N, DT), None, EDGE, cells=CELLS, order="adaptive", **kw)
    return SchurEdge(DiagonalCovariance(BandAveragedVariance(PSD, INTERIOR), INTERIOR.size), rows, EDGE,
                     form="scaled")


@pytest.mark.parametrize("staged", [False, True])
def test_one_compiled_program_per_key(jax_be, staged, rng):
    import jax.numpy as jnp

    kw = {"phi_cache": True, "staged": True, "chunk": 8} if staged else {}
    fm = FlatMarginal(_edge(**kw), U)
    L_max = fm.cov.rows.L_max
    P = [cheb_order(nu[1], L_max) for nu in NUS]
    assert P[0] == P[1] != P[2]
    y = rng.standard_normal(N)
    for _ in range(2):                                       # repeats hit the compiled programs
        for nu in NUS:
            Fj, Fn = fm.factor(jnp.asarray(nu), jax_be), fm.factor(nu, NUMPY)
            assert type(Fj.logdet).__module__.startswith("jax")
            assert abs(float(Fj.logdet) - float(Fn.logdet)) <= 1e-12 * abs(float(Fn.logdet))
            assert abs(float(Fj.quad(jnp.asarray(y))) - float(Fn.quad(y))) <= 1e-11 * abs(float(Fn.quad(y)))
    assert sorted(fm.compiled_keys()) == sorted({((p, None),) for p in P})
    for key in fm.compiled_keys():
        assert fm._jit_cache["factor", key]._cache_size() == 1     # compiled once, never retraced


def test_rows_are_never_evaluated_on_concrete_arrays(jax_be, monkeypatch):
    """Unstaged rows run inside the compiled program (traced ``nu``); staged rows run as their
    own stages in ``prepare`` and ``evaluate`` is not called."""
    import jax.numpy as jnp

    calls = []
    real = PushedRows.evaluate

    def spy(self, C, nu, be, P=None):
        calls.append(("evaluate", is_tracer(nu)))
        return real(self, C, nu, be, P=P)

    real_staged = PushedRows.evaluate_staged

    def spy_staged(self, C, nu, be, P):
        calls.append(("staged", is_tracer(nu)))
        return real_staged(self, C, nu, be, P)

    monkeypatch.setattr(PushedRows, "evaluate", spy)
    monkeypatch.setattr(PushedRows, "evaluate_staged", spy_staged)
    FlatMarginal(_edge(), U).factor(jnp.asarray(NUS[0]), jax_be)
    assert calls == [("evaluate", True)]
    calls.clear()
    FlatMarginal(_edge(phi_cache=True, staged=True, chunk=8), U).factor(jnp.asarray(NUS[0]), jax_be)
    assert calls == [("staged", False)]


def test_jit_false_is_the_eager_reference(jax_be):
    import jax.numpy as jnp

    fm = FlatMarginal(_edge(), U)
    F = fm.factor(jnp.asarray(NUS[2]), jax_be, jit=False)
    assert fm.compiled_keys() == ()
    assert abs(float(F.logdet) - float(fm.factor(NUS[2], NUMPY).logdet)) <= 1e-12 * abs(float(F.logdet))


def test_channel_stack_threads_one_key_per_channel(jax_be):
    import jax.numpy as jnp

    st = ChannelStack((FlatMarginal(_edge(), U), FlatMarginal(_edge(), U)), 2)
    nu = np.concatenate([NUS[0], NUS[2]])
    key = st.state_key(nu)
    assert key == (st.covs[0].state_key(NUS[0]), st.covs[1].state_key(NUS[2])) and key[0] != key[1]
    Fj, Fn = st.factor(jnp.asarray(nu), jax_be), st.factor(nu, NUMPY)
    assert st.compiled_keys() == (key,)
    assert abs(float(Fj.logdet) - float(Fn.logdet)) <= 1e-12 * abs(float(Fn.logdet))
