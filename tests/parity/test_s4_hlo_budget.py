"""S4 deferral (c): a production-size HLO budget for the JAX likelihoods.

At the sinusoid grid (64 x 256) the lowered programs must not carry domainia's constants as
literals (spec section 3.3: constants are jit arguments).  The only large literals allowed
are ``wdm.forward_time_many``'s own grid tables (the gather index, parity and sign tables,
``phi``: ~567 KB of text at 64 x 256, as WG's ``_Forward`` had them -- recorded here as the
budget instead of changing part 1); every dense literal over 200 characters in a domainia
program must be one of them, and the program may add at most 20 KB of text.  The Whittle
form uses no WDM transform: no large literal at all (its retained-bin index was one before
S4: 130 KB).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.paths import GOLDEN

pytestmark = [pytest.mark.jax, pytest.mark.golden]

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def wdm_literals():
    import jax.numpy as jnp
    import wdm
    from wdm.backend import jax_backend

    from scripts.golden.s2_cases import DT, NF, NT, N
    from scripts.golden.s4_cases import big_literals

    be = jax_backend()
    text = jax.jit(lambda x: wdm.forward_time_many(x, NF, NT, DT, be=be)).lower(jnp.zeros((1, N))).as_text()
    return len(text), set(big_literals(text))


@pytest.mark.parametrize("model", ["band_raw", "tri_raw"])
def test_wdm_programs_carry_only_the_transform_tables(model, wdm_literals):
    import jax.numpy as jnp

    from scripts.golden.s2_cases import sinusoid_inputs
    from scripts.golden.s4_cases import big_literals, sinusoid_jax_likelihood

    wdm_chars, allowed = wdm_literals
    truth = sinusoid_inputs().truth
    lik = sinusoid_jax_likelihood(model)
    nu = lik.noise_slice_host(truth)
    state = lik.noise_state(nu)
    ev = jax.jit(lik._evaluate).lower(lik.parts, jnp.asarray(truth), state).as_text()
    ns = jax.jit(lik._noise_state).lower(lik.parts, lik.be.asarray(nu)).as_text()
    lits = big_literals(ev)
    assert lits, "the regex found no literal: the transform tables should be there"
    assert [len(x) for x in lits if x not in allowed] == []          # nothing from domainia
    assert big_literals(ns) == []                                     # the noise state: none at all
    assert len(ev) <= wdm_chars + 20_000, len(ev)
    assert len(ns) <= 50_000, len(ns)


def test_whittle_program_has_no_large_literal(wdm_literals):
    import jax.numpy as jnp

    from scripts.golden.goldens_io import read_json, to_float
    from scripts.golden.s2_cases import sinusoid_inputs
    from scripts.golden.s4_cases import big_literals, sinusoid_jax_whittle

    assert wdm_literals[1], "LITERAL matched nothing in forward_time_many's program: the regex is dead"
    truth = sinusoid_inputs().truth_whittle
    w = sinusoid_jax_whittle("whittle_raw")
    text = jax.jit(w.lnl_row).lower(w.parts, jnp.asarray(truth)).as_text()
    assert big_literals(text) == [] and len(text) <= 20_000, len(text)
    ref = np.load(GOLDEN / "sinusoid" / "references" / "manuscript_mid_whittle_raw_reference.npz")
    assert np.array_equal(ref["theta"][0], truth)
    want = float(ref["loglike"][0])
    g = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")["approved"]["jax"]["references"]["whittle_raw"]
    atol, rtol = to_float(g["atol"]), to_float(g["rtol"])                 # the approved JAX gate
    got = float(w(truth[None])[0])
    assert abs(got - want) <= atol + rtol * abs(want), (got, want)
