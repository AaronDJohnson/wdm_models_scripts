"""F-D8 (S7): the GB JAX noise states and programs at production size.

* Per-state memory: one cached noise state of the release (both channels: the sum of its leaves'
  ``nbytes``, what one LRU slot holds) against the legacy class's at the anchor
  (``scripts.golden.gb_jax_legacy.LEGACY_JAX``, measured by lineage C): within 1.5x for all 23
  cached targets (measured 1.0000x: the release holds the legacy arrays -- ``MatrixCache``
  ``(L, y)``, ``EdgeSchurCache`` ``(1/v_I or v_I, B, L_S, y)``, ``FullCache`` ``(L or the NT/2
  blocks, y)`` -- plus 19-55 bytes of scalars).  A floor of 0.5x makes a measure that misses
  leaves fail.  Negative control: the NumPy column on JAX (``strategy="reference"``: the dense
  assembled band_exact matrix instead of the Schur complement) is 25x the legacy state.
* Programs: every GB JAX program carries no dense literal over 200 characters (constants are jit
  arguments, spec section 3.3: F-D2 removed the last ones, ``ghat``'s lag tables) and stays within
  80,000 characters (row / evaluation programs; ~65 KB of it is binary_galactica's template) or
  70,000 (noise-state programs; at most 60.1 KB, 3-month band_exact).

Default lane: gb_3mo; the 6- and 12-month targets are ``slow`` (12-month states are 111 MiB).
"""

from __future__ import annotations

import gc
import re

import pytest

jax = pytest.importorskip("jax", reason="the S7 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

import jax.numpy as jnp  # noqa: E402

from scripts.golden.gb_jax_legacy import LEGACY_JAX  # noqa: E402
from scripts.golden.s4_cases import LITERAL, big_literals, state_bytes  # noqa: E402
from wdm_models.galactic_binary.prepared import PreparedCase  # noqa: E402
from wdm_models.options import ModelOptions  # noqa: E402
from wdm_models.registry import build  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
GOLDEN_OPTIONS = ModelOptions(waveform_backend="fourier")
EVAL_BUDGET, STATE_BUDGET = 80_000, 70_000


def _cached():
    return [pytest.param(c, m, id=f"{c}-{m}", marks=[] if c == "gb_3mo" else [pytest.mark.slow])
            for c, models in LEGACY_JAX.items() for m, r in models.items() if r["state_bytes"] is not None]


@pytest.mark.parametrize("case, model", _cached())
def test_noise_state_memory_is_within_1p5x_of_legacy(case, model, record_property):
    lik = build("galactic_binary", case, model, backend="jax", options=GOLDEN_OPTIONS)
    st = lik.state(lik.lik.noise_slice_host(PreparedCase.load(case).anchor(model)))
    assert st is not None and bool(st.ok)
    ratio = state_bytes(st) / LEGACY_JAX[case][model]["state_bytes"]
    record_property("s7", f'{{"item": "galactic_binary/{case}/{model}/jax/state", "ratio": {ratio!r}}}')
    assert 0.5 <= ratio <= 1.5, ratio
    del lik, st
    gc.collect()


def test_the_memory_gate_bites():
    """The NumPy column on JAX (``strategy="reference"``): band_exact's state is the dense
    assembled matrix (2 x 720 x 720 floats), 25x the legacy ``EdgeSchurCache`` state."""
    prep = PreparedCase.load("gb_3mo")
    lik = build("galactic_binary", "gb_3mo", "band_exact", backend="jax",
                options=ModelOptions(waveform_backend="fourier", strategy="reference"))
    st = lik.noise_state(lik.noise_slice_host(prep.anchor("band_exact")))
    assert state_bytes(st) / LEGACY_JAX["gb_3mo"]["band_exact"]["state_bytes"] > 20


@pytest.mark.parametrize("case, model", [
    *(("gb_3mo", m) for m in ("whittle_raw", "whittle_cut", "band_raw", "tri_raw", "band_pad", "tri_pad",
                              "band_exact", "tri_exact", "exact")),
    *(pytest.param("gb_12mo", m, marks=pytest.mark.slow) for m in ("tri_raw", "tri_pad", "exact"))])
def test_gb_programs_carry_no_large_literal(case, model):
    prep = PreparedCase.load(case)
    lik = build("galactic_binary", case, model, backend="jax", options=GOLDEN_OPTIONS).lik
    theta = prep.anchor(model)
    if model.startswith("whittle"):
        texts = {"row": jax.jit(lik.lnl_row).lower(lik.parts, jnp.asarray(theta)).as_text()}
    else:
        nu = lik.noise_slice_host(theta)
        state = lik.noise_state(nu)
        texts = {"evaluate": jax.jit(lik._evaluate).lower(lik.parts, jnp.asarray(theta), state).as_text(),
                 "noise_state": jax.jit(lik._noise_state).lower(lik.parts, lik.be.asarray(nu)).as_text()}
    for kind, text in texts.items():
        assert re.search(LITERAL, text), "LITERAL matched nothing: the regex is dead"
        assert big_literals(text) == [], (kind, [len(x) for x in big_literals(text)])
        assert len(text) <= (STATE_BUDGET if kind == "noise_state" else EVAL_BUDGET), (kind, len(text))
