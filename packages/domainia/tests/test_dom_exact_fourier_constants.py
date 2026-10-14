"""F-D2 (S7): ``ExactFourierCovariance``'s lag tables are constants, never program literals.

``M'`` reads ``ghat`` at the ``2n - 1`` lags ``-(n-1) .. n-1``: the gather index ``|lag|`` and
the conjugation mask ``lag < 0`` depend on the bins alone, so they are host constants passed to
``jit`` as arguments (spec section 3.3).  Built inside the program they were dense literals of
the GB noise-state programs: 11.5 KB of text for ``exact`` at 3 months (360 bins), 13.4 KB for the
exact-edge rows (419 window bins).  The values are unchanged bit for bit (the same gather).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import domtest_inputs as T
from domainia.covariance import ExactFourierCovariance, HermitianCovariance
from domainia.covariance.exact_fourier import ghat

pytestmark = pytest.mark.golden
#: A dense literal of a lowered StableHLO program (``scripts/golden/s4_cases.LITERAL``).
LITERAL = r'dense<("0x[0-9A-Fa-f]*"|[^>]*)>'


def _reference(ef, mu, be):
    """``M'`` with the lag tables built in place (the pre-F-D2 arithmetic, GB ``full_covariance``)."""
    C = ef.host_constants()
    xp = be.xp
    gh = ghat(be.asarray(mu), be.asarray(C["basis_L"]), np.arange(-(ef.n - 1), ef.n), be)
    root = xp.sqrt(be.asarray(C["sG"]))
    return xp.diag(be.asarray(C["sI"]).astype(complex)) + (root[:, None] * root[None, :]) * xp.conj(gh)[C["lag"]]


def test_the_lag_tables_are_host_constants():
    c = T.gb_case("gb_3mo")
    ef = ExactFourierCovariance(c.psd, c.nominal, channel=0)
    C = ef.host_constants()
    lags = np.arange(-(ef.n - 1), ef.n)
    assert C["absl"].dtype == np.int64 and np.array_equal(C["absl"], np.abs(lags))
    assert C["neg"].dtype == np.bool_ and np.array_equal(C["neg"], lags < 0)


def test_mprime_is_unchanged_bit_for_bit(backend):
    c = T.gb_case("gb_3mo")
    for ch in range(2):
        ef = ExactFourierCovariance(c.psd, c.nominal, channel=ch)
        C = {k: backend.asarray(v) for k, v in ef.host_constants().items()}
        got = np.asarray(ef.evaluate(C, backend.asarray(c.mu[ch]), backend))
        assert np.array_equal(got, np.asarray(_reference(ef, c.mu[ch], backend))), ch


def test_the_jax_factor_program_carries_no_large_literal(jax_be):
    import jax

    c = T.gb_case("gb_3mo")
    hc = HermitianCovariance(ExactFourierCovariance(c.psd, c.nominal, channel=0))
    text = jax.jit(hc._factor, static_argnums=2).lower(hc.constants(jax_be), jax_be.asarray(c.mu[0]),
                                                       jax_be).as_text()
    big = [len(m.group(1)) for m in re.finditer(LITERAL, text) if len(m.group(1)) > 200]
    assert big == [], big
    assert re.search(LITERAL, text), "LITERAL matched nothing: the regex is dead"
