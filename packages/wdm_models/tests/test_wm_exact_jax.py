"""E-D3: the bandcov log-det correction's two evaluation orders (spec section 5.1, S4).

``chebval2d`` is WT ``BandCovariance.logdet`` (``numpy.polynomial.chebyshev.chebval2d`` on host
floats; the NumPy golden, refuses a tracer); ``table`` is WG ``BandFullCovarianceGPU._correction``
(``gpu_sinusoid/wdmtoysgpu/bandcov.py:87-103``: the Chebyshev tables by recurrence and one
einsum; traceable, the JAX golden).  They agree to ~7e-15 relative over the prior box
(measured 6.6e-15 at 205 points); the table order under ``jax.jit`` agrees with itself on
NumPy to <= 1e-11 absolute (|c| <= 800).  Its 40 x 64 coefficients are a covariance
constant (``ProjectedOperatorMarginal`` carries ``BandCorrection.host_constants()``), so the
compiled noise state takes them as a jit argument (spec section 3.3): before S4 they were a
42,644-character literal of it.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from wdm.backend import NUMPY
from wdm_models.sinusoid import config as C
from wdm_models.sinusoid import exact as E

pytestmark = pytest.mark.golden
CFG = C.config("manuscript_mid")


def _points() -> np.ndarray:
    rng = np.random.default_rng(0)
    inner = np.column_stack([rng.uniform(-4.0, 0.0, 200), rng.uniform(-2.0, 2.0, 200)])
    return np.vstack([inner, [[-4.0, -2.0], [-4.0, 2.0], [0.0, -2.0], [0.0, 2.0], [-2.0, 1.2]]])


def test_chebval2d_is_the_default_and_unknown_orders_raise():
    assert E.BandCorrection.load(CFG).algorithm == "chebval2d"
    assert E.ProjectedExact.load(CFG).correction.algorithm == "chebval2d"
    assert E.ProjectedExact.load(CFG, algorithm="table").correction.algorithm == "table"
    with pytest.raises(ValueError, match="algorithm"):
        E.BandCorrection.load(CFG, algorithm="clenshaw2")


def test_the_table_order_equals_the_wt_order_on_numpy():
    wt, wg = E.BandCorrection.load(CFG), E.BandCorrection.load(CFG, algorithm="table")
    worst = 0.0
    for nu in _points():
        a, b = float(wt(nu)), float(wg(nu, NUMPY))
        worst = max(worst, abs(a - b) / abs(a))
    assert worst <= 1e-13, worst


def test_the_table_order_is_traceable_under_jit(jax_be):
    import jax

    wg = E.BandCorrection.load(CFG, algorithm="table")
    f = jax.jit(lambda nu: wg(nu, jax_be))
    for nu in _points()[::20]:
        assert abs(float(f(jax_be.asarray(nu))) - float(wg(nu, NUMPY))) <= 1e-11


def test_the_wt_order_runs_eagerly_on_jax_and_refuses_tracing(jax_be):
    """``strategy="reference"`` on JAX evaluates WT's order on concrete JAX values (the
    S3 NumPy-only guard is lifted); under ``jit`` it cannot be traced."""
    import jax

    wt = E.BandCorrection.load(CFG)
    nu = np.array([-2.0, 1.2])
    assert float(wt(jax_be.asarray(nu), jax_be)) == float(wt(nu))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda x: wt(x, jax_be))(jax_be.asarray(nu))


def test_the_table_coefficients_are_a_jit_argument(jax_be):
    import jax

    pe = E.ProjectedExact.load(CFG, algorithm="table")
    cov = pe.covariance(on_fail="nan")
    C = cov.constants(jax_be)
    assert np.array_equal(np.asarray(C["correction"]["coefficients"]), pe.correction.coefficients)
    nu = np.array([-2.0, 1.2])
    f = jax.jit(lambda C, nu: cov._factor(C, nu, jax_be).logdet)
    text = f.lower(C, jax_be.asarray(nu)).as_text()
    literals = re.findall(r'dense<("0x[0-9A-Fa-f]*"|[^>]*)>', text)
    assert literals and max(len(x) for x in literals) <= 200, max(len(x) for x in literals)
    want = pe.logdet(nu) + float(pe.correction(nu, NUMPY))
    assert abs(float(f(C, jax_be.asarray(nu))) - want) <= 1e-11 * abs(want)
