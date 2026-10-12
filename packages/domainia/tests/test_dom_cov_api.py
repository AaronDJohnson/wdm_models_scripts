"""D-C11: the covariance package's public surface (section D-C gate).

* ``domainia.covariance`` exports exactly the documented names; importing it (and
  ``domainia.nuisance``) never imports JAX.
* Every factor class satisfies the ``Factor`` protocol.
* Under JAX every factor is a pytree: it flattens, round-trips, and two factors stack
  along a rung axis into one ``vmap`` (spec §3.3).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import domainia.covariance as cov
from domainia.nuisance import MarginalFactor, ProjectedFactor

EXPORTS = {
    "BlockTriCovFactor", "BlockTridiagonalCovariance", "CHEB_TOL", "ChannelStack", "CholFactor", "CovOperator",
    "Covariance", "DenseCovariance", "DiagFactor", "DiagonalCovariance", "Eq45Covariance", "Evaluator",
    "ExactFourierCovariance", "ExactTimeCovariance", "Factor", "HermitianCovariance", "HermitianFactor",
    "OperatorFactor", "PadMask", "PaddedFactor", "PushedRows", "QuadForm", "SchurEdge", "SchurFactor",
    "StackFactor", "ZeroPadded", "as_evaluator", "assemble_edge_rows", "bessel_coefficients", "cheb_order",
    "cheb_tail", "cut", "dense_factor", "ghat", "keep_mask", "positions", "quad_form", "schur_factor",
    "window_bins",
}
SRC = Path(__file__).resolve().parents[1] / "src"
WDM_SRC = Path(__file__).resolve().parents[2] / "wdm" / "src"


def test_exports_are_the_documented_names():
    assert set(cov.__all__) == EXPORTS
    for name in EXPORTS:
        assert getattr(cov, name) is not None


def test_import_does_not_import_jax():
    code = ("import sys; import domainia.covariance, domainia.nuisance; "
            "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if m.startswith('jax'))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(WDM_SRC)]), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-P", "-c", code], check=True, env=env, cwd=str(SRC.parent))


@pytest.mark.parametrize("cls", [cov.DiagFactor, cov.CholFactor, cov.BlockTriCovFactor, cov.HermitianFactor,
                                 cov.SchurFactor, cov.PaddedFactor, cov.OperatorFactor, cov.StackFactor,
                                 MarginalFactor, ProjectedFactor])
def test_factor_classes_follow_the_protocol(cls):
    for attr in ("solve", "whiten", "quad"):
        assert callable(getattr(cls, attr)), (cls.__name__, attr)
    assert "n_u" in dir(cls)


@pytest.mark.jax
def test_factors_are_pytrees_that_stack(jax_be):
    import jax
    import jax.numpy as jnp

    v1, v2 = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    c = cov.DiagonalCovariance(lambda nu, be: be.asarray(np.ones(3)) * be.xp.exp(nu[0]), 3)
    F1, F2 = (c.factor(jnp.asarray([np.log(s), 0.0])) for s in (1.0, 2.0))
    leaves, tree = jax.tree_util.tree_flatten(F1)
    assert isinstance(jax.tree_util.tree_unflatten(tree, leaves), cov.DiagFactor)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), F1, F2)
    b = jnp.ones(3)
    q = jax.vmap(lambda F: F.quad(b))(stacked)
    assert np.allclose(np.asarray(q), [3.0, 1.5], rtol=1e-15)
    D = cov.DenseCovariance(np.diag(v1), 3, chol_impl="numpy")
    G = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), D.factor(jnp.zeros(1)),
                               cov.DenseCovariance(np.diag(v2), 3, chol_impl="numpy").factor(jnp.zeros(1)))
    assert np.allclose(np.asarray(jax.vmap(lambda F: F.quad(b, form="whiten"))(G)), [1.75, 1.5], rtol=1e-15)
