"""Test configuration for ``domainia``.

This directory has no ``__init__.py``: in a root ``pytest`` run ``packages/wdm/tests`` is
already the package ``tests``, so every file here is a top-level module with a unique
``test_dom_*`` name.  Goldens are read through the harness helpers
(``scripts/golden/workers/_common.py``, re-exported by ``scripts.golden.goldens_io``),
loaded by file path so neither ``sys.path`` nor ``PYTHONPATH`` changes.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


class _HideJax(importlib.abc.MetaPathFinder):
    """NumPy-only lane: with WDM_TESTS_HIDE_JAX=1 jax cannot be imported, so every JAX
    test skips and every NumPy test must still pass (as packages/wdm/tests)."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.split(".")[0] in ("jax", "jaxlib"):
            raise ModuleNotFoundError(f"{name} hidden by WDM_TESTS_HIDE_JAX=1 (NumPy-only lane)",
                                      name=name)
        return None


if os.environ.get("WDM_TESTS_HIDE_JAX") == "1" and not any(
        isinstance(f, _HideJax) or type(f).__name__ == "_HideJax" for f in sys.meta_path):
    sys.meta_path.insert(0, _HideJax())

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent                                     # packages/domainia
REL = PKG.parents[1]                                   # wdm_models_scripts
GOLDEN = REL / "golden"
DATA = REL / "data"
COMMON = REL / "scripts" / "golden" / "workers" / "_common.py"


def _jax_or_skip():
    jax = pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    from wdm.backend import enable_x64

    enable_x64()
    return jax


@pytest.fixture
def jax_be():
    """The JAX backend with float64 enabled; skips when jax is not importable."""
    _jax_or_skip()
    from wdm.backend import jax_backend

    return jax_backend()


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    """Both backends (the JAX leg skips when jax is not importable)."""
    if request.param == "jax":
        _jax_or_skip()
        from wdm.backend import jax_backend

        return jax_backend()
    from wdm.backend import NUMPY

    return NUMPY


@pytest.fixture
def rng():
    return np.random.default_rng(20261012)


@pytest.fixture(scope="session")
def golden_io():
    """The harness's ``read_json`` / ``to_float`` / ``runtime_fingerprint`` (file-path import)."""
    if not COMMON.is_file():
        pytest.skip(f"golden harness helpers not found at {COMMON}")
    spec = importlib.util.spec_from_file_location("domainia_tests_golden_common", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def same_build(golden_io):
    """``same_build(recorded)``: True when this process matches a golden's recording build
    on every key the record has among machine / python / numpy / scipy / blas.  There the
    NumPy path must be bitwise; elsewhere the G6 tolerance applies."""
    current = golden_io.runtime_fingerprint()

    def check(recorded: dict) -> bool:
        keys = [k for k in ("machine", "python", "numpy", "scipy", "blas") if k in recorded]
        return all(recorded[k] == current.get(k) for k in keys)

    return check
