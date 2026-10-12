"""Package-wide test configuration for ``wdm-paper``.

Puts ``tests/oracle`` on ``sys.path`` so the frozen oracle imports as the
top-level package ``vendored_2f7136c`` (never as ``wdm``).
"""

from __future__ import annotations

import importlib.abc
import os
import sys
from pathlib import Path


class _HideJax(importlib.abc.MetaPathFinder):
    """CI lane 1 (NumPy only): with WDM_TESTS_HIDE_JAX=1, jax cannot be imported,
    so every JAX test skips and every NumPy test must still pass."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.split(".")[0] in ("jax", "jaxlib"):
            raise ModuleNotFoundError(f"{name} hidden by WDM_TESTS_HIDE_JAX=1 (NumPy-only lane)", name=name)
        return None


if os.environ.get("WDM_TESTS_HIDE_JAX") == "1":
    sys.meta_path.insert(0, _HideJax())

ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))
import numpy as np  # noqa: E402
import pytest  # noqa: E402


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
    return np.random.default_rng(20261011)
