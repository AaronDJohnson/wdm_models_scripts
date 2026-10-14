"""Test configuration for ``wdm_models``.

This directory has no ``__init__.py``: in a root ``pytest`` run ``packages/wdm/tests`` is
already the package ``tests``, so every file here is a top-level module with a unique
``test_wm_*`` name.  Goldens are read through the harness helpers
(``scripts/golden/workers/_common.py``), loaded by file path so neither ``sys.path`` nor
``PYTHONPATH`` changes and nothing under ``scripts/`` is imported as a package (the
dependency direction ``wdm <- domainia <- wdm_models <- scripts`` holds for the tests too).
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
PKG = TESTS.parent                                     # packages/wdm_models
REL = PKG.parents[1]                                   # wdm_models_scripts
GOLDEN = REL / "golden"
DATA = REL / "data"
COMMON = REL / "scripts" / "golden" / "workers" / "_common.py"
SIN_GOLDEN = GOLDEN / "sinusoid"


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


@pytest.fixture
def rng():
    return np.random.default_rng(20261013)


@pytest.fixture(scope="session")
def golden_io():
    """The harness's ``read_json`` / ``to_float`` / ``runtime_fingerprint`` / ``sha256_*``
    (file-path import).  Missing helpers FAIL the golden tests (a gate never passes by being
    skipped); outside the release repository deselect them with ``-m "not golden"``."""
    if not COMMON.is_file():
        pytest.fail(f"golden harness helpers not found at {COMMON}; deselect the golden tests "
                    "with -m 'not golden' outside the release repository")
    spec = importlib.util.spec_from_file_location("wdm_models_tests_golden_common", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def same_build(golden_io):
    """``same_build(recorded)``: True when this process matches a golden's recording build
    on every key the record has among machine / python / numpy / scipy / blas.  There the
    NumPy path must be bitwise; elsewhere the approved gate applies."""
    current = golden_io.runtime_fingerprint()

    def check(recorded: dict) -> bool:
        keys = [k for k in ("machine", "python", "numpy", "scipy", "blas") if k in recorded]
        if not keys:
            raise AssertionError("the golden records no build fingerprint")
        return all(recorded[k] == current.get(k) for k in keys)

    return check


@pytest.fixture(scope="session")
def g6(golden_io):
    """golden/sinusoid/intermediates/manuscript_mid_truth.json (G6 at the truth)."""
    return golden_io.read_json(SIN_GOLDEN / "intermediates" / "manuscript_mid_truth.json")


@pytest.fixture(scope="session")
def recording_build(g6, same_build):
    """True on the build that recorded the sinusoid goldens (G1 lineage A is bitwise there)."""
    return same_build(g6["runtime"])


@pytest.fixture(scope="session")
def g1():
    """``g1(case, model) -> {"theta", "loglike", "near"}`` from golden/sinusoid/references."""
    def load(case: str, model: str) -> dict[str, np.ndarray]:
        path = SIN_GOLDEN / "references" / f"{case}_{model}_reference.npz"
        with np.load(path, allow_pickle=False) as z:
            return {k: np.array(z[k]) for k in z.files}

    return load


@pytest.fixture(scope="session")
def numpy_gate(golden_io):
    """``numpy_gate(model) -> (atol, rtol)``: the approved NumPy G1 gate (2026-10-12,
    golden/sinusoid/tolerances_measured.json ``approved.numpy.references``)."""
    approved = golden_io.read_json(SIN_GOLDEN / "tolerances_measured.json")["approved"]

    def gate(model: str) -> tuple[float, float]:
        g = approved["numpy"]["references"][model]
        return golden_io.to_float(g["atol"]), golden_io.to_float(g["rtol"])

    return gate


@pytest.fixture(scope="session")
def facts(golden_io):
    """golden/sinusoid/model_facts.json (E-A2): the oracle's configs, prior tables, noise
    bounds, signal sha256, PSD sample and the 45 final runs' prior records."""
    return golden_io.read_json(SIN_GOLDEN / "model_facts.json")


# --- galactic binary (S5, section F-A) ---------------------------------------------------------
GB_GOLDEN = GOLDEN / "galactic_binary"
#: The keys of a golden ``build`` record that decide whether templates/PSDs may be bitwise
#: (binary_galactica runs on JAX, so jax/jaxlib join the NumPy build keys).
GB_BUILD_KEYS = ("machine", "python", "numpy", "scipy", "jax", "jaxlib")


@pytest.fixture
def bg():
    """``binary_galactica`` with float64 JAX on; skips in the NumPy-only lane (bg is JAX-only)."""
    _jax_or_skip()
    return pytest.importorskip("binary_galactica", reason="binary_galactica not installed")


@pytest.fixture
def s5(record_property):
    """``s5(item, observed, gate)``: one ``s5`` junit property ``{"item", "observed", "gate"}``
    (the S5 report's records; the ``s3``/``s4`` convention)."""
    import json

    def rec(item: str, observed: float, gate: float) -> None:
        record_property("s5", json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))

    return rec


@pytest.fixture(scope="session")
def gb_products(golden_io):
    """golden/galactic_binary/products.json (G5, lineage C: templates, describe counts,
    snr_full, prior boxes, the envelope's C and Table III, the constants)."""
    return golden_io.read_json(GB_GOLDEN / "products.json")


@pytest.fixture(scope="session")
def gb_recording_build(gb_products, golden_io):
    """True when this process matches products.json's build on every :data:`GB_BUILD_KEYS`."""
    current = golden_io.runtime_fingerprint()
    rec = gb_products["build"]
    return all(rec.get(k) == current.get(k) for k in GB_BUILD_KEYS)


@pytest.fixture(scope="session")
def gb_g6(golden_io):
    """``gb_g6(case) -> (json, npz arrays)``: golden/galactic_binary/intermediates_<case>."""
    def load(case: str):
        with np.load(GB_GOLDEN / f"intermediates_{case}.npz", allow_pickle=False) as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        return golden_io.read_json(GB_GOLDEN / f"intermediates_{case}.json"), arrays

    return load


@pytest.fixture(scope="session")
def gb_psd_window():
    """golden/galactic_binary/psd_window.npz: lineage C's ``prepare.psd_components`` (the
    ``_nora`` instrument PSD and ``fit_foreground.s_fg_glass``) on each window's bins."""
    with np.load(GB_GOLDEN / "psd_window.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}
