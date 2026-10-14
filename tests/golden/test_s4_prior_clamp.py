"""E-D4 (S4): the JAX registry build -- its evaluator per model, prior sanitising, the alpha_d
clamp and the exact model's validity mask on CPU-JAX (spec sections 3.4, 3.5 and 4.7; WG
``run.py:135-178``, ``matrix.py:192, 635-637``, ``bandcov.py:117-126``).

* ``registry.build(..., backend="jax")`` is the domainia likelihood inside WG's evaluator
  (``NoiseStateCache`` for the cached models, ``BatchedLikelihood`` otherwise); NumPy stays
  the bare likelihood.  Its composition is E-C's WG-shaped one (``s4_cases``), byte for byte.
* ``sanitize(build(...))``: rows outside the prior (or non-finite) are replaced by the
  truth before the device call and come back ``-inf``; the phase is wrapped first, so a
  phase outside [-pi, pi) gives the wrapped row's value.
* ``noise_clamp``: on JAX the noise slice's alpha_d is clipped to [-4, 4] before the
  factorisation and in the cache key, so alpha_d = 5 IS alpha_d = 4 (same bytes, a cache
  hit); NumPy (WT) has no clamp.
* the exact model's ``row_ok``: outside the bandcov box (or |ln A_n| >= 700) a row is
  ``-inf`` without a PD flag, as WG's ``valid`` mask; its compiled noise state takes the
  correction's coefficients as a jit argument.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the S4 JAX gates need jax")

from wdm.backend import enable_x64  # noqa: E402

enable_x64()

from domainia.likelihood import GaussianLikelihood  # noqa: E402
from domainia.sampling import BatchedLikelihood, NoiseStateCache  # noqa: E402
from scripts.golden.goldens_io import read_json, read_npz  # noqa: E402
from scripts.golden.paths import GOLDEN, REL, tmp_root  # noqa: E402
from wdm_models.options import ModelOptions, resolve  # noqa: E402
from wdm_models.registry import build  # noqa: E402
from wdm_models.sampling.prior_sanitize import sanitize  # noqa: E402
from wdm_models.sinusoid import models as M  # noqa: E402
from wdm_models.sinusoid.config import config  # noqa: E402
from wdm_models.sinusoid.data import load  # noqa: E402
from wdm_models.sinusoid.exact import ProjectedExact  # noqa: E402

pytestmark = [pytest.mark.golden, pytest.mark.jax]
CASE = "manuscript_mid"


def _truth(model: str) -> np.ndarray:
    return config(CASE).truth(model)


def test_the_build_is_wg_s_evaluator_on_jax_and_bare_on_numpy():
    for model, kind in (("band_raw", BatchedLikelihood), ("tri_raw", NoiseStateCache),
                        ("whittle_raw", BatchedLikelihood)):
        lik = build("sinusoid", CASE, model, backend="jax")
        assert type(lik) is kind, model
        assert lik.lik.be.name == "jax" and lik.ndim == len(_truth(model))
    tri = build("sinusoid", CASE, "tri_raw", backend="jax")
    assert tri.capacity == 32 and tri.batch_rows is False          # the CPU column
    assert build("sinusoid", CASE, "band_raw", backend="jax").count_states is False
    assert type(build("sinusoid", CASE, "tri_raw", backend="numpy")) is GaussianLikelihood


@pytest.mark.parametrize("model", ["tri_raw", "band_pad", pytest.param("band_exact", marks=pytest.mark.slow),
                                   pytest.param("tri_exact", marks=pytest.mark.slow)])
def test_the_jax_build_is_e_c_s_wg_composition(model):
    """E-C's wiring contract (``scripts/golden/s4_cases.py::sinusoid_jax_likelihood``, the
    WG-shaped JAX states it measured): the registry builds the same composition -- the same
    bytes at G1 points 0-2 -- and its noise state is within 1.5x WG's per-state memory
    (spec section 7.6 S4; WG 9.3 / 120 / 81 MiB for band_pad / band_exact / tri_raw)."""
    from scripts.golden.s4_cases import WG_STATE_MIB, sinusoid_jax_likelihood, state_bytes

    X = np.asarray(read_npz(GOLDEN / "sinusoid" / "references" / f"{CASE}_{model}_reference.npz")["theta"][:3])
    reg = build("sinusoid", CASE, model, backend="jax")
    cov = reg.lik.covs[0]
    if model == "band_pad":
        assert cov.form == "reduce" and cov.cov.form == "reduce"         # FlatMarginal(ZeroPadded)
    if model.endswith("_exact"):
        assert cov.cov.form == ("scaled" if model == "band_exact" else "solve")    # SchurEdge
    want = NoiseStateCache(sinusoid_jax_likelihood(model), 32, batch_rows=False)(X)
    assert np.array_equal(reg(X), want)
    if model in WG_STATE_MIB:
        mib = state_bytes(reg.lik.noise_state(reg.lik.noise_slice_host(X[0]))) / 2**20
        assert mib <= 1.5 * WG_STATE_MIB[model], (model, mib)


def test_the_gpu_column_can_batch_every_model_it_batches():
    """On a GPU, E-A8 batches the rows of one call for the cached models other than
    edge-exact, and E-C4's ``NoiseStateCache`` refuses ``batch_rows=True`` on a likelihood that
    is not ``batchable``: every such model must be batchable.  Built here on the CPU with the
    GPU column (``resolve(..., device="gpu")``), so a wrong ``batchable`` fails before Colab."""
    from wdm_models.sinusoid.evaluation import CACHED, EDGE_EXACT_MODELS, wrap

    s = load(CASE)
    batched = [m for m in CACHED if m not in EDGE_EXACT_MODELS]
    assert batched == ["tri_raw", "tri_cut", "band_pad", "mid_pad", "tri_pad"]
    for model in batched:
        r = resolve("sinusoid", model, "jax", device="gpu")
        assert r.batch_rows is True and r.cache_size == 16 and r.exact_chunk == 192, model
        lik = M.likelihood(model, s.cfg, s.d, resolved=r)
        assert lik.batchable is True, model
        w = wrap(lik, model, r)
        assert isinstance(w, NoiseStateCache) and w.batch_rows is True and w.capacity == 16, model


@pytest.mark.parametrize("model", ["band_raw", "tri_raw", "exact"])
def test_out_of_prior_rows_are_replaced_by_the_truth_and_masked(model):
    lik = build("sinusoid", CASE, model, backend="jax")
    s = sanitize(lik, "sinusoid", CASE, model)
    truth = _truth(model)
    X = np.repeat(truth[None, :], 5, axis=0)
    X[1, 5] = 3.0                                  # alpha_d outside U[-2, 2]
    X[2, 1] = np.nan                               # non-finite f0
    X[3, 2] = truth[2] + 2.0 * np.pi               # phase outside [-pi, pi): wrapped, not rejected
    X[4, 3] = 4.5                                  # log10 A_n outside U[1, 4]
    out = s(X)
    assert np.isneginf(out[[1, 2, 4]]).all() and np.isfinite(out[[0, 3]]).all()
    assert abs(out[3] - out[0]) <= 1e-9 * abs(out[0])           # the wrapped phase is the truth's
    assert not lik.last_failed.any()                           # masked rows are not PD failures
    assert lik.counters()["evaluations"] == 5
    if model == "tri_raw":                                      # one state: every row is the truth's noise
        assert lik.counters()["noise_states"] == 1 and lik.counters()["cache_hits"] == 4


@pytest.mark.parametrize("model", ["band_raw", "tri_raw"])
def test_jax_clamps_alpha_d_and_numpy_does_not(model):
    truth = _truth(model)
    X = np.repeat(truth[None, :], 2, axis=0)
    X[0, 5], X[1, 5] = 4.0, 5.0
    jx = build("sinusoid", CASE, model, backend="jax")
    assert jx.lik.noise_slice_host(X[1]).tolist() == [truth[4], 4.0]
    v = jx(X)
    assert v[0] == v[1]                                         # bitwise: the same noise state
    if model == "tri_raw":
        assert jx.counters()["noise_states"] == 1 and jx.counters()["cache_hits"] == 1
    npy = build("sinusoid", CASE, model, backend="numpy")
    assert npy.noise_slice_host(X[1]).tolist() == [truth[4], 5.0]


def test_exact_row_ok_masks_rows_outside_the_bandcov_box():
    lik = build("sinusoid", CASE, "exact", backend="jax")
    truth = _truth("exact")
    X = np.repeat(truth[None, :], 4, axis=0)
    X[1, 4] = 0.5                                  # alpha_c above the box [-4, 0]
    X[2, 5] = -2.5                                 # alpha_d below the box [-2, 2] (inside the clamp)
    X[3, 3] = 400.0                                # |ln A_n| > 700
    out = lik(X)
    assert np.isfinite(out[0]) and np.isneginf(out[1:]).all()
    assert not lik.last_failed.any()               # a validity mask, not a PD failure
    assert build("sinusoid", CASE, "exact", backend="numpy").row_ok is None    # WT raised instead


def test_the_jax_exact_noise_state_takes_its_tables_as_arguments():
    """Spec 3.3 for the exact model: the bandcov coefficients (E-D3) and every other constant
    are jit arguments of the compiled noise state -- no dense literal over 200 characters
    (before E-D3 the 40 x 64 coefficients were a 42,644-character literal)."""
    from scripts.golden.s4_cases import big_literals

    lik = build("sinusoid", CASE, "exact", backend="jax").lik
    nu = lik.noise_slice_host(_truth("exact"))
    text = jax.jit(lik._noise_state).lower(lik.parts, lik.be.asarray(nu)).as_text()
    assert big_literals(text) == [] and len(text) <= 80_000, len(text)


def test_the_exact_assets_must_match_the_correction_order():
    s = load(CASE)
    r = resolve("sinusoid", "exact", "jax", device="cpu")
    with pytest.raises(ValueError, match="correction"):
        M.likelihood("exact", s.cfg, s.d, resolved=r, exact=ProjectedExact.load(s.cfg))


@pytest.mark.parametrize("model", ["tri_raw", "exact"])
def test_reference_strategy_runs_the_numpy_algorithms_on_jax(model):
    """``strategy="reference"`` (spec 3.2): bessel + ragged (eager), fixed rows, WT PCG
    (``on_fail="raise"``, eager) and the ``chebval2d`` correction, on JAX arrays; neither is
    jit-able there, so both run through the eager noise-state LRU."""
    ref = read_npz(GOLDEN / "sinusoid" / "references" / f"{CASE}_{model}_reference.npz")
    gate = read_json(GOLDEN / "sinusoid" / "tolerances_measured.json")["approved"]["jax"]["references"][model]
    lik = build("sinusoid", CASE, model, backend="jax", options=ModelOptions(strategy="reference"))
    assert isinstance(lik, NoiseStateCache) and lik.lik.jittable is False and lik.lik.be.name == "jax"
    got = lik(ref["theta"][:2])
    assert ref["near"][:2].all() and np.all(np.abs(got - ref["loglike"][:2]) <= gate["atol"])


def test_the_jax_build_refuses_float32():
    """Every JAX constant is built under ``require_x64`` (spec 3.3): with x64 off the build
    raises before any constant exists."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(JAX_ENABLE_X64="false", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(str(REL / p) for p in (".", "packages/wdm/src", "packages/domainia/src",
                                                                "packages/wdm_models/src")))
    code = ("from wdm_models.registry import build\n"
            "build('sinusoid', 'manuscript_mid', 'band_raw', backend='jax')\n")
    p = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(tmp_root()), capture_output=True,
                       text=True, check=False)
    assert p.returncode != 0 and "wdm needs float64 under JAX" in p.stderr
