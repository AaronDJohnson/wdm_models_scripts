"""F-C3: the 15 galactic-binary compositions (spec sections 4.9, 5.3; ``galactic_binary.models``).

At the anchor of each case (``inject_glass`` + ``mu_fit``, ``mu_freq`` for the Whittle forms)
and with the golden gbonfly store template, every target reproduces G6
(``golden/galactic_binary/intermediates_<case>``): each channel's (quad, logdet) and lnL
(``make_lnlike``) bit for bit on the build that recorded them, else (quad, logdet) at 1e-12
relative and lnL at the approved G3 gate.  The cache evaluator counts like lineage C's cache
classes on the stress points.  No binary_galactica here except the last test (the template
is a plain callable), so this file runs in the NumPy-only lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from domainia.likelihood import FourierWhittle, GaussianLikelihood
from domainia.sampling import NoiseStateCache
from wdm_models.galactic_binary import models as M
from wdm_models.galactic_binary.prepared import PreparedCase
from wdm_models.options import ModelOptions, resolve

pytestmark = pytest.mark.golden
GB = Path(__file__).resolve().parents[3] / "golden" / "galactic_binary"
#: Heavy at the anchor (dense P = 2700 factorisations, the 12-month and 6-month pads): ``slow``.
SLOW = {("gb_12mo", "tri_raw"), ("gb_12mo", "band_pad"), ("gb_12mo", "mid_pad"), ("gb_12mo", "tri_pad"),
        ("gb_6mo", "tri_pad")}
NUMPY = resolve("galactic_binary", "band_raw", "numpy")


def _targets():
    return [pytest.param(c, m, id=f"{c}-{m}", marks=[pytest.mark.slow] if (c, m) in SLOW else [])
            for c, ms in M.TARGET_MODELS.items() for m in ms]


def _g6(case: str):
    g = json.loads((GB / f"intermediates_{case}.json").read_text())
    with np.load(GB / f"intermediates_{case}.npz", allow_pickle=False) as z:
        return g, {k: np.array(z[k]) for k in z.files}


def _anchor(prep, model):
    a = prep.arrays
    mu = a["mu_freq"] if model.startswith("whittle") else np.asarray(a["mu_fit"]).ravel()
    return np.concatenate([a["inject_glass"], mu])


@pytest.mark.parametrize("case,model", _targets())
def test_composition_at_the_anchor_is_g6(case, model, same_build, golden_io):
    prep = PreparedCase.load(case)
    g, z = _g6(case)
    h = z["store_template"]
    lik = M.likelihood(model, prep, template=lambda p8: h, resolved=resolve("galactic_binary", model, "numpy"))
    assert isinstance(lik, FourierWhittle if model.startswith("whittle") else GaussianLikelihood)
    theta = _anchor(prep, model)
    ql = lik.quad_logdet(theta)
    lnl = float(lik(theta[None, :])[0])
    ref = z["quad_logdet"][g["quad_logdet_models"].index(model)]
    want = golden_io.to_float(g["lnl_at_anchor"][model]["lnl_make_lnlike"])
    assert np.all(np.isfinite(ql)) and np.isfinite(lnl)
    if same_build(g["build"]):
        assert np.array_equal(ql, ref) and lnl == want, (ql - ref, lnl - want)
    gate = json.loads((GB / "tolerances_measured.json").read_text())["approved"]["G3"]["numpy"][f"{case}/{model}"]
    assert np.allclose(ql, ref, rtol=1e-12, atol=0) and abs(lnl - want) <= gate


def test_geometry_counts():
    want = {"gb_3mo": (720, 693, 27, 360, 347, 719), "gb_6mo": (1314, 1287, 27, 657, 644, 1313),
            "gb_12mo": (2700, 2700, 0, 1350, 1350, 2700)}
    for case, (cells, cut, edge, nominal, crop, _) in want.items():
        geo = M.geometry(PreparedCase.load(case))
        assert (geo.cells.size, geo.cut_cells.size, geo.edge.size, geo.nominal.n, geo.crop.n) == \
            (cells, cut, edge, nominal, crop), case
        assert len(geo.store_bands) == 11 and geo.pad_kept.size == cells and geo.pad_grid.Nt == geo.NT + 16
        assert geo.periodic is (case == "gb_12mo")


def test_the_cache_evaluator_counts_like_lineage_c():
    """3-month tri_exact (MatrixCache) and band_exact (EdgeSchurCache, the Schur form) over the
    11 stress points in one call: lineage C's counters and lnl_cache at the NumPy cache gate."""
    prep = PreparedCase.load("gb_3mo")
    _, z = _g6("gb_3mo")
    h = z["store_template"]                     # in-box stress sources are the injection
    for model in ("tri_exact", "band_exact"):
        st = json.loads((GB / "stress" / "gb_3mo" / f"{model}.fast.json").read_text())
        r = resolve("galactic_binary", model, "numpy", ModelOptions(evaluator="cache"))
        cache = M.wrap(M.likelihood(model, prep, template=lambda p8: h, resolved=r), model, r)
        assert isinstance(cache, NoiseStateCache) and cache.capacity == 24
        X = np.array([p["theta"] for p in st["points"]])
        v = cache(X)
        c = cache.counters()
        assert {k: c[k] for k in st["counters_cache"]} == st["counters_cache"], model
        ref = np.array([p["lnl_cache"] for p in st["points"]], dtype=float)
        assert np.array_equal(np.isinf(v), np.isinf(ref)), model
        fin = np.isfinite(ref)
        assert np.all(np.abs(v[fin] - ref[fin]) <= 1e-10 + 1e-12 * np.abs(ref[fin])), model


def test_refusals():
    prep = PreparedCase.load("gb_3mo")
    _, z = _g6("gb_3mo")
    h = z["store_template"]
    window = h[:, 360:779]
    with pytest.raises(ValueError, match="store needs"):
        M.likelihood("band_raw", prep, template=lambda p8: window, resolved=NUMPY)(
            _anchor(prep, "band_raw")[None, :])
    other = SimpleNamespace(case="gb_3mo", arrays={**prep.arrays, "matrix_cut": np.array([4, 4])})
    with pytest.raises(ValueError, match="crop is for cut"):
        M.likelihood("whittle_cut", other, template=lambda p8: h, resolved=NUMPY)
    with pytest.raises(ValueError, match="not a target"):
        M.likelihood("band_cut", PreparedCase.load("gb_12mo"), template=lambda p8: h, resolved=NUMPY)
    with pytest.raises(ValueError, match="no cached evaluator"):
        M.wrap(object(), "band_raw", SimpleNamespace(evaluator="cache"))


def test_the_jax_build_is_s7s():
    """On JAX the signal must be F-A's ``GBTemplate`` pytree (S7, F-D3 lifted F-C's guards): a
    plain callable template is refused with ``TypeError`` by ``likelihood`` and ``build`` alike
    (the skip: the NumPy-only lane has no JAX)."""
    pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    prep = PreparedCase.load("gb_3mo")
    h = _g6("gb_3mo")[1]["store_template"]
    with pytest.raises(TypeError, match="GBTemplate"):        # S7 (F-D3): on JAX the signal is the pytree
        M.likelihood("band_raw", prep, template=lambda p8: h, resolved=SimpleNamespace(backend="jax"))
    with pytest.raises(TypeError, match="GBTemplate"):
        M.build("gb_3mo", "band_raw", backend="jax", template=lambda p8: h)


def test_build_uses_the_release_template():
    """``build`` with no template: binary_galactica on the store (``onfly`` by default,
    ``fourier`` on request); with ``fourier`` the anchor is G6's lnL at the G3 gate."""
    pytest.importorskip("binary_galactica", reason="binary_galactica (JAX) not installed")
    g, _ = _g6("gb_3mo")
    lik = M.build("gb_3mo", "tri_exact", options=ModelOptions(waveform_backend="fourier"))
    assert lik.parts.signal.fn.template.backend == "fourier"
    assert M.build("gb_3mo", "band_raw").parts.signal.fn.template.backend == "onfly"
    theta = _anchor(PreparedCase.load("gb_3mo"), "tri_exact")
    assert abs(float(lik(theta[None, :])[0]) - g["lnl_at_anchor"]["tri_exact"]["lnl_make_lnlike"]) <= 1e-8
