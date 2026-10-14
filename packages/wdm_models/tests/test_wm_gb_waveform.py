"""F-A9: the GB template (spec §5.2 waveform.py): prepared units, the eager orbit, a pytree
with static window, the two-tier check (construction guard with the half-amplitude diagnosis;
G5 template sums against golden/galactic_binary/products.json), ``onfly`` the default and
audited, ``fourier`` within 1e-12 of the golden gbonfly store template."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wdm_models.galactic_binary import config as C
from wdm_models.galactic_binary import waveform as W
from wdm_models.galactic_binary.prepared import PreparedCase

pytestmark = pytest.mark.golden
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
TESTS = Path(__file__).resolve().parent
SRC = [TESTS.parent / "src", TESTS.parents[1] / "domainia" / "src", TESTS.parents[1] / "wdm" / "src"]


@pytest.fixture(scope="module")
def preps():
    return {case: PreparedCase.load(case) for case in CASES}


def test_onfly_is_the_default_and_backends_are_checked(bg, preps):
    tpl = W.GBTemplate(preps["gb_3mo"], check=False)
    assert (tpl.backend, tpl.BW, tpl.bin_lo, tpl.n_bins, tpl.T) == ("onfly", 4096, 7351, 1139, 7884000.0)
    assert tpl.orbit.shape == (3, 4096, 3) and np.array_equal(tpl.scale, 5.0 * preps["gb_3mo"].sigma0[:, None])
    with pytest.raises(ValueError, match="backend must be one of"):
        W.GBTemplate(preps["gb_3mo"], backend="gbonfly")
    with pytest.raises(ValueError, match="positive even"):
        W.GBTemplate(preps["gb_3mo"], BW=4095)
    with pytest.raises(ValueError, match="window must be"):
        W.GBTemplate(preps["gb_3mo"], window="band", check=False)
    win = tpl.on("nominal", preps["gb_3mo"])
    assert (win.bin_lo, win.n_bins) == (7740, 360) and win.orbit is tpl.orbit


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ["fourier", "onfly"])
def test_construction_guard_passes_on_the_physical_amplitude(case, backend, bg, preps):
    deltas = W.GBTemplate(preps[case], backend=backend).check(preps[case])
    worst = max(v for d in deltas.values() for v in d.values())
    assert worst <= W.GUARD_RTOL[backend]
    if backend == "fourier":
        assert worst < 1e-12                       # only the 10-decimal rounding of the literals


def test_guard_diagnoses_the_half_amplitude_convention(bg, preps, monkeypatch):
    real = W._template_on_window
    monkeypatch.setattr(W, "_template_on_window", lambda p8, **kw: 0.5 * real(p8, **kw))
    with pytest.raises(W.TemplateCheckError, match=r"binary_galactica < 0\.5\.0 uses the half-amplitude "
                                                   r"native convention; install >= 0\.5\.0: sum\|h\|\^2 on "
                                                   r"the window is 0\.2500"):
        W.GBTemplate(preps["gb_3mo"], backend="fourier")
    monkeypatch.setattr(W, "_template_on_window", lambda p8, **kw: 1.01 * real(p8, **kw))
    with pytest.raises(W.TemplateCheckError, match="gb_3mo fourier template on the window"):
        W.GBTemplate(preps["gb_3mo"], backend="fourier")
    monkeypatch.setattr(W, "_template_on_window", lambda p8, **kw: -real(p8, **kw))   # a sign error
    with pytest.raises(W.TemplateCheckError, match="Re<z,h>"):
        W.GBTemplate(preps["gb_3mo"], backend="onfly")


def test_old_binary_galactica_is_refused(bg, monkeypatch):
    monkeypatch.setattr(bg, "__version__", "0.4.0")
    with pytest.raises(ImportError, match=r"half-amplitude native convention; install >= 0\.5\.0 \(found 0\.4\.0"):
        W.check_binary_galactica()
    monkeypatch.setattr(bg, "__version__", "0.10.0")                 # versions, not strings
    assert W.check_binary_galactica() is bg


@pytest.mark.parametrize("case", CASES)
def test_g5_template_sums_against_the_golden_products(case, bg, preps, gb_products, gb_recording_build, s5):
    """Fine tier: ``fourier`` sums at 1e-10 relative on the recording build (spec §7.3), else
    1e-8; both windows."""
    prep = preps[case]
    gold = gb_products["cases"][case]["template"]
    assert gold["BW"] == C.TEMPLATE_BW and gold["units"] == "template_on_window(...) / (DT * sigma0[:, None])"
    gate = 1e-10 if gb_recording_build else 1e-8
    tpl = W.GBTemplate(prep, backend="fourier", check=False)
    for name, z in (("window", prep.z), ("store", prep.store_z)):
        s2, zh = W.template_sums(tpl.on(name, prep), prep.inject, z)
        s5(f"G5/template/{case}/{name}/sum_h2", abs(s2 / gold[name]["sum_h2"] - 1.0), gate)
        s5(f"G5/template/{case}/{name}/re_zh", abs(zh / gold[name]["re_zh"] - 1.0), gate)
        assert abs(s2 / gold[name]["sum_h2"] - 1.0) <= gate, (name, s2)
        assert abs(zh / gold[name]["re_zh"] - 1.0) <= gate, (name, zh)


@pytest.mark.parametrize("case", CASES)
def test_fourier_store_template_is_gbonfly_to_1e_12(case, bg, preps, gb_g6, s5):
    """``golden/galactic_binary/intermediates_<case>.npz["store_template"]`` is gbonfly's template
    at the injection (lineage C); binary_galactica ``fourier`` agrees to ~2e-16 of the peak, never
    bitwise (Taylor sinc, scalar contractions)."""
    _, A = gb_g6(case)
    prep = preps[case]
    h = W.GBTemplate(prep, backend="fourier", check=False)(np.asarray(prep.inject))
    ref = A["store_template"]
    assert isinstance(h, np.ndarray) and h.shape == ref.shape == (2, prep.store_n) and h.dtype == np.complex128
    rel = float(np.max(np.abs(h - ref)) / np.max(np.abs(ref)))
    s5(f"G5/template/{case}/store_vs_gbonfly", rel, 1e-12)
    assert rel <= 1e-12


@pytest.mark.parametrize("case", CASES)
def test_onfly_audit_is_reported_not_gated(case, bg, preps, record_property):
    """Spec §5.2/§7.3: the release default ``onfly`` against ``fourier`` at the injection on
    the store -- reported (1 - FF, the complex scale, the sums), only sanity-bounded here."""
    prep = preps[case]
    f = W.GBTemplate(prep, backend="fourier", check=False)(np.asarray(prep.inject))
    o = W.GBTemplate(prep, backend="onfly", check=False)(np.asarray(prep.inject))
    ff = float(np.real(np.vdot(f, o)) / np.sqrt(np.real(np.vdot(f, f)) * np.real(np.vdot(o, o))))
    scale = complex(np.vdot(f, o) / np.vdot(f, f))
    record_property("s5_audit", json.dumps({"item": f"audit/onfly/{case}/store", "one_minus_ff": 1.0 - ff,
                                            "scale_re": scale.real, "scale_im": scale.imag}))
    assert 0.0 <= 1.0 - ff < 1e-4 and abs(scale - 1.0) < 1e-3


def test_template_is_an_immutable_pytree_with_static_window(bg, preps):
    import jax
    import jax.numpy as jnp

    prep = preps["gb_3mo"]
    tpl = W.GBTemplate(prep, backend="fourier", check=False)
    leaves, tree = jax.tree_util.tree_flatten(tpl)
    assert [np.shape(x) for x in leaves] == [(3, 4096, 3), (2, 1)]
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert isinstance(back, W.GBTemplate) and (back.bin_lo, back.n_bins, back.backend) == (7351, 1139, "fourier")
    with pytest.raises(AttributeError, match="immutable"):
        tpl.backend = "onfly"
    eager = tpl(np.asarray(prep.inject))
    jitted = jax.jit(lambda t, p: t(p))(tpl, jnp.asarray(prep.inject))
    assert not isinstance(jitted, np.ndarray)
    assert float(jnp.max(jnp.abs(jitted - eager))) <= 1e-10 * float(np.max(np.abs(eager)))


def test_x64_off_is_refused_before_any_template():
    code = ("import numpy as np\n"
            "from wdm_models.galactic_binary.prepared import PreparedCase\n"
            "from wdm_models.galactic_binary.waveform import GBTemplate\n"
            "try:\n"
            "    GBTemplate(PreparedCase.load('gb_3mo'))\n"
            "except RuntimeError as e:\n"
            "    print('REFUSED', e)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "JAX_"))}
    env.update(PYTHONPATH=os.pathsep.join(map(str, SRC)), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               JAX_ENABLE_X64="false", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(TESTS),
                         capture_output=True, text=True)
    if "No module named 'binary_galactica'" in out.stderr or "No module named 'jax'" in out.stderr:
        pytest.skip("binary_galactica/jax not installed")
    assert out.returncode == 0, out.stderr
    assert "REFUSED wdm needs float64 under JAX: call wdm.backend.enable_x64()" in out.stdout
