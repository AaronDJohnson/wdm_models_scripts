"""B7 (G8): posterior summaries of the sinusoid final chains, and the summariser itself."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.golden.paths import ROOT, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")

REL = Path(__file__).resolve().parents[2]
GOLDEN = REL / "golden"
SIN_ROOT = ROOT / "wdm_paper_2" / "sinusoid_final"
#: impulse v2 checkout at 964283c (env/constraints.txt); the Geyer code is vendored from it.
IMPULSE_DIAGNOSTICS = Path.home() / "Documents" / "github" / "impulse_mcmc" / "impulse" / "diagnostics.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("summarise_chains", REL / "scripts" / "golden" / "summarise_chains.py")

IMPULSE_SCRIPT = r"""
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("impulse_diagnostics", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
np.save(sys.argv[3], mod.effective_sample_size(np.load(sys.argv[2])))
"""


@pytest.mark.skipif(not IMPULSE_DIAGNOSTICS.is_file(), reason="impulse checkout not present")
def test_geyer_is_impulse_verbatim(tmp_path):
    rng = np.random.default_rng(7)
    x = np.empty((5000, 3))
    x[0] = rng.standard_normal(3)
    for t in range(1, len(x)):                       # AR(1) with three correlation lengths
        x[t] = np.array([0.5, 0.9, 0.99]) * x[t - 1] + rng.standard_normal(3)
    np.save(tmp_path / "x.npy", x)
    proc = subprocess.run([sys.executable, "-P", "-c", IMPULSE_SCRIPT, str(IMPULSE_DIAGNOSTICS),
                           str(tmp_path / "x.npy"), str(tmp_path / "ess.npy")],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert proc.returncode == 0, proc.stderr
    assert np.array_equal(np.load(tmp_path / "ess.npy"), sc.effective_sample_size(x))


@requires_legacy
def test_summariser_is_read_only():
    run = SIN_ROOT / "manuscript_low" / "band_raw_flow_seed301"
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.iterdir()}
    sc.summarise_target("sinusoid", SIN_ROOT, "manuscript_low", "band_raw",
                        ["manuscript_low/band_raw_flow_seed301"], hash_files=True)
    assert {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.iterdir()} == before


@requires_legacy
def test_45_sinusoid_summaries_in_manifest():
    files = json.loads((GOLDEN / "MANIFEST.json").read_text())["files"]
    runs = json.loads((SIN_ROOT / "manifest.json").read_text())["runs"]
    expected = {f"chain_summaries/sinusoid/{case}_{model}.json" for case, d in runs.items() for model in d}
    assert len(expected) == 45 and expected <= set(files)


@requires_legacy
@pytest.mark.parametrize("case", ["manuscript_low", "manuscript_mid", "manuscript_high"])
def test_sinusoid_summaries_reproduce_stored_diagnostics(case):
    runs = json.loads((SIN_ROOT / "manifest.json").read_text())["runs"][case]
    n_fresh = 0
    for model, rel in runs.items():
        s = json.loads((GOLDEN / f"chain_summaries/sinusoid/{case}_{model}.json").read_text())
        assert s["schema"] == 1 and not s["pooled"] and s["runs"] == [rel]
        pc = s["per_chain"][0]
        assert pc["files"]["chain_0.bin"]["bytes"] == 8 * (len(s["parameter_names"]) + 4) * pc["rows"]
        assert pc["kept"] == pc["rows"] - int(0.3 * pc["rows"]) == s["samples"]
        for name, p in s["parameters"].items():
            assert p["q05"] <= p["q16"] <= p["q50"] <= p["q84"] <= p["q95"]
            assert p["mcse_mean"] == pytest.approx(p["std"] / np.sqrt(p["ess"]))
        assert (s["parameters"]["phi_0"]["periodic"] == [-np.pi, np.pi])
        if pc["stale"]:
            continue
        n_fresh += 1
        stored = json.loads((SIN_ROOT / rel / "diagnostics.json").read_text())["parameters"]
        for name, mine in pc["parameters"].items():
            if s["parameters"][name]["periodic"]:     # stored runs disagree on unwrapping phi_0
                continue
            for k in ("ess_batch", "ess_geyer", "rhat"):
                assert mine[k] == pytest.approx(stored[name][k], rel=1e-12, abs=0.0), (model, name, k)
    assert n_fresh >= 12


def test_known_caveats_and_stale_runs():
    high = json.loads((GOLDEN / "chain_summaries/sinusoid/manuscript_high_tri_exact.json").read_text())
    assert high["caveats"] and round(high["min_ess"], 1) == 2638.6
    stale = sorted(f"{s['case']}/{s['model']}" for s in (
        json.loads(p.read_text()) for p in (GOLDEN / "chain_summaries/sinusoid").glob("*.json"))
        if s["per_chain"][0]["stale"])
    assert stale == ["manuscript_mid/band_raw", "manuscript_mid/exact", "manuscript_mid/tri_cut"]
