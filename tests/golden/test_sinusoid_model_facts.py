"""E-A2: the oracle's model-layer facts, golden/sinusoid/model_facts.json (spec §5.1, §7.2 G5).

Captured by ``$PY -m scripts.golden.sinusoid_goldens facts``: the lineage-A worker records,
per case, WT ``CONFIGS[case]()`` (``dataclasses.asdict``), the WT prior tables
``priors.for_model("freq" | "wdm_full", cfg)``, ``bandcov.noise_bounds(cfg)`` (the bounds that
key the bandcov artifact), the sha256 of ``likelihood.signal(cfg, A_h, f_0, phi_0)`` and a WT
``noise.psd`` sample; the harness adds the runner's record of all 45 final runs
(``wdm_paper_2/sinusoid_final/<run>/run_meta.json`` ``target``: ``prior_bounds``,
``parameter_names``, ``periodic`` -- ``toy_models/sinusoid/gpu/run.py`` ``_DRIFT_BOUNDS`` /
``_FREQUENCY_BOUNDS`` as written by ``run_flow_matrix.py:213``).  ``wdm_models`` tests read
this file; nothing here imports the release packages.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.golden import sinusoid_goldens as SG
from scripts.golden.goldens_io import read_json, read_npz, sha256_array
from scripts.golden.paths import DATA, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
PI = math.pi
FREQ = {"names": ["A_h", "f_0", "phi_0", "log10_A_n", "alpha"],
        "lo": [0.0, 16.0, -PI, 1.0, -4.0], "hi": [3.0, 496.0, PI, 4.0, 0.0],
        "periodic": [False, False, True, False, False]}
FULL = {"names": ["A_h", "f_0", "phi_0", "log10_A_n", "alpha_c", "alpha_d"],
        "lo": [0.0, 16.0, -PI, 1.0, -4.0, -2.0], "hi": [3.0, 496.0, PI, 4.0, 0.0, 2.0],
        "periodic": [False, False, True, False, False, False]}


@pytest.fixture(scope="module")
def facts() -> dict:
    return read_json(SG.FACTS)


def test_every_case_and_every_run_is_recorded(facts):
    assert sorted(facts["cases"]) == sorted(SG.CASES)
    assert sorted(facts["runner"]["runs"]) == sorted(f"{c}/{m}" for c in SG.CASES for m in SG.MODELS)
    assert facts["schema"] == 1


def test_the_wt_prior_tables_are_the_manuscript_study_tables(facts):
    for case, f in facts["cases"].items():
        assert f["priors"] == {"freq": FREQ, "wdm_full": FULL}, case
        assert f["noise_bounds"] == [[-4.0, 0.0], [-2.0, 2.0]], case
        assert f["source_names"] == ["A_h", "f_0", "phi_0"], case


def test_the_runner_recorded_the_wt_prior_for_every_run(facts):
    for key, r in facts["runner"]["runs"].items():
        case, model = key.split("/")
        prior = facts["cases"][case]["priors"]["freq" if model.startswith("whittle") else "wdm_full"]
        assert (r["case"], r["model"]) == (case, model), key
        assert r["parameter_names"] == prior["names"], key
        assert r["prior_bounds"] == [[lo, hi] for lo, hi in zip(prior["lo"], prior["hi"], strict=True)], key
        assert r["periodic"] == {"2": [-PI, PI]}, key
        assert r.get("analysis_fmin", 4.0) == 4.0, key
    gpu = [k for k, r in facts["runner"]["runs"].items() if "configuration" in r]
    assert len(gpu) == 19 and all("analysis_fmin" in facts["runner"]["runs"][k] for k in gpu)


def test_the_configurations_agree_with_the_runs_and_the_shipped_data(facts):
    for key, r in facts["runner"]["runs"].items():
        if "configuration" in r:
            cfg = facts["cases"][key.split("/")[0]]["config"]
            assert {k: cfg[k] for k in r["configuration"]} == r["configuration"], key
    for case, f in facts["cases"].items():
        z = read_npz(DATA / "sinusoid" / case / "toy_data.npz")
        stored = {str(k): float(v) for k, v in zip(z["_cfg_keys"], z["_cfg_vals"], strict=True)}
        assert {k: float(f["config"][k]) for k in stored} == stored, case
        assert (f["config"]["fdot"], f["config"]["fit_fdot"]) == (0.0, False), case


def test_the_signal_is_the_shipped_h(facts):
    for case, f in facts["cases"].items():
        z = read_npz(DATA / "sinusoid" / case / "toy_data.npz")
        assert sha256_array(z["h"]) == f["signal_sha256"], case


def test_the_psd_sample(facts):
    p = facts["psd"]
    S = np.array(p["values"])
    assert S.shape == (len(p["t"]), len(p["f"])) == (16, 17)
    assert sha256_array(S) == p["sha256"]
    assert np.array(p["midpoint"]).shape == (17,)
    assert (p["A_n"], p["alpha_c"], p["alpha_d"]) == (400.0, -2.0, 1.2)


@requires_legacy
@pytest.mark.legacy
def test_a_fresh_capture_reproduces_the_golden(facts):
    """Re-run the lineage-A worker and the run_meta scan (~3 s): identical apart from the
    runtime fingerprint."""
    fresh = SG.facts_report()
    assert {k: v for k, v in fresh.items() if k != "runtime"} == \
        {k: v for k, v in facts.items() if k != "runtime"}
