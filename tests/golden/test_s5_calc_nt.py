"""F-A8 (S5): ``wdm_models.galactic_binary.prepare.calc_Nt`` against the legacy
``prepare.calc_Nt``, run fresh in lineage C (``gb_legacy calcnt``).

The shipped prepared files' ``nt_star`` (253.63 at 3 months) predates ``calc_Nt`` (the earlier
tiling-dependent estimate), so the legacy function itself is the oracle; the package test
``test_wm_gb_prepare.py::test_calc_nt_is_the_legacy_function`` pins its values recorded from this
worker.  Default lane: 3 months (~6 s); ``slow``: 6 and 12 months (~25 s).
"""

from __future__ import annotations

import json

import pytest

from scripts.golden.lineage import run_worker
from scripts.golden.paths import ROOT, legacy_available

pytestmark = [pytest.mark.golden, pytest.mark.legacy,
              pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]
#: Recorded 2026-10-13 from this worker on the recording build (same values as the package test).
LEGACY_CALC_NT = {"gb_3mo": (246.86598221011158, 240), "gb_6mo": (444.3893998481366, 450),
                  "gb_12mo": (925.7765186242784, 900)}


@pytest.mark.parametrize("case", ["gb_3mo", pytest.param("gb_6mo", marks=pytest.mark.slow),
                                  pytest.param("gb_12mo", marks=pytest.mark.slow)])
def test_release_calc_nt_is_the_legacy_function(case, record_property):
    pytest.importorskip("binary_galactica")
    from wdm.backend import enable_x64

    from wdm_models.galactic_binary.prepare import calc_Nt

    out = run_worker("C", "gb_legacy", "calcnt", "--workspace", str(ROOT), "--case", case)
    legacy = json.loads((out / "payload" / "galactic_binary" / "calc_nt.json").read_text())["cases"][case]
    star, valid = LEGACY_CALC_NT[case]
    # the worker runs without the release's XLA_FLAGS (lineage.worker_env strips them), so the
    # legacy L-BFGS may move in its last digits between runs: compare the recording at 1e-9
    assert abs(legacy["nt_star"] / star - 1.0) <= 1e-9 and legacy["nt_valid"] == valid
    enable_x64()
    got = calc_Nt(case)
    record_property("s5", json.dumps({"item": f"G5/prepare_port/{case}/calc_nt_legacy", "gate": 1e-6,
                                      "observed": abs(got["nt_star"] / legacy["nt_star"] - 1.0)}))
    assert abs(got["nt_star"] / legacy["nt_star"] - 1.0) <= 1e-6
    assert got["nt_valid"] == legacy["nt_valid"] and got["nt_band_bins"] == legacy["nt_band_bins"]
