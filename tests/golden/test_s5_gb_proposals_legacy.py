"""F-B7: the release proposal Fisher and starting covariance (binary_galactica ``fourier``)
against the frozen ``gpu/proposals.py`` (gbonfly), evaluated in a lineage-C subprocess
(``scripts/golden/workers/gbw_proposals.py``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, ROOT, legacy_available

pytestmark = [pytest.mark.golden, pytest.mark.legacy,
              pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]
GATE = 1e-12


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_the_release_fisher_and_initial_covariance_are_the_legacy_ones(record_property):
    pytest.importorskip("binary_galactica")
    from wdm.backend import enable_x64

    enable_x64()
    from wdm_models.galactic_binary import proposals as P
    from wdm_models.galactic_binary.prepared import PreparedCase

    out = run_worker("C", "gb_legacy", "proposals", "--workspace", str(ROOT), "--data-dir", str(DATA))
    legacy = json.loads((out / "proposals.json").read_text())
    prep = PreparedCase(DATA / "galactic_binary" / "prepared" / "gb_3mo.npz")
    fisher = P.SourceFisher.from_prepared(prep, backend="fourier")
    inject = np.asarray(prep.arrays["inject_glass"], float)
    observed = {"fisher_inject": _rel(fisher(inject), legacy["fisher_inject"]),
                "fisher_point": _rel(fisher(np.array(legacy["point"])), legacy["fisher_point"])}
    for model in ("whittle_raw", "band_raw"):
        observed[f"noise_fisher_{model}"] = _rel(P.noise_fisher(prep, model), legacy["noise_fisher"][model])
        observed[f"initial_covariance_{model}"] = _rel(P.initial_covariance(prep, model, fisher),
                                                       legacy["initial_covariance"][model])
    for item, value in observed.items():
        record_property("s5", json.dumps({"item": f"proposals/gb_3mo/{item}", "observed": value, "gate": GATE}))
        assert value <= GATE, (item, value)
    onfly = P.SourceFisher.from_prepared(prep, backend="onfly")
    assert 1e-6 < _rel(onfly(inject), legacy["fisher_inject"]) < 1e-2        # reported, not gated
