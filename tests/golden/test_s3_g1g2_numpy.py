"""E-B3 (S3 gate, spec section 7.6): the NumPy release reproduces G1 and G2 at the fast tier.

For each of the 45 targets, ``scripts.sinusoid.validate.run`` builds the release likelihood
(``wdm_models.registry.build``, backend ``numpy``, strategy ``historical``) and evaluates

* G1 at the fast-tier reference points (truth, first tight, first wide, first -inf or
  first stress corner): on the recording build **bitwise** (every |delta| == 0.0: lineage A
  is bitwise there for all 15 models), elsewhere the approved NumPy gate;
* G2 at chain rows 0, 1, 19 of 20 (phi_0 wrapped): the approved chain-row gate (atol 1e-5,
  exact 1e-4) against the chain's own lnlike, and on the recording build lineage A's
  re-evaluation of the same rows at the model's approved NumPy reference gate (measured:
  bitwise at all 135 rows, recorded but not required);

with identical -inf masks, ``last_failed`` == the -inf rows and the counters
(``evaluations``/``noise_states`` == rows, ``pd_failed_rows`` == ``pd_failed_states`` == the
-inf rows -- 1 per edge-exact target here; the 12/13/12 per case are the full tier's, E-B7 --;
Whittle: no noise states).  Each comparison is recorded as an ``s3`` property for
``scripts/golden/s3_report.py``.  Cheap targets run in the default lane; tri_*, *_pad and
*_exact are ``slow`` (the S3 report runs them with ``--full``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.golden.paths import GOLDEN
from scripts.golden.s3_cases import CHEAP, NEG_INF, record_target, recording_build
from scripts.sinusoid import validate as V

pytestmark = pytest.mark.golden
RECORDING = recording_build()


def _targets():
    return [pytest.param(c, m, id=f"{c}-{m}", marks=() if m in CHEAP else pytest.mark.slow)
            for c in V.CASES for m in V.MODELS]


def test_g1_neg_inf_points_and_lineage_a_are_as_captured():
    """Golden integrity (cheap): 12/13/12 -inf reference points per case for band/mid/tri_exact
    and none elsewhere; lineage A reproduced every fast-tier point of all 45 targets bitwise."""
    files = json.loads((GOLDEN / "sinusoid" / "references" / "INDEX.json").read_text())["files"]
    a = json.loads((GOLDEN / "sinusoid" / "legacy_eval" / "A_references_fast.json").read_text())
    assert len(files) == 45 and len(a["points"]) == 45
    for case in V.CASES:
        for model in V.MODELS:
            entry = files[f"{case}_{model}_reference.npz"]
            assert len(entry["neg_inf_index"]) == NEG_INF.get(model, 0), (case, model)
            point = a["points"][f"{case}__{model}"]
            assert point["index"] == entry["fast_index"] and point["inf_mismatch"] == 0
            assert point["abs_delta"] == [0.0] * 4, (case, model)


@pytest.mark.parametrize("case,model", _targets())
def test_numpy_fast_tier(case, model, record_property):
    rep = V.run(backend="numpy", tier="fast", cases=[case], models=[model])
    t = rep["targets"][f"{case}/{model}"]
    record_target(record_property, t, recording=rep["recording_build"])
    assert rep["recording_build"] is RECORDING
    assert t["ok"] and rep["ok"], rep["problems"]
    g1, g2 = t["g1"], t["g2"]
    assert len(g1["index"]) == 4 and len(g2["index"]) == 3
    assert np.array_equal(g1["value"] == -np.inf, g1["reference"] == -np.inf)
    assert g1["counters"]["pd_failed_rows"] == (1 if model in NEG_INF else 0)
    if RECORDING:
        assert g1["max_abs"] == 0.0, g1["abs_delta"]                  # the approved bitwise rule
        assert g2["lineage_A"]["passed"].all()                        # NumPy reference gate vs lineage A


def test_registry_exact_model_has_the_folded_density_order(record_property):
    """D1 (part 2, G-F1): the registry's ``exact`` model evaluates WT
    ``FastFullCovariance.lnlike``'s association (``density_order="folded"``); G1 points 8 and
    18 of manuscript_mid -- outside the fast tier -- are where the default order would differ
    (7.3e-12, 5.8e-11).  Bitwise on the recording build, else the approved exact gate."""
    from wdm_models.registry import build

    pts = V.reference_points("manuscript_mid", "exact", "full")
    idx = [0, 8, 18]
    lik = build("sinusoid", "manuscript_mid", "exact", backend="numpy")
    got = lik(pts["theta"][idx])
    gate = V.approved_gates("numpy", "references")["exact"]
    allowed = np.zeros(3) if RECORDING else gate.allowed(pts["reference"][idx], pts["near"][idx])
    for i, g, r, a in zip(idx, got, pts["reference"][idx], allowed, strict=True):
        record_property("s3", json.dumps({"item": f"sinusoid/manuscript_mid/exact/g1/point{i}/folded",
                                          "observed": abs(float(g) - float(r)), "gate": float(a)}))
        assert abs(float(g) - float(r)) <= a, (i, g, r)
