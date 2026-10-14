"""E-B7 (approval-gated, spec sections 6 and 7.6): the NumPy release at the FULL tier --
all 30 G1 reference points (Whittle 27) and all 20 G2 chain rows of each of the 45 targets.

Marked ``golden_full``: the root ``addopts`` deselect it.  Local runs need the user's
approval (estimated 45-50 min, never run here; edge-exact ~3-5 s per point).  Command:

    cd wdm_models_scripts && source env/activate.sh && $PY -m pytest -q -m golden_full tests/golden/test_s3_full_tier.py

(or the CLI: ``scripts.sinusoid.validate --backend numpy --tier full --report <outside>``).
Gates: the approved NumPy reference gates (no bitwise claim: lineage A has no full-tier
report, and the part-2 review measured 27-30 of 30 bitwise per model), identical -inf
masks with ``pd_failed_rows`` = 12 / 13 / 12 for band/mid/tri_exact, and the approved
chain-row gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.s3_cases import NEG_INF, record_target
from scripts.sinusoid import validate as V

pytestmark = pytest.mark.golden_full


@pytest.mark.parametrize("case,model", [pytest.param(c, m, id=f"{c}-{m}") for c in V.CASES for m in V.MODELS])
def test_numpy_full_tier(case, model, record_property):
    rep = V.run(backend="numpy", tier="full", cases=[case], models=[model])
    t = rep["targets"][f"{case}/{model}"]
    record_target(record_property, t, recording=False)       # full tier: the approved gates, never 0.0
    assert t["ok"] and rep["ok"], rep["problems"]
    g1, g2 = t["g1"], t["g2"]
    assert len(g1["index"]) == (27 if model.startswith("whittle") else 30) and len(g2["index"]) == 20
    want = NEG_INF.get(model, 0)
    assert int(np.sum(g1["value"] == -np.inf)) == want == int(np.sum(g1["reference"] == -np.inf))
    assert g1["counters"]["pd_failed_rows"] == g1["counters"]["pd_failed_states"] == want
    assert int(np.sum(g1["last_failed"])) == want
