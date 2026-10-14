"""F-C8: the ``onfly`` audit (spec section 7.3: "bg onfly (release default): not a golden; the
audit reports dlnL at the G3 rows and 1 - FF against fourier").  Reported, never gated: each
target writes one ``s6_audit`` property (max 1 - FF on the store, max |lnL_onfly - lnL_fourier|
over its G3 rows) that ``scripts/golden/s6_report.py`` lists under ``audit``; the test fails
only on a non-finite audit value."""

from __future__ import annotations

import json

import pytest

from scripts.galactic_binary import validate as V
from scripts.golden import s6_cases as S6

pytestmark = pytest.mark.golden


@pytest.mark.parametrize("case,model", [pytest.param(c, m, id=f"{c}-{m}",
                                                     marks=[pytest.mark.slow] if (c, m) in S6.SLOW else [])
                                        for c, m in S6.targets()])
def test_onfly_audit(case, model, record_property):
    rep = V.run(cases=[case], models=[model], goldens=["audit"])
    a = rep["targets"][f"{case}/{model}"]["audit"]
    record_property(S6.AUDIT_PROPERTY, json.dumps({"item": f"galactic_binary/{case}/{model}/audit",
                                                   "max_one_minus_ff": a["max_one_minus_ff"],
                                                   "max_abs_dlnl": a["max_abs_dlnl"],
                                                   "rows": int(a["dlnl"].size)}))
    assert rep["ok"], rep["problems"]
