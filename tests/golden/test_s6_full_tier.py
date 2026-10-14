"""F-C9 (approval-gated, spec sections 6 and 7.6): the NumPy release at the FULL tier -- 32 rows
per run directory (rows 0, n/2, n-1 and 29 from ``default_rng(20261011)``) of the 58 run
directories and the 12 stress points (+ the capture-time non-PD scan point) of the 30 three- and
six-month targets.

Marked ``golden_full``: the root ``addopts`` deselect it.  Neither the full-tier goldens nor this
run exist yet; both need the user's approval (their run times are not measured):

    cd wdm_models_scripts && source env/activate.sh
    for c in gb_3mo gb_6mo gb_12mo; do $PY -P -m scripts.golden.capture_gb rows --case $c --tier full; done
    for c in gb_3mo gb_6mo; do $PY -P -m scripts.golden.capture_gb stress --case $c --tier full; done
    $PY -m pytest -q -m golden_full tests/golden/test_s6_full_tier.py

(or ``$PY -P -m scripts.galactic_binary.validate --tier full --golden g3 g4 --report <outside>``).
Gates as the fast tier (approved: G3 1e-8, G4 1e-10 + 1e-12 |lnL| at well-conditioned points).
"""

from __future__ import annotations

import pytest

from scripts.galactic_binary import validate as V
from scripts.golden import s6_cases as S6

pytestmark = pytest.mark.golden_full


@pytest.mark.parametrize("case,model", [pytest.param(c, m, id=f"{c}-{m}") for c, m in S6.targets()])
def test_numpy_full_tier(case, model):
    rep = V.run(tier="full", cases=[case], models=[model], goldens=["g3", "g4"])
    t = rep["targets"][f"{case}/{model}"]
    assert t["ok"] and rep["ok"], rep["problems"]
    assert t["g3"]["n_rows"] == 32 * len(t["g3"]["runs"])
