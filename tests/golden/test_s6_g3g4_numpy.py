"""F-C8 (S6 gate, spec section 7.6): the NumPy release reproduces G3 and G4 at the fast tier.

For each of the 38 targets ``scripts.galactic_binary.validate.run`` builds the release
likelihood (``wdm_models.registry.build``, backend ``numpy``, binary_galactica ``fourier``) and
evaluates

* G3 at rows 0, n/2, n-1 of every run directory (58 directories, 174 rows): the chain's own
  ``lnlike`` at the approved uniform gate 1e-8 (measured max 7.02e-10, 12-month tri_pad), and
  lineage C's NumPy value at the same gate (measured max 9.1e-13: the waveform swap, bg
  ``fourier`` = gbonfly to ~2e-16);
* G4 (3 and 6 months) at the 11 stress points: identical -inf masks against lineage C's
  reference and the expected mask, 1e-10 + 1e-12 |lnL| at the well-conditioned points
  (measured max 9.1e-13), the bare counters; the cached models also through the cache
  evaluator (lineage C's ``counters_cache`` exactly, ``lnl_cache`` at the same gate; measured
  max 1.8e-12);

and, with the test-only gbonfly adapter (lineage C streams gbonfly store templates; ``legacy``:
needs the frozen trees), G3 and G4 again -- spec section 7.6, S6: the adapter first, "G3 and G4
(mask and counters)" -- G3 on the recording build **bitwise** against lineage C at all 174 rows
(the likelihood numerics alone), G4 under the same rules as above (asserted; its items are
recorded by the ``fourier`` test).  Each comparison is an ``s6`` property for
``scripts/golden/s6_report.py``.  ``slow``: the 12-month tri_raw and pads, the 6-month tri_pad.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.galactic_binary import validate as V
from scripts.golden import s6_cases as S6
from scripts.golden.paths import legacy_available

pytestmark = pytest.mark.golden
RECORDING = V.recording_build()


def _targets():
    return [pytest.param(c, m, id=f"{c}-{m}", marks=[pytest.mark.slow] if (c, m) in S6.SLOW else [])
            for c, m in S6.targets()]


@pytest.fixture(scope="session")
def gbonfly_tables():
    """One lineage-C stream for every G3 row of the 38 targets and every G4 stress point of the
    30 three- and six-month targets (~3 s)."""
    if not legacy_available():
        pytest.skip("frozen legacy trees not present (CI/Colab): the gbonfly adapter needs lineage C")
    from scripts.golden.gb_templates import streamed_templates

    rows = [(c, r["theta"][:, :8]) for c, m in S6.targets() for r in V.chain_runs(c, m, "fast")]
    rows += [(c, V.stress_points(c, m, "fast")["theta"][:, :8]) for c, m in S6.targets() if c in S6.STRESS_CASES]
    tables, _ = streamed_templates(rows)
    return tables


@pytest.mark.parametrize("case,model", _targets())
def test_g3_and_g4_with_binary_galactica_fourier(case, model, record_property):
    rep = V.run(cases=[case], models=[model], goldens=["g3", "g4"])
    t = rep["targets"][f"{case}/{model}"]
    S6.record_g3(record_property, t, "g3", exact_lineage=False)
    S6.record_g4(record_property, t)
    assert rep["recording_build"] is RECORDING
    assert t["ok"] and rep["ok"], rep["problems"]
    assert t["g3"]["n_rows"] == 3 * len(t["g3"]["runs"]) and t["g3"]["max_abs"] <= S6.g3_gate(case, model)
    if case in S6.STRESS_CASES:
        g4 = t["g4"]
        assert g4["reference"]["inf_mismatch"] == g4["reference"]["expected_mismatch"] == 0
        assert ("cache" in g4) is (model in S6.CACHED)
    else:
        assert t["g4"] == {"applicable": False, "ok": True, "problems": []}


@pytest.mark.legacy
@pytest.mark.parametrize("case,model", _targets())
def test_g3_and_g4_with_the_gbonfly_adapter(case, model, gbonfly_tables, record_property):
    goldens = ["g3", "g4"] if case in S6.STRESS_CASES else ["g3"]
    rep = V.run(cases=[case], models=[model], goldens=goldens, templates="gbonfly", tables=gbonfly_tables)
    t = rep["targets"][f"{case}/{model}"]
    S6.record_g3(record_property, t, "g3_gbonfly", exact_lineage=RECORDING)
    assert t["ok"] and rep["ok"], rep["problems"]          # G4 too: masks, counters, the cache, the gates
    if RECORDING:
        assert all(np.all(r["lineage_c"]["abs_delta"] == 0.0) for r in t["g3"]["runs"].values())
    if case in S6.STRESS_CASES:
        g4 = t["g4"]
        assert g4["reference"]["inf_mismatch"] == g4["reference"]["expected_mismatch"] == 0
        assert ("cache" in g4) is (model in S6.CACHED)
