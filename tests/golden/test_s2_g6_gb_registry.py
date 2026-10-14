"""F-C7 (S6 closes its part of the S2 gate, spec section 7.2 G6): the galactic-binary
compositions the S2 preview left to the full registry -- 11 at 3 months, all 15 at 6 months and
the 8 targets at 12 months -- built by ``wdm_models.galactic_binary.models.build`` (the
registry's GB builder) with the golden gbonfly store template, reproduce
golden/galactic_binary/intermediates_<case>.{json,npz} at the anchor: each channel's (quad,
logdet) at 1e-12 relative and lnL (``lnl_make_lnlike``) at the approved G3 gate.  The four S2
previews (``s2_cases.GB_MODELS`` at 3 months) run here too, for the S6 report.

Each comparison is recorded twice: ``g6`` (``s2_report``: the
``galactic_binary/<case>/quad_logdet/<model>/composition`` items, which ``s2_report.deferred()``
no longer lists) and ``s6`` (``s6_report``: ``galactic_binary/<case>/<model>/g6/<key>``).  On
the recording build every target is also bitwise (asserted).  Default lane: 33 targets
(~10 s); ``slow``: the 12-month tri_raw and pads and the 6-month tri_pad (~20 s).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.galactic_binary.validate import validate_g6
from scripts.golden import s6_cases as S6
from scripts.golden.s2_cases import GB_CASE, GB_MODELS, record, recording_build
from scripts.golden.workers._common import runtime_fingerprint

pytestmark = pytest.mark.golden


def test_the_registry_covers_every_gb_target():
    from wdm_models import registry

    assert S6.TARGETS == registry.TARGETS["galactic_binary"]
    assert set(GB_MODELS) <= set(S6.TARGETS[GB_CASE]) and len(S6.targets()) == 38


@pytest.mark.parametrize("case,model", [pytest.param(c, m, id=f"{c}-{m}",
                                                     marks=[pytest.mark.slow] if (c, m) in S6.SLOW else [])
                                        for c, m in S6.targets()])
def test_registry_composition_at_the_anchor(case, model, record_property):
    out = validate_g6(case, model, S6.g3_gate(case, model), golden=S6.GOLDEN, data_root=None)
    for key, e in out["items"].items():
        assert np.isfinite(e["abs_delta"]), (case, model, key)              # never a NaN pass
        s2_key = key.replace("_", "/") if key != "lnL" else key
        record(record_property, f"galactic_binary/{case}/quad_logdet/{model}/composition/{s2_key}",
               e["abs_delta"], e["gate"])
        S6.record(record_property, f"galactic_binary/{case}/{model}/g6/{key}", e["abs_delta"], e["gate"])
    assert out["ok"], out["problems"]
    if recording_build() and _gb_recording():
        assert out["bitwise"], out["items"]


def _gb_recording() -> bool:
    """The GB goldens' build (python, machine, numpy, scipy) is this one."""
    import json

    rec = json.loads((S6.GOLDEN / "galactic_binary" / "products.json").read_text())["build"]
    now = runtime_fingerprint()
    return all(rec.get(k) == now.get(k) for k in ("python", "machine", "numpy", "scipy"))
