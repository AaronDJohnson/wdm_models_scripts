"""F-A9 (S5 closes its part of the S2 gate, spec section 7.2 G6): the GB store template at the
injection, ``wdm_models.galactic_binary.waveform.GBTemplate(backend="fourier")`` on the store,
reproduces ``golden/galactic_binary/intermediates_<case>.npz["store_template"]`` (gbonfly,
lineage C) to 1e-12 of its peak -- binary_galactica 0.5.0 is never bitwise with gbonfly
(~2e-16: Taylor sinc, scalar contractions; spec section 5.2).

The S2 gate (``scripts/golden/s2_report.py --full --strict``, glob ``tests/golden/test_s2_g6_*.py``)
runs this file: its records cover the three ``galactic_binary/<case>/arrays/store_template``
items, which ``s2_report.deferred()`` no longer lists.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden.s2_cases import gb_inputs, record

pytestmark = pytest.mark.golden
CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
GATE = 1e-12


@pytest.mark.parametrize("case", CASES)
def test_store_template_at_the_injection(case, record_property):
    pytest.importorskip("binary_galactica")
    from wdm.backend import enable_x64

    from wdm_models.galactic_binary.prepared import PreparedCase
    from wdm_models.galactic_binary.waveform import GBTemplate

    enable_x64()
    prep = PreparedCase.load(case)
    ref = gb_inputs(case).npz["store_template"]
    h = GBTemplate(prep, backend="fourier", check=False)(np.asarray(prep.inject))
    assert h.shape == ref.shape and np.all(np.isfinite(h))
    rel = float(np.max(np.abs(h - ref)) / np.max(np.abs(ref)))
    record(record_property, f"galactic_binary/{case}/arrays/store_template", rel, GATE)
    assert rel <= GATE
