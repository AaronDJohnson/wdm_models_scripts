"""F-C8 (S6 gate, G5): per case the prepared file's sha256 (release pin = data/MANIFEST.json = every
G3/G4 golden's record), ``registry.describe`` == lineage C's ``describe`` for every target, the
``fourier`` template sums at the injection (store and window: measured <= 3.4e-16 relative;
gate 1e-10 on the recording build, else 1e-8) and the GLASS prior box (bitwise)."""

from __future__ import annotations

import pytest

from scripts.galactic_binary import validate as V
from scripts.golden import s6_cases as S6

pytestmark = pytest.mark.golden


@pytest.mark.parametrize("case", list(S6.TARGETS))
def test_g5_products(case, record_property):
    rep = V.run(cases=[case], goldens=["g5"])
    entry = rep["g5"][case]
    S6.record_g5(record_property, case, entry)
    assert rep["ok"], rep["problems"]
    assert set(entry["items"]) == set(S6.G5_KEYS)
    assert all(entry["items"][k]["observed"] == 0.0 for k in ("prepared_sha256", "describe", "prior_box"))
