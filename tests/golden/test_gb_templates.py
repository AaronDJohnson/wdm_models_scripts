"""F-C5: the test-only gbonfly adapter (spec section 7.6, S6) -- lineage C computes gbonfly store
templates on demand (``workers/gbw_templates.py``) and streams them in
(``scripts.golden.gb_templates``); nothing is stored in golden/.  The streamed template at
the injection is the golden gbonfly store template bit for bit; a row that was not streamed
raises."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.golden import gb_templates as G
from scripts.golden.paths import GOLDEN, legacy_available

pytestmark = pytest.mark.golden
requires_legacy = [pytest.mark.legacy,
                   pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")]


def _inject(case: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(GOLDEN.parent / "data" / "galactic_binary" / "prepared" / f"{case}.npz") as z:
        p8 = np.array(z["inject_glass"])
    with np.load(GOLDEN / "galactic_binary" / f"intermediates_{case}.npz") as z:
        return p8, np.array(z["store_template"])


def test_the_table_is_an_exact_fail_closed_lookup():
    P8 = np.arange(16.0).reshape(2, 8)
    H = np.arange(2 * 2 * 3).reshape(2, 2, 3) + 0j
    t = G.TemplateTable(P8, H)
    assert len(t) == 2 and np.array_equal(t(P8[1]), H[1])
    assert not t(P8[1]).flags.writeable                   # a lookup cannot corrupt the table
    H[1] += 1                                             # nor can the caller's array afterwards
    assert np.array_equal(t(P8[1]), H[1] - 1)
    with pytest.raises(KeyError, match="no streamed gbonfly template"):
        t(P8[1] + np.r_[np.zeros(7), 1e-12])
    with pytest.raises(ValueError, match="pair up"):
        G.TemplateTable(P8, H[:1])


def test_refused_without_the_frozen_trees(monkeypatch):
    monkeypatch.setattr(G, "legacy_available", lambda: False)
    with pytest.raises(RuntimeError, match="frozen legacy trees"):
        G.streamed_templates([("gb_3mo", np.zeros((1, 8)))])


@pytest.mark.parametrize("case", [pytest.param(c, marks=requires_legacy) for c in ("gb_3mo", "gb_12mo")])
def test_streamed_injection_is_the_golden_gbonfly_template(case):
    p8, h = _inject(case)
    rows = np.stack([p8, p8 + np.r_[1e-3, np.zeros(7)]])
    tables, meta = G.streamed_templates([(case, rows), (case, rows[:1])])
    assert len(tables[case]) == 2                          # merged, duplicates computed once
    assert np.array_equal(tables[case](p8), h)
    info = meta["jobs"][case]
    assert (info["n"], info["store_n"]) == (2, h.shape[1]) and meta["lineage"] == "C"
