"""S6 gate helpers shared by tests/golden/test_s6_*.py, tests/golden/test_s2_g6_gb_registry.py and
scripts/golden/s6_report.py (spec section 7.6, S6: the galactic binary on NumPy).

Every comparison of the S6 gate is one junit property ``s6`` whose value is
``{"item": ..., "observed": ..., "gate": ...}`` (the S2 ``g6`` / S3 ``s3`` convention).
Item names (``<root>`` = ``galactic_binary/<case>/<model>``):

* ``<root>/g6/{c0_quad,c0_logdet,c1_quad,c1_logdet,lnL}``: the anchor composition against G6
  (``quad_logdet`` at 1e-12 relative, lnL at the approved G3 gate);
* ``<root>/<kind>/<run>/row<r>`` (``kind`` ``g3``: binary_galactica ``fourier`` templates;
  ``g3_gbonfly``: the test-only gbonfly adapter): |release - stored chain lnlike| at the G3
  gate; ``.../row<r>/lineage_C``: |release - lineage C NumPy| (``g3_gbonfly`` on the recording
  build: gate 0.0; otherwise the G3 gate); ``<root>/<kind>/{mask,counters}`` (gate 0);
* ``<root>/g4/<point>/mask`` and ``<root>/g4/cache/<point>/mask`` (gate 0: the reference's and
  the expected -inf mask, no NaN), ``<root>/g4/<point>`` and ``<root>/g4/cache/<point>`` at the
  well-conditioned finite points (1e-10 + 1e-12 |lnL|), ``<root>/g4/counters`` and
  ``<root>/g4/cache/counters`` (gate 0) -- 3- and 6-month targets;
* ``galactic_binary/<case>/g5/<item>``: the data products (``G5_KEYS``).

The G6 items come with S6's closing of the S2 gate (F-C7); the G3/G4/G5 inventory and
recorders with the S6 gate tests (F-C8).

Nothing here imports the release packages: the report runs without them on the path.
"""

from __future__ import annotations

import json

from scripts.golden.paths import GOLDEN

PROPERTY = "s6"
#: ``wdm_models.registry.TARGETS["galactic_binary"]`` (pinned against it by
#: tests/golden/test_s6_report.py), repeated so the report imports without the release packages.
MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact", "mid_raw", "mid_cut",
          "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad", "tri_exact", "exact")
TARGETS = {"gb_3mo": MODELS, "gb_6mo": MODELS,
           "gb_12mo": ("whittle_raw", "band_raw", "mid_raw", "tri_raw", "band_pad", "mid_pad", "tri_pad", "exact")}
#: Heavy targets (dense P = 2700 factorisations; the 6-month tri pad): ``slow`` in every S6 test.
SLOW = {("gb_12mo", "tri_raw"), ("gb_12mo", "band_pad"), ("gb_12mo", "mid_pad"), ("gb_12mo", "tri_pad"),
        ("gb_6mo", "tri_pad")}
G6_KEYS = ("c0_quad", "c0_logdet", "c1_quad", "c1_logdet", "lnL")


def targets() -> list[tuple[str, str]]:
    return [(c, m) for c, ms in TARGETS.items() for m in ms]


def record(record_property, item: str, observed: float, gate: float) -> None:
    record_property(PROPERTY, json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


def g3_gate(case: str, model: str) -> float:
    """The approved G3 NumPy gate (uniform 1e-8)."""
    tol = json.loads((GOLDEN / "galactic_binary" / "tolerances_measured.json").read_text())
    return float(tol["approved"]["G3"]["numpy"][f"{case}/{model}"])


def g6_items() -> list[str]:
    return [f"galactic_binary/{c}/{m}/g6/{k}" for c, m in targets() for k in G6_KEYS]
