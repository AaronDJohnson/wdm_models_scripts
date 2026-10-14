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


G3_KINDS = ("g3", "g3_gbonfly")
G5_KEYS = ("prepared_sha256", "describe", "template_store_sum_h2", "template_store_re_zh", "template_window_sum_h2",
           "template_window_re_zh", "prior_box")
STRESS_CASES = ("gb_3mo", "gb_6mo")
#: Points with a PD margin at least this are gated (approved G4 rule).
WELL = 1e-6
CACHED = ("tri_raw", "tri_cut", "tri_pad", "tri_exact", "band_exact", "mid_exact", "band_pad", "mid_pad", "exact")


def _json(rel: str) -> dict:
    return json.loads((GOLDEN / "galactic_binary" / rel).read_text())


def g3_items(kind: str) -> list[str]:
    """Every fast-tier G3 row item of ``kind`` (plus ``lineage_C``), mask and counters."""
    items: list[str] = []
    for case, model in targets():
        root = f"galactic_binary/{case}/{model}/{kind}"
        for run in _json(f"chain_rows/{case}/index.json")["targets"][model]:
            for r in _json(f"chain_rows/{case}/{run}.fast.json")["rows"]:
                items += [f"{root}/{run}/row{r['row']}", f"{root}/{run}/row{r['row']}/lineage_C"]
        items += [f"{root}/mask", f"{root}/counters"]
    return items


def g4_items() -> list[str]:
    """Every fast-tier G4 item: masks and counters always, deltas at the well-conditioned points
    whose reference (cache) value is finite."""
    items: list[str] = []
    for case, model in targets():
        if case not in STRESS_CASES:
            continue
        st = _json(f"stress/{case}/{model}.fast.json")
        root = f"galactic_binary/{case}/{model}/g4"
        for sub, key in (("", "lnl_ref"), ("cache/", "lnl_cache")):
            if sub and model not in CACHED:
                continue
            for p in st["points"]:
                items.append(f"{root}/{sub}{p['name']}/mask")
                v = p[key]
                if not isinstance(v, str) and v == v and abs(v) != float("inf") and p["pd_margin"] >= WELL:
                    items.append(f"{root}/{sub}{p['name']}")
            items.append(f"{root}/{sub}counters")
    return items


def g5_items() -> list[str]:
    return [f"galactic_binary/{c}/g5/{k}" for c in TARGETS for k in G5_KEYS]


def inventory() -> list[str]:
    """The S6 gate: G3 (both template sources), G4, G5, G6 -- fast tier."""
    return g3_items("g3") + g3_items("g3_gbonfly") + g4_items() + g5_items() + g6_items()


def record_g3(record_property, target: dict, kind: str, *, exact_lineage: bool) -> None:
    """The ``s6`` records of one ``validate.run`` target's G3 entry."""
    e = target.get("g3")
    if e is None or "runs" not in e:
        return                                          # nothing evaluated: the items stay uncovered
    root = f"galactic_binary/{target['case']}/{target['model']}/{kind}"
    mask = counters = 0
    for run, r in e["runs"].items():
        for row, d, allowed, dl in zip(r["rows"], r["abs_delta"], r["allowed"], r["lineage_c"]["abs_delta"],
                                       strict=True):
            record(record_property, f"{root}/{run}/row{int(row)}", d, allowed)
            record(record_property, f"{root}/{run}/row{int(row)}/lineage_C", dl, 0.0 if exact_lineage else allowed)
        mask += r["inf_mismatch"] + r["invalid"]
        counters += sum(p.startswith(("counter", "last_failed")) for p in r["problems"])
    record(record_property, f"{root}/mask", mask, 0.0)
    record(record_property, f"{root}/counters", counters, 0.0)


def record_g4(record_property, target: dict) -> None:
    """The ``s6`` records of one ``validate.run`` target's G4 entry (reference, cache)."""
    e = target.get("g4")
    if e is None or "reference" not in e:
        return
    root = f"galactic_binary/{target['case']}/{target['model']}/g4"
    for sub, key in (("", "reference"), ("cache/", "cache")):
        if key not in e:
            continue
        c = e[key]
        for i, name in enumerate(e["names"]):
            record(record_property, f"{root}/{sub}{name}/mask", float(bool(c["mask_bad"][i])), 0.0)
            if c["well"][i]:
                record(record_property, f"{root}/{sub}{name}", c["abs_delta"][i], c["allowed"][i])
        probs = [p for p in e["problems"] if (p.startswith("cache ") if sub else p.startswith(("counter", "last")))]
        record(record_property, f"{root}/{sub}counters", len(probs), 0.0)


def record_g5(record_property, case: str, entry: dict) -> None:
    for k, v in entry.get("items", {}).items():
        record(record_property, f"galactic_binary/{case}/g5/{k}", v["observed"], v["gate"])


#: The onfly audit's junit property (reported by s6_report, never gated).
AUDIT_PROPERTY = "s6_audit"
