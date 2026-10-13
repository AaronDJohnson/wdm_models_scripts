"""S3 gate helpers shared by tests/golden/test_s3_*.py and scripts/golden/s3_report.py.

Every G1/G2/G5 comparison of the S3 gate is one junit property ``s3`` whose value is
``{"item": ..., "observed": ..., "gate": ...}`` (the S2 ``g6`` convention,
``scripts.golden.s2_cases.record``).  Item names:

* ``sinusoid/<case>/<model>/g1/point<i>``: |release - G1 reference| at reference point ``i``;
  gate 0.0 on the build that recorded lineage A (spec section 3.6: bitwise), else the
  approved NumPy reference gate (``atol`` near, ``atol + rtol |ref|`` stress);
* ``sinusoid/<case>/<model>/g2/row<r>``: |release - stored chain lnlike| at chain row ``r``
  (approved chain-row gate);
* on the recording build also ``.../{point<i>,row<r>}/lineage_A``: |release - lineage A's value|
  (G1 gate 0.0; G2 the model's approved NumPy reference gate);
* ``sinusoid/<case>/<model>/{g1,g2}/mask``: -inf mismatches + NaN/+inf values (gate 0);
* ``sinusoid/<case>/<model>/{g1,g2}/counters``: last_failed / counter inconsistencies (gate 0);
* ``sinusoid/g5/...``: the data products (:data:`G5_ITEMS`).
"""

from __future__ import annotations

import json

import numpy as np

from scripts.golden.paths import GOLDEN

PROPERTY = "s3"
#: ``scripts.sinusoid.validate.CASES`` / ``MODELS`` (spec section 5.3 order), repeated here so
#: that ``scripts.golden.s3_report`` imports without the release packages on the path.
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact",
          "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad",
          "tri_exact", "exact")
#: Targets cheap enough for the default lane (< 2 s per target for G1 + G2 on the recording
#: build); tri_*, *_pad and *_exact run under ``slow`` (2-41 s per target).
CHEAP = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "mid_raw", "mid_cut", "exact")
#: G1 reference points that are -inf per case (the PD truncation, spec section 4.6).
NEG_INF = {"band_exact": 12, "mid_exact": 13, "tri_exact": 12}


def recording_build() -> bool:
    """True on the build whose lineage-A evaluation is golden/sinusoid/legacy_eval/A_*_fast.json."""
    from wdm_models.sampling.validate import build_fingerprint, same_build   # tests only

    rec = json.loads((GOLDEN / "sinusoid" / "legacy_eval" / "A_references_fast.json").read_text())
    return same_build(build_fingerprint("numpy"), rec["worker"]["runtime"])


def record(record_property, item: str, observed: float, gate: float) -> None:
    record_property(PROPERTY, json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


def record_target(record_property, target: dict, *, recording: bool) -> None:
    """The ``s3`` records of one ``scripts.sinusoid.validate.validate_target`` result."""
    root = f"sinusoid/{target['case']}/{target['model']}"
    for kind, label in (("g1", "point"), ("g2", "row")):
        e = target.get(kind)
        if e is None or "abs_delta" not in e:
            continue                                   # nothing evaluated: the items stay uncovered
        exact_g1 = kind == "g1" and recording
        for i, d, allowed in zip(e["index"], e["abs_delta"], e["allowed"], strict=True):
            record(record_property, f"{root}/{kind}/{label}{int(i)}", d, 0.0 if exact_g1 else allowed)
        if "lineage_A" in e:
            la = e["lineage_A"]
            for i, d, allowed in zip(e["index"], la["abs_delta"], la["allowed"], strict=True):
                record(record_property, f"{root}/{kind}/{label}{int(i)}/lineage_A", d, allowed)
        record(record_property, f"{root}/{kind}/mask", e["inf_mismatch"] + e["invalid"], 0.0)
        bad = [p for p in e["problems"] if p.startswith(("last_failed", "counter"))]
        record(record_property, f"{root}/{kind}/counters", len(bad), 0.0)


def g1g2_items(tier: str = "fast") -> list[str]:
    """Every G1/G2 item of ``tier`` (the report's inventory; lineage-A items are extra)."""
    files = json.loads((GOLDEN / "sinusoid" / "references" / "INDEX.json").read_text())["files"]
    rows = json.loads((GOLDEN / "sinusoid" / "chain_rows" / "INDEX.json").read_text())
    items: list[str] = []
    for case in CASES:
        for model in MODELS:
            root = f"sinusoid/{case}/{model}"
            entry = files[f"{case}_{model}_reference.npz"]
            idx = entry["fast_index"] if tier == "fast" else range(entry["n_points"])
            items += [f"{root}/g1/point{int(i)}" for i in idx] + [f"{root}/g1/mask", f"{root}/g1/counters"]
            with np.load(GOLDEN / "sinusoid" / "chain_rows" / f"{case}_{model}.npz") as z:
                r = z["row_index"]
            pos = rows["fast_positions"] if tier == "fast" else range(len(r))
            items += [f"{root}/g2/row{int(r[p])}" for p in pos] + [f"{root}/g2/mask", f"{root}/g2/counters"]
    return items


#: The G5 items (tests/golden/test_s3_g5_sinusoid.py records exactly these).
G5_ITEMS: tuple[str, ...] = (
    *(f"sinusoid/{c}/g5/{k}" for c in CASES for k in ("toy_data_sha256", "w_d")),
    "sinusoid/g5/cN_sha256", "sinusoid/g5/bandcov_sha256",
    "sinusoid/g5/wrap_cut", "sinusoid/g5/wrap_first_dropped_lead", "sinusoid/g5/wrap_first_dropped_trail",
    "sinusoid/g5/wrap_worst_kept",
    "sinusoid/g5/c_N", "sinusoid/g5/bandcov_key", "sinusoid/g5/bandcov_identity", "sinusoid/g5/noise_bounds",
    *(f"sinusoid/g5/describe/{k}" for k in ("raw_cells", "cut_cells", "exact_rows", "pad_cells")),
)
