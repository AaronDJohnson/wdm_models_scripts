"""P0 measurement of the galactic-binary gates (spec 7.3) from the captured G3/G4 goldens.

    $PY -P scripts/golden/measure_tolerances_gb.py [--replace]

Reads ``golden/galactic_binary/chain_rows/**`` and ``golden/galactic_binary/stress/**`` (every
tier present), writes ``golden/galactic_binary/tolerances_measured.json`` and merges a summary
into ``golden/MANIFEST.json["tolerances"]["galactic_binary"]`` with status
``"measured; pending user approval"``. ``measure(golden_dir)`` is the hook for Section A's
``measure_tolerances.py``.

Rules (each gate = max(10 x observed, floor), floors from spec 7.3):

* G3 rows, NumPy: observed |lineage-C NumPy - stored| per target; floor 1e-9 (the proposal).
* G3 rows, JAX (CPU): observed |lineage-C JAX - stored| per target; floor 1e-8.
* G4 stress, well-conditioned finite points (PD margin >= 1e-6): |JAX - NumPy reference| with the
  1e-6 ceiling, tightened to max(10 x observed, 1e-10); |NumPy cache (expanded) - reference|
  against 1e-10 + 1e-12 |lnL|.
* G4 stress, ill-conditioned non-marginal points (1e-10 <= margin < 1e-6): reported in units of
  ``eps |lnL| / margin`` (condition-scaled), a proposal for the user, not a gate yet.
* -inf masks on non-marginal points must agree (count reported; any mismatch fails S0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if not sys.flags.safe_path:
    raise SystemExit("run with python -P (PYTHONSAFEPATH)")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest_merge  # noqa: E402

REL = HERE.parents[1]
GOLDEN = REL / "golden"
EPS = 2.220446049250313e-16
FLOOR = {"G3_numpy": 1e-9, "G3_jax": 1e-8, "G4_jax_floor": 1e-10, "G4_jax_ceiling": 1e-6,
         "G4_numpy_abs": 1e-10, "G4_numpy_rel": 1e-12}
WELL_CONDITIONED = 1e-6


def _gate(observed: float, floor: float) -> float:
    return max(10.0 * observed, floor)


def measure_rows(gb: Path) -> dict:
    per = {}
    for path in sorted((gb / "chain_rows").glob("*/*.json")):
        if path.name == "index.json":
            continue
        r = json.loads(path.read_text())
        t = per.setdefault(f"{r['case']}/{r['model']}", {"runs": set(), "rows": 0, "numpy": 0.0,
                                                          "jax": 0.0, "tiers": set()})
        t["runs"].add(r["run"])
        t["tiers"].add(r["tier"])
        t["rows"] += len(r["rows"])
        for row in r["rows"]:
            t["numpy"] = max(t["numpy"], row.get("abs_delta_numpy", 0.0))
            t["jax"] = max(t["jax"], row.get("abs_delta_jax", 0.0))
    out = {}
    for k, t in sorted(per.items()):
        out[k] = {"runs": len(t["runs"]), "rows": t["rows"], "tiers": sorted(t["tiers"]),
                  "observed_numpy": t["numpy"], "observed_jax": t["jax"],
                  "gate_numpy": _gate(t["numpy"], FLOOR["G3_numpy"]),
                  "gate_jax": _gate(t["jax"], FLOOR["G3_jax"]),
                  "exceeds_proposal_numpy": 10.0 * t["numpy"] > FLOOR["G3_numpy"],
                  "exceeds_proposal_jax": 10.0 * t["jax"] > FLOOR["G3_jax"]}
    return out


def measure_stress(gb: Path) -> dict:
    per, mismatches = {}, []
    for path in sorted((gb / "stress").glob("*/*.json")):
        r = json.loads(path.read_text())
        key = f"{r['case']}/{r['model']}/{r['tier']}"
        t = {"points": r["n_points"], "marginal": 0, "jax_well": 0.0, "cache_well": 0.0,
             "cache_well_rel": 0.0, "jax_cond": 0.0, "cache_cond": 0.0, "counters_equal": None,
             "cache_within_numpy_gate": True}
        for p in r["points"]:
            if p["marginal"]:
                t["marginal"] += 1
                continue
            ref = p["lnl_ref"]
            for name in ("lnl_cache", "lnl_jax"):
                if name in p and math.isinf(p[name]) != math.isinf(ref):
                    mismatches.append(f"{key}:{p['name']}:{name}")
            if math.isinf(ref):
                continue
            for name, tag in (("lnl_jax", "jax"), ("lnl_cache", "cache")):
                if name not in p or math.isinf(p[name]):
                    continue
                d = abs(p[name] - ref)
                if p["pd_margin"] >= WELL_CONDITIONED:
                    t[f"{tag}_well"] = max(t[f"{tag}_well"], d)
                    if tag == "cache":
                        t["cache_well_rel"] = max(t["cache_well_rel"], d / max(1.0, abs(ref)))
                        t["cache_within_numpy_gate"] &= (
                            d <= FLOOR["G4_numpy_abs"] + FLOOR["G4_numpy_rel"] * abs(ref))
                else:
                    t[f"{tag}_cond"] = max(t[f"{tag}_cond"], d / (EPS * abs(ref) / p["pd_margin"]))
        if r["counters_cache"] is not None and r["counters_jax"] is not None:
            t["counters_equal"] = r["counters_cache"] == r["counters_jax"]
        per[key] = t
    jax_well = max((t["jax_well"] for t in per.values()), default=0.0)
    return {"per_target": per, "mask_mismatches": mismatches,
            "observed_jax_well_conditioned": jax_well,
            "gate_jax_well_conditioned": min(FLOOR["G4_jax_ceiling"],
                                             _gate(jax_well, FLOOR["G4_jax_floor"])),
            "observed_cache_well_conditioned": max((t["cache_well"] for t in per.values()), default=0.0),
            "cache_within_numpy_gate": all(t["cache_within_numpy_gate"] for t in per.values()),
            "condition_scaled_jax_max": max((t["jax_cond"] for t in per.values()), default=0.0),
            "condition_scaled_cache_max": max((t["cache_cond"] for t in per.values()), default=0.0)}


def measure(golden_dir: Path = GOLDEN) -> dict:
    gb = Path(golden_dir) / "galactic_binary"
    rows, stress = measure_rows(gb), measure_stress(gb)
    return {"schema": 1, "status": "measured; pending user approval", "floors": FLOOR,
            "well_conditioned_margin": WELL_CONDITIONED, "G3": rows, "G4": stress,
            "summary": {
                "G3_targets": len(rows),
                "G3_runs": sum(v["runs"] for v in rows.values()),
                "G3_max_observed_numpy": max(v["observed_numpy"] for v in rows.values()),
                "G3_max_observed_jax": max(v["observed_jax"] for v in rows.values()),
                "G3_exceeds_proposal_numpy": sorted(k for k, v in rows.items()
                                                    if v["exceeds_proposal_numpy"]),
                "G3_exceeds_proposal_jax": sorted(k for k, v in rows.items() if v["exceeds_proposal_jax"]),
                "G4_files": len(stress["per_target"]),
                "G4_mask_mismatches": len(stress["mask_mismatches"]),
                "G4_gate_jax_well_conditioned": stress["gate_jax_well_conditioned"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    result = measure(GOLDEN)
    rel = "galactic_binary/tolerances_measured.json"
    (GOLDEN / rel).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest_merge.merge_entries(GOLDEN / "MANIFEST.json",
                                 {rel: manifest_merge.sha256_file(GOLDEN / rel)},
                                 provenance={rel: {"script": "scripts/golden/measure_tolerances_gb.py"}},
                                 tolerances={"galactic_binary": dict(result["summary"],
                                                                     status=result["status"],
                                                                     file=rel)},
                                 replace=a.replace)
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
