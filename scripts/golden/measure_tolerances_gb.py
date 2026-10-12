"""P0 measurement of the galactic-binary gates (spec 7.3) from the captured G3/G4 goldens.

    $PY -m scripts.golden.measure_tolerances_gb [--replace]

Reads ``golden/galactic_binary/chain_rows/**`` and ``golden/galactic_binary/stress/**`` (every
tier present), writes ``golden/galactic_binary/tolerances_measured.json`` and merges a summary
into ``golden/MANIFEST.json["tolerances"]["galactic_binary"]`` with status
``"measured; pending user approval"``. ``measure(golden_dir)`` is the hook for Section A's
``measure_tolerances.py``.

Rules (each gate = max(10 x observed, floor), floors from spec 7.3):

* G3 rows, NumPy: observed |lineage-C NumPy - stored| per target; floor 1e-9 (the proposal).
* G3 rows, JAX (CPU): observed |lineage-C JAX - stored| per target; floor 1e-8.
* G4 stress, well-conditioned finite points (PD margin >= 1e-6): |JAX - NumPy reference|, gate
  max(10 x observed, 1e-10), pooled over all targets (``gate_jax_well_conditioned``) and per
  target (``gate_jax_well_conditioned_per_target``). The 1e-6 ceiling (spec 7.3, today's
  ``test_jax``) is a spec limit, not a clamp: if 10 x observed exceeds it, the gate cannot be both
  >= 10 x observed and <= 1e-6, so ``measure`` raises ``ValueError`` (S0 fails, nothing written)
  instead of capping the gate below its own data. |NumPy cache (expanded) - reference| against
  1e-10 + 1e-12 |lnL|.
* Marginal points are those with ``|margin| < marginal_threshold``, the threshold each stress file
  records (the worker's ``MARGINAL``, 1e-10 in every capture). The PD margin is the worker's
  (gbw_stress ``_margin``): ``lambda_min / max|lambda|`` for a dense covariance, 1.0 for a diagonal
  one with every entry > 0, else ``min / max|.|``.
* G4 stress, ill-conditioned non-marginal points (threshold <= margin < 1e-6): reported in units of
  ``eps |lnL| / margin`` (condition-scaled), a proposal for the user, not a gate yet.
* -inf masks on non-marginal points must match the worker's ``expected_neg_inf = source_out or
  margin < 0``: -inf when the covariance is not PD (margin <= -threshold) or the source is outside
  the GLASS box, finite otherwise. Every evaluator is checked, the NumPy reference included; each
  wrong one is one mismatch (count reported; any mismatch fails S0). A finite lnL at a non-PD point
  is therefore a mismatch, never a delta. A NaN or +inf lnL at a non-marginal point (JAX Cholesky
  returns NaN rather than raising) is recorded as a mismatch too, never folded into a maximum.
  Deltas are taken only where the mask is finite.

Inputs are validated, never defaulted: every row file and stress file must have been captured
with both backends (``numpy`` and ``jax``), every row must carry both deltas, every G3 delta must
be finite, and every stress file must record a finite positive ``marginal_threshold``. Every
stress point, marginal or not, must carry ``lnl_jax`` (and ``lnl_cache`` when the target has a
cache class), ``source_out``, ``expected_neg_inf`` and a finite PD margin, and its stored flags
must follow their rules: ``marginal == (|margin| < marginal_threshold)`` and ``expected_neg_inf ==
(source_out or margin < 0)``. A marginal point is exempt from the mask and delta checks only: its
lnL values must still be finite or exactly -inf (spec G4). Otherwise ``measure`` raises
``ValueError``.
``main`` builds and hashes the output in memory and refuses (``SystemExit``, nothing written) when
``golden/MANIFEST.json`` records other bytes for it, unless ``--replace``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from scripts.golden import manifest_merge
from scripts.golden.paths import GOLDEN

EPS = 2.220446049250313e-16
FLOOR = {"G3_numpy": 1e-9, "G3_jax": 1e-8, "G4_jax_floor": 1e-10, "G4_jax_ceiling": 1e-6,
         "G4_numpy_abs": 1e-10, "G4_numpy_rel": 1e-12}
WELL_CONDITIONED = 1e-6
#: Every G3/G4 gate is per backend, so a capture that skipped one cannot be measured.
BACKENDS = ("numpy", "jax")
#: The only valid lnL values; NaN or +inf at a compared point is a mismatch.
VALID = ("finite", "-inf")
OUT = "galactic_binary/tolerances_measured.json"


def _gate(observed: float, floor: float) -> float:
    return max(10.0 * observed, floor)


def _fail(what: str, problems: list[str]) -> None:
    if problems:
        raise ValueError(f"{what}: {len(problems)} problem(s): " + "; ".join(problems[:10])
                         + (" ..." if len(problems) > 10 else ""))


def _missing_backends(r: dict) -> list[str]:
    return [b for b in BACKENDS if b not in r.get("backends", ())]


def measure_rows(gb: Path) -> dict:
    per, problems = {}, []
    for path in sorted((gb / "chain_rows").glob("*/*.json")):
        if path.name == "index.json":
            continue
        r = json.loads(path.read_text())
        where = path.relative_to(gb).as_posix()
        if missing := _missing_backends(r):
            problems.append(f"{where}: backend(s) {missing} not captured")
        t = per.setdefault(f"{r['case']}/{r['model']}", {"runs": set(), "rows": 0, "numpy": 0.0,
                                                          "jax": 0.0, "tiers": set()})
        t["runs"].add(r["run"])
        t["tiers"].add(r["tier"])
        t["rows"] += len(r["rows"])
        for row in r["rows"]:
            for b in BACKENDS:
                d = row.get(f"abs_delta_{b}")
                if d is None:
                    problems.append(f"{where}: row {row['row']} has no abs_delta_{b}")
                elif not math.isfinite(d):
                    problems.append(f"{where}: row {row['row']} abs_delta_{b} = {d}")
                else:
                    t[b] = max(t[b], d)
    _fail("G3 chain rows", problems)
    out = {}
    for k, t in sorted(per.items()):
        out[k] = {"runs": len(t["runs"]), "rows": t["rows"], "tiers": sorted(t["tiers"]),
                  "observed_numpy": t["numpy"], "observed_jax": t["jax"],
                  "gate_numpy": _gate(t["numpy"], FLOOR["G3_numpy"]),
                  "gate_jax": _gate(t["jax"], FLOOR["G3_jax"]),
                  "exceeds_proposal_numpy": 10.0 * t["numpy"] > FLOOR["G3_numpy"],
                  "exceeds_proposal_jax": 10.0 * t["jax"] > FLOOR["G3_jax"]}
    return out


def _kind(v: float) -> str:
    """``finite`` or ``-inf`` (the only valid lnL values, ``VALID``), else ``nan`` / ``+inf``."""
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "+inf"
    return "finite"


def _threshold(r: dict) -> float | None:
    """The file's ``marginal_threshold`` if it is a finite positive number, else None."""
    thr = r.get("marginal_threshold")
    ok = isinstance(thr, (int, float)) and not isinstance(thr, bool) and math.isfinite(thr) and thr > 0
    return float(thr) if ok else None


def _point_problems(p: dict, needed: tuple[str, ...], thr: float) -> list[str]:
    """Why a stress point cannot be measured: missing fields, a non-finite margin, stored
    ``marginal`` / ``expected_neg_inf`` flags that disagree with their rules, or (marginal points)
    an lnL that is neither finite nor -inf."""
    fields = ("lnl_ref",) + needed + ("pd_margin", "marginal", "source_out", "expected_neg_inf")
    if absent := [name for name in fields if name not in p]:
        return [f"has no {absent}"]
    margin = p["pd_margin"]
    if not math.isfinite(margin):
        return [f"pd_margin = {margin}"]
    out = []
    if p["marginal"] != (abs(margin) < thr):
        out.append(f"marginal={p['marginal']} but |pd_margin| = {abs(margin):.3e} "
                   f"{'<' if abs(margin) < thr else '>='} {thr:g}")
    if p["expected_neg_inf"] != bool(p["source_out"] or margin < 0):
        out.append(f"expected_neg_inf={p['expected_neg_inf']} but source_out {p['source_out']}, "
                   f"pd_margin {margin:.3e}")
    if p["marginal"]:                       # exempt from the mask/delta checks, not from spec G4
        out += [f"is marginal but {name} = {p[name]} (finite or -inf required)"
                for name in ("lnl_ref", "lnl_cache", "lnl_jax") if name in p and _kind(p[name]) not in VALID]
    return out


def measure_stress(gb: Path) -> dict:
    per, mismatches, problems = {}, [], []
    for path in sorted((gb / "stress").glob("*/*.json")):
        r = json.loads(path.read_text())
        where = path.relative_to(gb).as_posix()
        key = f"{r['case']}/{r['model']}/{r['tier']}"
        if missing := _missing_backends(r):
            problems.append(f"{where}: backend(s) {missing} not captured")
        thr = _threshold(r)
        if thr is None:
            problems.append(f"{where}: marginal_threshold = {r.get('marginal_threshold')!r} "
                            "(a finite positive number is required)")
            continue
        needed = ("lnl_jax",) + (("lnl_cache",) if r.get("cache_class") is not None else ())
        t = {"points": r["n_points"], "marginal": 0, "marginal_threshold": thr, "jax_well": 0.0,
             "cache_well": 0.0, "cache_well_rel": 0.0, "jax_cond": 0.0, "cache_cond": 0.0,
             "counters_equal": None, "cache_within_numpy_gate": True}
        for p in r["points"]:
            if bad := _point_problems(p, needed, thr):
                problems += [f"{where}: point {p['name']} {b}" for b in bad]
                continue
            if p["marginal"]:               # exempt from the mask and delta checks only
                t["marginal"] += 1
                continue
            margin = p["pd_margin"]
            # The worker's mask (gbw_stress: expected_neg_inf = source_out or margin < 0). Every
            # evaluator is checked against it, the NumPy reference included.
            expected = "-inf" if (p["source_out"] or margin < 0) else "finite"
            why = f"expected {expected} (pd_margin {margin:.3e}, source_out {p['source_out']})"
            ref = p["lnl_ref"]
            k_ref = _kind(ref)
            if k_ref not in VALID:
                mismatches.append(f"{key}:{p['name']}:lnl_ref is {k_ref}")
            elif k_ref != expected:
                mismatches.append(f"{key}:{p['name']}:lnl_ref is {k_ref}, {why}")
            for name in ("lnl_cache", "lnl_jax"):
                if name not in p:
                    continue
                k = _kind(p[name])
                if k not in VALID:
                    mismatches.append(f"{key}:{p['name']}:{name} is {k} (ref {k_ref})")
                elif k != expected and k != k_ref:     # disagrees with a correct reference
                    mismatches.append(f"{key}:{p['name']}:{name}")
                elif k != expected:                    # agrees with a wrong reference
                    mismatches.append(f"{key}:{p['name']}:{name} is {k}, {why}")
            # Deltas only where every evaluator must be finite: PD (margin >= thr) and in the box.
            if expected != "finite" or k_ref != "finite":
                continue
            for name, tag in (("lnl_jax", "jax"), ("lnl_cache", "cache")):
                if name not in p or _kind(p[name]) != "finite":
                    continue
                d = abs(p[name] - ref)
                if margin >= WELL_CONDITIONED:
                    t[f"{tag}_well"] = max(t[f"{tag}_well"], d)
                    if tag == "cache":
                        t["cache_well_rel"] = max(t["cache_well_rel"], d / max(1.0, abs(ref)))
                        t["cache_within_numpy_gate"] &= (
                            d <= FLOOR["G4_numpy_abs"] + FLOOR["G4_numpy_rel"] * abs(ref))
                elif thr <= margin < WELL_CONDITIONED:
                    t[f"{tag}_cond"] = max(t[f"{tag}_cond"], d / (EPS * abs(ref) / margin))
                else:                                  # unreachable: expected finite => margin >= thr
                    raise AssertionError(f"{key}:{p['name']}: pd_margin {margin} reached the delta branch")
        if r["counters_cache"] is not None and r["counters_jax"] is not None:
            t["counters_equal"] = r["counters_cache"] == r["counters_jax"]
        per[key] = t
    _fail("G4 stress", problems)
    jax_well = max((t["jax_well"] for t in per.values()), default=0.0)
    # The ceiling is a spec limit, not a clamp: min(ceiling, 10 x observed) would put the gate
    # below 10 x observed (observed 5e-7 -> gate 1e-6) or even below the observation itself
    # (observed 3e-6 -> gate 1e-6) without any error.
    _fail("G4 stress: a proposed JAX well-conditioned gate fails its own data (10 x observed > "
          f"the spec 7.3 ceiling {FLOOR['G4_jax_ceiling']:g}; S0 fails, report to the user)",
          [f"{k}: observed {t['jax_well']:.3e}, 10 x observed {10.0 * t['jax_well']:.3e}"
           for k, t in per.items() if 10.0 * t["jax_well"] > FLOOR["G4_jax_ceiling"]])
    gate_jax = _gate(jax_well, FLOOR["G4_jax_floor"])
    assert gate_jax <= FLOOR["G4_jax_ceiling"]
    return {"per_target": per, "mask_mismatches": mismatches,
            "observed_jax_well_conditioned": jax_well,
            "gate_jax_well_conditioned": gate_jax,
            "gate_jax_well_conditioned_per_target": {k: _gate(t["jax_well"], FLOOR["G4_jax_floor"])
                                                     for k, t in per.items()},
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


def render(result: dict) -> bytes:
    """The exact bytes of ``tolerances_measured.json`` (NaN/inf refused: strict JSON)."""
    return (json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    return cli(argv)


def cli(argv=None) -> int:
    """``main`` without the safe-path guard (the tests call it in-process)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    # Measured, serialised and hashed in memory first; nothing is written until the manifest check
    # has passed (as in summarise_chains.main / capture_payload.capture), so a refused run leaves
    # golden/ untouched.
    result = measure(GOLDEN)
    data = render(result)
    files = {OUT: hashlib.sha256(data).hexdigest()}
    bad = manifest_merge.conflicts(manifest_merge.load(GOLDEN / "MANIFEST.json"), files)
    if bad and not a.replace:
        raise SystemExit(f"golden/MANIFEST.json holds other bytes for {bad}; nothing written; "
                         "rerun with --replace only if the change is intended")
    (GOLDEN / OUT).write_bytes(data)
    if manifest_merge.sha256_file(GOLDEN / OUT) != files[OUT]:
        raise RuntimeError(f"{OUT}: written bytes differ from the measurement")
    manifest_merge.merge_entries(GOLDEN / "MANIFEST.json", files,
                                 provenance={OUT: {"script": "scripts/golden/measure_tolerances_gb.py"}},
                                 tolerances={"galactic_binary": dict(result["summary"],
                                                                     status=result["status"],
                                                                     file=OUT)},
                                 replace=a.replace)
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
