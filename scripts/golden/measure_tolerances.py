"""P0 measurement of the sinusoid gates (spec §7.3; Tasks A10, F4).

    $PY -m scripts.golden.measure_tolerances --family sinusoid [--replace]

Reads golden/sinusoid/legacy_eval/{A,B}_{references,chain_rows}_<tier>.json. Gates are split
like toy_models/sinusoid/gpu/run.py::validate_reference: near points |delta| <= atol;
stress points |delta| <= atol + rtol |lnL_ref|; -inf masks identical.

Writes golden/sinusoid/tolerances_measured.json with three separate records:

* ``observed``: the measured maxima per lineage/backend;
* ``proposed``: the original P0 proposal, max(FACTOR x observed, floor) with the spec §7.3 floors
  (every proposal equals its floor);
* ``approved``: the OPERATIVE gates, the ones every later part reads. The user approved them on
  2026-10-12 (``DECISIONS``): every proposed gate as proposed, plus D1 ("Add rtol 1e-12 at
  stress"), which gives the NumPy reference gates of {band,mid,tri}_pad and exact rtol 1e-12 at
  stress points (atol unchanged at near points), like the edge-exact models.

Each gate, proposed or approved, must pass its own data; otherwise nothing is written
(``AssertionError``). The decisions live here, never in hand-edited JSON.

The command line measures, renders and hashes in memory first and refuses (``SystemExit``, nothing
written) when golden/MANIFEST.json records other bytes for the file, unless ``--replace``; it then
writes the file and merges ``manifest_summary`` into ``MANIFEST["tolerances"]["sinusoid"]``.
``measure_sinusoid()`` (the capture hook, ``capture.FAMILIES``) writes the file only; capture pins it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

from scripts.golden import manifest_merge
from scripts.golden.goldens_io import jsonable, read_json, to_float
from scripts.golden.sinusoid_goldens import EVAL, GS, MODELS

FACTOR = 10.0
BITWISE = ("whittle_raw", "whittle_cut", "band_raw", "mid_raw", "band_cut", "tri_raw", "mid_cut", "tri_cut")
PADS = ("band_pad", "mid_pad", "tri_pad")
EDGE = ("band_exact", "mid_exact", "tri_exact")
#: spec §7.3 floors: "numpy" gates the release NumPy path against lineage A, "jax" the release
#: JAX path against lineage B; chain rows always use the JAX floor (the chains ran JAX).
FLOORS = {
    "numpy": {**{m: {"atol": 1e-9, "rtol": 0.0} for m in BITWISE + PADS},
              **{m: {"atol": 1e-8, "rtol": 1e-12} for m in EDGE},
              "exact": {"atol": 1e-7, "rtol": 0.0}},
    "jax": {**{m: {"atol": 1e-5, "rtol": 1e-8} for m in BITWISE + PADS + EDGE},
            "exact": {"atol": 1e-4, "rtol": 1e-8}},
}
BACKEND = {"A": "numpy", "B": "jax"}
OUT = GS / "tolerances_measured.json"

# --- the user's approval (spec §7.3 "Approved by the user on 2026-10-12") ------------------------

APPROVED_ON = "2026-10-12"
STATUS = f"APPROVED by the user {APPROVED_ON}"
#: D1 scope and value: the NumPy reference gates of these models gain rtol 1e-12 at stress points.
D1_MODELS = PADS + ("exact",)
D1_RTOL = 1e-12
#: The decisions, verbatim labels first. Rendered into the file as ``approval``.
DECISIONS = {
    "D1": {"label": "Add rtol 1e-12 at stress",
           "decision": "sinusoid pad/exact NumPy gates: keep atol 1e-9 ({band,mid,tri}_pad) and atol "
                       "1e-7 (exact) at near points, and add rtol 1e-12 (gate = atol + 1e-12*|lnL|) at "
                       "stress points, like the edge-exact models",
           "applies_to": {"backend": "numpy", "kind": "references", "models": list(D1_MODELS),
                          "rtol": D1_RTOL}},
    "as_proposed": {"label": "approved as proposed",
                    "decision": "all other measured sinusoid gates are approved as proposed (status "
                                "PROPOSED -> APPROVED); each proposal equals its spec §7.3 floor"},
}


def propose(observed: dict, floor: dict) -> dict:
    return {"atol": max(FACTOR * observed["near_max_abs"], floor["atol"]),
            "rtol": max(FACTOR * observed["stress_max_rel"], floor["rtol"])}


def approved_gates() -> dict:
    """The operative gates (``DECISIONS``): the proposal as presented on 2026-10-12, where every gate
    equals its floor (chain rows: the JAX floor's atol, rtol 0), with D1's stress rtol on top. Fixed
    values, not a function of the measurement: a re-capture cannot move an approved gate."""
    out = {}
    for backend in ("numpy", "jax"):
        refs = {m: dict(FLOORS[backend][m]) for m in MODELS}
        if backend == DECISIONS["D1"]["applies_to"]["backend"]:
            for m in D1_MODELS:
                refs[m]["rtol"] = D1_RTOL
        out[backend] = {"references": refs,
                        "chain_rows": {m: {"atol": FLOORS["jax"][m]["atol"], "rtol": 0.0} for m in MODELS},
                        "inf_mask": "identical"}
    return out


def violations(points: dict, gates: dict) -> list[str]:
    """Every gate failure of ``points`` (``{tag: {"model", "value", "reference", "abs_delta"[, "near"]}}``):
    a -inf mask mismatch (``isneginf(value) != isneginf(reference)``) at any gate; NaN or +inf in a
    value or reference; |delta| above its gate (near: atol; stress: atol + rtol |ref|), NaN
    included; and value/reference/abs_delta/near lists of different lengths. Points where both
    are -inf agree."""
    out = []
    for tag, p in points.items():
        gate = gates[p["model"]]
        n = len(p["reference"])
        near = p.get("near", [True] * n)
        if not len(p["value"]) == len(p["abs_delta"]) == len(near) == n:
            out.append(f"{tag}: value/reference/abs_delta/near lengths differ")
            continue
        for i, (val, ref, delta, is_near) in enumerate(zip(p["value"], p["reference"], p["abs_delta"], near)):
            v, r, d = to_float(val), to_float(ref), to_float(delta)
            if math.isnan(v) or math.isnan(r) or math.inf in (v, r):
                out.append(f"{tag}[{i}]: NaN or +inf (value {v}, reference {r})")
            elif (v == -math.inf) != (r == -math.inf):
                out.append(f"{tag}[{i}]: -inf mask mismatch (value {v}, reference {r})")
            elif r != -math.inf:
                allowed = gate["atol"] if is_near else gate["atol"] + gate["rtol"] * abs(r)
                if not d <= allowed:
                    out.append(f"{tag}[{i}]: |delta| {d:.3e} > allowed {allowed:.3e}")
    return out


def _reports(lineage: str, kind: str) -> list[dict]:
    reps = [read_json(p) for p in sorted(EVAL.glob(f"{lineage}_{kind}_*.json"))]
    if not reps:
        raise FileNotFoundError(f"no {kind} report for lineage {lineage} under {EVAL}")
    return reps


def build_sinusoid() -> dict:
    """The measured, proposed and approved gates (nothing written); raises when a gate fails its data."""
    approved = approved_gates()
    out = {"schema": 1, "family": "sinusoid", "status": STATUS, "factor": FACTOR, "floors": FLOORS,
           "file": OUT.relative_to(GS.parent).as_posix(), "reports": [], "observed": {}, "proposed": {},
           "approval": {"date": APPROVED_ON, "decisions": DECISIONS}, "approved": approved}
    for lineage, backend in BACKEND.items():
        refs, rows = _reports(lineage, "references"), _reports(lineage, "chain_rows")
        obs = {m: {"near_max_abs": 0.0, "stress_max_abs": 0.0, "stress_max_rel": 0.0,
                   "inf_mismatch": 0, "n_points": 0} for m in MODELS}
        for rep in refs:
            out["reports"].append(f"{lineage}_references_{rep['tier']}")
            for model, s in rep["models"].items():
                o = obs[model]
                for key in ("near_max_abs", "stress_max_abs", "stress_max_rel"):
                    o[key] = max(o[key], to_float(s[key]))      # "inf" after a -inf mismatch
                o["inf_mismatch"] += s["inf_mismatch"]
                o["n_points"] += s["n_points"]
        cobs = {m: {"max_abs": 0.0, "max_rel": 0.0, "inf_mismatch": 0, "n_rows": 0} for m in MODELS}
        for rep in rows:
            out["reports"].append(f"{lineage}_chain_rows_{rep['tier']}")
            for model, s in rep["models"].items():
                c = cobs[model]
                c["max_abs"] = max(c["max_abs"], to_float(s["max_abs"]))
                c["max_rel"] = max(c["max_rel"], to_float(s["max_rel"]))
                c["inf_mismatch"] += s["inf_mismatch"]
                c["n_rows"] += s["n_rows"]
        empty = [m for m in MODELS if obs[m]["n_points"] <= 0 or cobs[m]["n_rows"] <= 0]
        if empty:
            raise ValueError(f"lineage {lineage}: no reference points or no chain rows for {empty}: "
                             "gate S0 cannot be measured")
        bad = [m for m in MODELS if obs[m]["inf_mismatch"] or cobs[m]["inf_mismatch"]]
        if bad:
            raise ValueError(f"lineage {lineage} disagrees with a stored -inf mask for {bad}: gate S0 fails")
        gates = {m: propose(obs[m], FLOORS[backend][m]) for m in MODELS}
        chain_gates = {m: {"atol": max(FACTOR * cobs[m]["max_abs"], FLOORS["jax"][m]["atol"]), "rtol": 0.0}
                       for m in MODELS}
        problems = [v for rep in refs for v in violations(rep["points"], gates)]
        problems += [v for rep in rows for v in violations(rep["points"], chain_gates)]
        if problems:
            raise AssertionError("a proposed gate fails its own data:\n" + "\n".join(problems[:10]))
        problems = [v for rep in refs for v in violations(rep["points"], approved[backend]["references"])]
        problems += [v for rep in rows for v in violations(rep["points"], approved[backend]["chain_rows"])]
        if problems:
            raise AssertionError(f"an approved gate fails its own data ({backend}, lineage {lineage}; S0 "
                                 "fails, report to the user):\n" + "\n".join(problems[:10]))
        out["observed"][backend] = {"lineage": lineage, "references": obs, "chain_rows": cobs}
        out["proposed"][backend] = {"references": gates, "chain_rows": chain_gates, "inf_mask": "identical"}
    return out


def render(result: dict) -> bytes:
    """The exact bytes of ``tolerances_measured.json`` (``goldens_io.write_json``'s format)."""
    return (json.dumps(jsonable(result), indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def manifest_summary(result: dict) -> dict:
    """``MANIFEST["tolerances"]["sinusoid"]``: file, status and the approval (date, decision labels,
    the approved gates). Shared by the command line and ``capture.capture``."""
    out = {"file": result["file"], "status": result["status"]}
    if "approval" in result:
        out.update(approved_on=result["approval"]["date"],
                   decisions={k: d["label"] for k, d in result["approval"]["decisions"].items()},
                   approved=result["approved"])
    return out


def _report(result: dict) -> None:
    print(f"{OUT.relative_to(GS.parent.parent)}: {result['status']}")
    for backend, p in result["approved"].items():
        for model in MODELS:
            g = p["references"][model]
            print(f"  {backend:5s} {model:12s} atol={g['atol']:.1e} rtol={g['rtol']:.1e} "
                  f"chain_atol={p['chain_rows'][model]['atol']:.1e}")


def measure_sinusoid() -> dict:
    """Measure and write ``OUT`` (no MANIFEST change: ``capture.capture`` pins it)."""
    out = build_sinusoid()
    OUT.write_bytes(render(out))
    _report(out)
    return out


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    return cli(argv)


def cli(argv: list[str] | None = None) -> int:
    """``main`` without the safe-path guard (the tests call it in-process)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=("sinusoid",), default="sinusoid")
    ap.add_argument("--replace", action="store_true",
                    help="re-pin the file in golden/MANIFEST.json when its recorded bytes differ")
    a = ap.parse_args(argv)
    # Measured, serialised and hashed in memory; nothing is written until the manifest check passed.
    result = build_sinusoid()
    data = render(result)
    manifest = GS.parent / "MANIFEST.json"
    files = {result["file"]: hashlib.sha256(data).hexdigest()}
    bad = manifest_merge.conflicts(manifest_merge.load(manifest), files)
    if bad and not a.replace:
        raise SystemExit(f"golden/MANIFEST.json holds other bytes for {bad}; nothing written; "
                         "rerun with --replace only if the change is intended")
    OUT.write_bytes(data)
    if manifest_merge.sha256_file(OUT) != files[result["file"]]:
        raise RuntimeError(f"{OUT}: written bytes differ from the measurement")
    manifest_merge.merge_entries(manifest, files, tolerances={"sinusoid": manifest_summary(result)},
                                 replace=a.replace)
    _report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
