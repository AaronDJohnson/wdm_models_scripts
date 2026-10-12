"""P0 measurement of the sinusoid gates (spec §7.3; Task A10).

    $PY -m scripts.golden.measure_tolerances --family sinusoid

Reads golden/sinusoid/legacy_eval/{A,B}_{references,chain_rows}_<tier>.json. Gates are split
like toy_models/sinusoid/gpu/run.py::validate_reference: near points |delta| <= atol;
stress points |delta| <= atol + rtol |lnL_ref|; -inf masks identical. Proposed gate =
max(FACTOR x observed, floor) with the spec §7.3 floors. Writes
golden/sinusoid/tolerances_measured.json with status PROPOSED: the user approves the gates
(spec §9 "Still open").
"""
from __future__ import annotations

import argparse
import math
import sys

from scripts.golden.goldens_io import read_json, to_float, write_json
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
STATUS = "PROPOSED - pending user approval (spec §7.3, §9 'Still open')"
OUT = GS / "tolerances_measured.json"


def propose(observed: dict, floor: dict) -> dict:
    return {"atol": max(FACTOR * observed["near_max_abs"], floor["atol"]),
            "rtol": max(FACTOR * observed["stress_max_rel"], floor["rtol"])}


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


def measure_sinusoid() -> dict:
    out = {"schema": 1, "family": "sinusoid", "status": STATUS, "factor": FACTOR, "floors": FLOORS,
           "file": OUT.relative_to(GS.parent).as_posix(), "reports": [], "observed": {}, "proposed": {}}
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
        out["observed"][backend] = {"lineage": lineage, "references": obs, "chain_rows": cobs}
        out["proposed"][backend] = {"references": gates, "chain_rows": chain_gates, "inf_mask": "identical"}
    write_json(OUT, out)
    print(f"{OUT.relative_to(GS.parent.parent)}: {STATUS}")
    for backend, p in out["proposed"].items():
        for model in MODELS:
            g = p["references"][model]
            print(f"  {backend:5s} {model:12s} atol={g['atol']:.1e} rtol={g['rtol']:.1e} "
                  f"chain_atol={p['chain_rows'][model]['atol']:.1e}")
    return out


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=("sinusoid",), default="sinusoid")
    ap.parse_args(argv)
    measure_sinusoid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
