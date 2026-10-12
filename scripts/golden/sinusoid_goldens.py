"""Sinusoid goldens and shipped data (spec §1 'Shipped data', §7.2 G1/G2/G5/G6; Tasks A4-A10).

From the release root, after ``source env/activate.sh``:

    $PY -m scripts.golden.sinusoid_goldens ship-data              # A4: the only writer under data/
    $PY -m scripts.golden.sinusoid_goldens g1                     # A5
    $PY -m scripts.golden.sinusoid_goldens eval --lineage A       # A6 (lineage B: A7)
    $PY -m scripts.golden.sinusoid_goldens g2                     # A8
    $PY -m scripts.golden.sinusoid_goldens eval-chains --lineage A
    $PY -m scripts.golden.sinusoid_goldens g6                     # A9

Apart from ship-data, every command writes only below golden/sinusoid/. Legacy code runs
only inside lineage subprocesses (scripts.golden.legacy_eval); this process reads the
frozen trees as bytes.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from scripts.golden.goldens_io import (read_json, read_npz, sha256_bytes, sha256_file,
                                       update_data_manifest, write_json, write_npz)
from scripts.golden.paths import DATA, GOLDEN, ROOT

CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
#: The 15 WDM II models (spec §0 rule 7; §5.3 order).
MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact",
          "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad",
          "tri_exact", "exact")
N_REF = 24                                  # toy_models/sinusoid/gpu/make_matrix_bundle.py::N_REF
BUNDLE_REL = "toy_models/sinusoid/gpu/bundles/sinusoid_matrix_colab.zip"
BUNDLE = ROOT / BUNDLE_REL
BUNDLE_ROOT = "sinusoid_matrix_colab"
SINUSOID_FINAL = ROOT / "wdm_paper_2" / "sinusoid_final"
GS = GOLDEN / "sinusoid"
REFS = GS / "references"
EVAL = GS / "legacy_eval"
CHAIN_ROWS = GS / "chain_rows"
INTERMEDIATES = GS / "intermediates"

_CHIRP = "wdm_paper_2/final_chains/frequency_chirp_study"      # make_bundle.DATA_ROOT
_CACHE = "wdm_paper_2/final_chains/frequency_cut_4hz/cache"     # make_bundle.CACHE_ROOT
#: data/<rel> -> (canonical source under ROOT, member under the bundle root, sha256)
DATA_FILES: dict[str, tuple[str, str, str]] = {
    "sinusoid/manuscript_low/toy_data.npz": (
        f"{_CHIRP}/manuscript_low/toy_data.npz", "data/manuscript_low/toy_data.npz",
        "650cf032a6e30fc1ad3bd40248ea943e21a70f8243a27ff1ef40703f86d530e8"),
    "sinusoid/manuscript_mid/toy_data.npz": (
        f"{_CHIRP}/manuscript_mid/toy_data.npz", "data/manuscript_mid/toy_data.npz",
        "02cdb2595ce1058a063b131abbd1588070892bb0f18260f4cf545936785d222d"),
    "sinusoid/manuscript_high/toy_data.npz": (
        f"{_CHIRP}/manuscript_high/toy_data.npz", "data/manuscript_high/toy_data.npz",
        "a6db39e500be548d2d6453d04f1f01d3b891d815ba8c90ce2de2958d972bc05a"),
    "sinusoid/cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy": (
        f"{_CACHE}/cN_N16384_dt0.0009765625_fmin4_fref1.npy",
        "cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy",
        "d263bf25960ab6f4e371fd597da0e28123cba569ca085e2af5e6c3599257213f"),
    "sinusoid/cache/bandcov_v1_4bfb466c191c79f9879a.npz": (
        f"{_CACHE}/bandcov_v1_4bfb466c191c79f9879a.npz",
        "cache/bandcov_v1_4bfb466c191c79f9879a.npz",
        "752b3ea46cf14feaf5b8845ad39999a0b596a956d6af9e58072f691a8cd6c2c2"),
}

COMMANDS: dict[str, Callable[[list[str]], object]] = {}


def command(name: str):
    def register(fn):
        COMMANDS[name] = fn
        return fn
    return register


@command("ship-data")
def ship_data(argv: list[str]) -> dict:
    """Copy the canonical sinusoid inputs into data/sinusoid; record them in data/MANIFEST.json."""
    argparse.ArgumentParser(prog="sinusoid_goldens ship-data").parse_args(argv)
    files, provenance = {}, {}
    with zipfile.ZipFile(BUNDLE) as bundle:
        for rel, (source, member, sha) in DATA_FILES.items():
            body = (ROOT / source).read_bytes()
            if sha256_bytes(body) != sha:
                raise ValueError(f"{source}: sha256 {sha256_bytes(body)} differs from the pinned {sha}")
            if sha256_bytes(bundle.read(f"{BUNDLE_ROOT}/{member}")) != sha:
                raise ValueError(f"bundle member {member} differs from {source}")
            target = DATA / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            files[rel] = sha
            provenance[rel] = {"sha256": sha, "bytes": len(body), "from": source,
                               "bundle": BUNDLE_REL, "bundle_member": f"{BUNDLE_ROOT}/{member}"}
    update_data_manifest(files, provenance)
    print(f"shipped {len(files)} sinusoid files into {DATA / 'sinusoid'}")
    return provenance


# --- G1: the 45 CPU reference files (Task A5) ---------------------------------------------

def fast_indices(loglike) -> list[int]:
    """Fast tier of one reference file: the truth (0), the first tight point (1), the first
    wide point (2), then the first -inf point (edge-exact: 4) or else the first stress
    corner (1 + N_REF = 25)."""
    loglike = np.asarray(loglike, dtype=float)
    inf = np.flatnonzero(loglike == -np.inf)
    return [0, 1, 2, int(inf[0]) if inf.size else 1 + N_REF]


@command("g1")
def g1(argv: list[str]) -> dict:
    """Extract prepared/<case>_<model>_reference.npz of the 15 paper models from the Colab
    bundle; each sha256 must equal asset_sha256.reference of its sinusoid_final run."""
    argparse.ArgumentParser(prog="sinusoid_goldens g1").parse_args(argv)
    manifest = read_json(SINUSOID_FINAL / "manifest.json")
    REFS.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(BUNDLE) as bundle:
        for case in CASES:
            for model in MODELS:
                name = f"{case}_{model}_reference.npz"
                body = bundle.read(f"{BUNDLE_ROOT}/prepared/{name}")
                sha = sha256_bytes(body)
                run = manifest["runs"][case][model]
                target = read_json(SINUSOID_FINAL / run / "run_meta.json")["target"]
                if (target["case"], target["model"]) != (case, model):
                    raise ValueError(f"{run}: run_meta target is {target['case']}/{target['model']}")
                if target["asset_sha256"]["reference"] != sha:
                    raise ValueError(f"{name}: sha256 {sha} != run_meta {target['asset_sha256']['reference']}")
                (REFS / name).write_bytes(body)
                with np.load(io.BytesIO(body), allow_pickle=False) as z:
                    theta, loglike, near = z["theta"], z["loglike"], z["near"]
                files[name] = {"case": case, "model": model, "run": run, "sha256": sha, "bytes": len(body),
                               "n_points": int(len(loglike)), "ndim": int(theta.shape[1]),
                               "near_index": np.flatnonzero(near).tolist(),
                               "neg_inf_index": np.flatnonzero(loglike == -np.inf).tolist(),
                               "truth_lnlike": float(loglike[0]), "fast_index": fast_indices(loglike)}
    write_json(REFS / "INDEX.json", {
        "schema": 1, "bundle": {"path": BUNDLE_REL, "sha256": sha256_file(BUNDLE)},
        "sinusoid_final_manifest_sha256": sha256_file(SINUSOID_FINAL / "manifest.json"),
        "files": files})
    print(f"G1: {len(files)} reference files verified against sinusoid_final run_meta -> {REFS}")
    return files


# --- re-evaluation of the references by a lineage (Tasks A6, A7) ---------------------------

def compare(values, reference, near) -> dict:
    """|value - reference| split like gpu/run.py::validate_reference: near points (atol) and
    stress points (atol + rtol |ref|); -inf must agree exactly (counted, not differenced)."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    near = np.asarray(near, dtype=bool)
    if not np.all(np.isfinite(values) | (values == -np.inf)):
        raise FloatingPointError("a lineage returned NaN or +inf")
    v_inf, r_inf = values == -np.inf, reference == -np.inf
    both = ~v_inf & ~r_inf
    delta = np.zeros(values.shape)
    delta[both] = np.abs(values[both] - reference[both])
    rel = np.zeros(values.shape)
    rel[both] = delta[both] / np.maximum(np.abs(reference[both]), np.finfo(float).tiny)
    return {"abs_delta": delta, "rel_delta": rel, "inf_mismatch": int(np.sum(v_inf != r_inf)),
            "near_max_abs": float(delta[near].max(initial=0.0)),
            "stress_max_abs": float(delta[~near].max(initial=0.0)),
            "stress_max_rel": float(rel[~near].max(initial=0.0))}


def _merge(models: dict, model: str, cmp: dict, n: int) -> None:
    m = models.setdefault(model, {"near_max_abs": 0.0, "stress_max_abs": 0.0, "stress_max_rel": 0.0,
                                  "inf_mismatch": 0, "n_points": 0})
    for key in ("near_max_abs", "stress_max_abs", "stress_max_rel"):
        m[key] = max(m[key], cmp[key])
    m["inf_mismatch"] += cmp["inf_mismatch"]
    m["n_points"] += n


def _print_models(title: str, models: dict) -> None:
    print(title)
    for model, m in models.items():
        print(f"  {model:12s} " + " ".join(
            f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


def _portable_worker(meta: dict) -> dict:
    """Worker meta for a committed report: ``wdm_file`` relative to ROOT (WDM_LEGACY_ROOT), so the
    golden does not depend on where this machine keeps the frozen trees. ``None`` (a lineage
    that must not import wdm) passes through; a file outside ROOT raises ValueError."""
    worker = dict(meta)
    if worker.get("wdm_file") is not None:
        worker["wdm_file"] = Path(worker["wdm_file"]).relative_to(ROOT).as_posix()
    return worker


def run_reference_eval(lineage: str, tier: str = "fast") -> dict:
    from scripts.golden.legacy_eval import Job, evaluate_jobs
    files = read_json(REFS / "INDEX.json")["files"]
    jobs, expected = [], {}
    for case in CASES:
        for model in MODELS:
            name = f"{case}_{model}_reference.npz"
            ref = read_npz(REFS / name)
            idx = np.asarray(files[name]["fast_index"] if tier == "fast" else np.arange(len(ref["loglike"])),
                             dtype=int)
            tag = f"{case}__{model}"
            jobs.append(Job(tag, case, model, ref["theta"][idx]))
            expected[tag] = (case, model, idx, ref["loglike"][idx], ref["near"][idx])
    results, meta = evaluate_jobs(lineage, "sinusoid", jobs, timeout=7200.0)
    report = {"schema": 1, "family": "sinusoid", "kind": "references", "lineage": lineage, "tier": tier,
              "worker": _portable_worker(meta), "models": {}, "points": {}}
    for tag, (case, model, idx, ref_ll, near) in expected.items():
        r = results[tag]
        cmp = compare(r["lnlike"], ref_ll, near)
        point = {"case": case, "model": model, "index": idx, "near": near, "reference": ref_ll,
                 "value": r["lnlike"], "abs_delta": cmp["abs_delta"], "seconds": r["seconds"],
                 "inf_mismatch": cmp["inf_mismatch"], "info": r["info"]}
        if "failed" in r:
            if not np.array_equal(r["failed"], r["lnlike"] == -np.inf):
                raise AssertionError(f"{tag}: last_failed disagrees with the -inf values")
            point["failed"] = r["failed"]
        report["points"][tag] = point
        _merge(report["models"], model, cmp, len(idx))
    EVAL.mkdir(parents=True, exist_ok=True)
    write_json(EVAL / f"{lineage}_references_{tier}.json", report)
    _print_models(f"lineage {lineage}, references, {tier} tier:", report["models"])
    return report


@command("eval")
def eval_cmd(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(prog="sinusoid_goldens eval")
    ap.add_argument("--lineage", choices=("A", "B"), required=True)
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    a = ap.parse_args(argv)
    return run_reference_eval(a.lineage, a.tier)


# --- G2: sinusoid chain rows (Task A8) -----------------------------------------------------
G2_SEED = 20261011
G2_RANDOM = 18
#: Positions within the 20 stored rows re-evaluated in the fast tier: first row, first random
#: row, last row.
G2_FAST_POSITIONS = (0, 1, 19)
#: impulse v2 964283c: impulse/file_io.py ncols = ndim + 4; save_chain column_stack order.
CHAIN_EXTRA_COLUMNS = ("lnlike", "lnprob", "accepted", "temperature")


def chain_row_index(nrows: int) -> np.ndarray:
    """First and last row plus G2_RANDOM distinct interior rows from default_rng(G2_SEED), sorted."""
    rng = np.random.default_rng(G2_SEED)
    middle = rng.choice(np.arange(1, nrows - 1), size=G2_RANDOM, replace=False)
    return np.sort(np.r_[0, middle, nrows - 1]).astype(np.int64)


@command("g2")
def g2(argv: list[str]) -> dict:
    """20 rows of each sinusoid_final run's cold chain (chain_0.bin): theta and lnlike."""
    argparse.ArgumentParser(prog="sinusoid_goldens g2").parse_args(argv)
    manifest = read_json(SINUSOID_FINAL / "manifest.json")
    CHAIN_ROWS.mkdir(parents=True, exist_ok=True)
    runs = {}
    for case in CASES:
        for model in MODELS:
            run = manifest["runs"][case][model]
            meta_path = SINUSOID_FINAL / run / "run_meta.json"
            meta = read_json(meta_path)
            names = list(meta["parameter_names"])
            ndim = len(names)
            if meta.get("ndim") != ndim or (meta["target"]["case"], meta["target"]["model"]) != (case, model):
                raise ValueError(f"{run}: run_meta disagrees with ({case}, {model}, ndim {ndim})")
            path = SINUSOID_FINAL / run / "chain_0.bin"
            ncols = ndim + len(CHAIN_EXTRA_COLUMNS)
            size = path.stat().st_size
            if size % (8 * ncols):
                raise ValueError(f"{path}: {size} bytes is not a whole number of {ncols}-column rows")
            nrows = size // (8 * ncols)
            idx = chain_row_index(nrows)
            rows = np.array(np.memmap(path, dtype="<f8", mode="r", shape=(nrows, ncols))[idx])
            lnlike, lnprob, accepted, temperature = (rows[:, ndim + k].copy() for k in range(4))
            if not np.all(temperature == 1.0):
                raise ValueError(f"{path}: chain_0 is not the cold chain")
            if not np.all(np.isfinite(lnlike)):
                raise ValueError(f"{path}: non-finite lnlike in a stored row")
            write_npz(CHAIN_ROWS / f"{case}_{model}.npz",
                      {"row_index": idx, "theta": rows[:, :ndim].copy(), "lnlike": lnlike, "lnprob": lnprob,
                       "accepted": accepted, "temperature": temperature})
            runs[f"{case}_{model}"] = {"case": case, "model": model, "run": run, "nrows": int(nrows),
                                       "ndim": ndim, "parameter_names": names,
                                       "chain_0_sha256": sha256_file(path),
                                       "run_meta_sha256": sha256_file(meta_path),
                                       "device": meta.get("device"), "thin": meta.get("thin")}
    write_json(CHAIN_ROWS / "INDEX.json", {
        "schema": 1, "seed": G2_SEED, "n_random": G2_RANDOM, "fast_positions": list(G2_FAST_POSITIONS),
        "layout": "impulse v2 964283c: float64 rows of ndim + 4 columns = parameters, lnlike, lnprob, "
                  "accepted, temperature (impulse/file_io.py ncols, save_chain)",
        "runs": runs})
    print(f"G2: 20 rows from each of {len(runs)} sinusoid_final cold chains -> {CHAIN_ROWS}")
    return runs


def run_chain_eval(lineage: str, tier: str = "fast") -> dict:
    """Re-evaluate stored chain rows (phase wrapped as gpu/run.py did) against their lnlike column."""
    from scripts.golden.legacy_eval import Job, evaluate_jobs
    jobs, expected = [], {}
    for case in CASES:
        for model in MODELS:
            rows = read_npz(CHAIN_ROWS / f"{case}_{model}.npz")
            pos = np.asarray(G2_FAST_POSITIONS if tier == "fast" else np.arange(len(rows["lnlike"])), dtype=int)
            tag = f"{case}__{model}"
            jobs.append(Job(tag, case, model, rows["theta"][pos], wrap_phase=True))
            expected[tag] = (case, model, rows["row_index"][pos], rows["lnlike"][pos])
    results, meta = evaluate_jobs(lineage, "sinusoid", jobs, timeout=7200.0)
    report = {"schema": 1, "family": "sinusoid", "kind": "chain_rows", "lineage": lineage, "tier": tier,
              "worker": _portable_worker(meta), "models": {}, "points": {}}
    for tag, (case, model, row_index, chain_ll) in expected.items():
        r = results[tag]
        cmp = compare(r["lnlike"], chain_ll, np.ones(len(chain_ll), dtype=bool))
        point = {"case": case, "model": model, "row_index": row_index, "reference": chain_ll,
                 "value": r["lnlike"], "abs_delta": cmp["abs_delta"], "rel_delta": cmp["rel_delta"],
                 "seconds": r["seconds"], "inf_mismatch": cmp["inf_mismatch"], "info": r["info"]}
        if "failed" in r:
            if not np.array_equal(r["failed"], r["lnlike"] == -np.inf):
                raise AssertionError(f"{tag}: last_failed disagrees with the -inf values")
            point["failed"] = r["failed"]
        report["points"][tag] = point
        m = report["models"].setdefault(model, {"max_abs": 0.0, "max_rel": 0.0, "inf_mismatch": 0, "n_rows": 0})
        m["max_abs"] = max(m["max_abs"], cmp["near_max_abs"])
        m["max_rel"] = max(m["max_rel"], float(cmp["rel_delta"].max(initial=0.0)))
        m["inf_mismatch"] += cmp["inf_mismatch"]
        m["n_rows"] += len(chain_ll)
    EVAL.mkdir(parents=True, exist_ok=True)
    write_json(EVAL / f"{lineage}_chain_rows_{tier}.json", report)
    _print_models(f"lineage {lineage}, chain rows, {tier} tier:", report["models"])
    return report


@command("eval-chains")
def eval_chains_cmd(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(prog="sinusoid_goldens eval-chains")
    ap.add_argument("--lineage", choices=("A", "B"), required=True)
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    a = ap.parse_args(argv)
    return run_chain_eval(a.lineage, a.tier)


# --- G6: per-module intermediates at the manuscript_mid truth (Task A9) --------------------

@command("g6")
def g6(argv: list[str]) -> dict:
    argparse.ArgumentParser(prog="sinusoid_goldens g6").parse_args(argv)
    from scripts.golden.lineage import run_worker
    out = run_worker("A", "sinusoid_wt", "intermediates", "--data-root", str(DATA / "sinusoid"),
                     "--cache-src", str(DATA / "sinusoid" / "cache"), timeout=3600.0)
    # wdm_file relative to ROOT, as in every committed lineage report (_portable_worker)
    report = _portable_worker(read_json(out / "intermediates.json"))
    arrays = read_npz(out / "intermediates.npz")
    INTERMEDIATES.mkdir(parents=True, exist_ok=True)
    write_json(INTERMEDIATES / "manuscript_mid_truth.json", report)
    write_npz(INTERMEDIATES / "manuscript_mid_truth.npz", arrays)
    print(f"G6: {len(report['parts'])} models split into their terms; arrays {sorted(arrays)} -> {INTERMEDIATES}")
    return report


# --- the whole sinusoid capture (Task A10) ----------------------------------------------------

def capture_all(tier: str = "fast") -> dict:
    """Every sinusoid golden for one tier; returns the MANIFEST 'sources' block."""
    for rel, (_, _, sha) in DATA_FILES.items():
        if sha256_file(DATA / rel) != sha:
            raise ValueError(f"shipped data/{rel} changed; rerun ship-data")
    g1([])
    for lineage in ("A", "B"):
        run_reference_eval(lineage, tier)
    g2([])
    for lineage in ("A", "B"):
        run_chain_eval(lineage, tier)
    g6([])
    return sources()


def sources() -> dict:
    from scripts.golden.lineage import LINEAGES
    runs = read_json(CHAIN_ROWS / "INDEX.json")["runs"]
    return {"bundle": {"path": BUNDLE_REL, "sha256": sha256_file(BUNDLE)},
            "sinusoid_final_manifest": {"path": "wdm_paper_2/sinusoid_final/manifest.json",
                                        "sha256": sha256_file(SINUSOID_FINAL / "manifest.json")},
            "data": {rel: sha for rel, (_, _, sha) in DATA_FILES.items()},
            "chains": {key: {k: v[k] for k in ("run", "nrows", "chain_0_sha256", "run_meta_sha256")}
                       for key, v in runs.items()},
            "lineages": {k: LINEAGES[k] for k in ("A", "B")}}


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python -m scripts.golden.sinusoid_goldens {" + ",".join(sorted(COMMANDS)) + "} [options]",
              file=sys.stderr)
        return 2
    COMMANDS[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
