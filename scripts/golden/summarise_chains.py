"""G8: posterior summaries of the final chains, computed read-only (spec 7.4).

    $PY -P scripts/golden/summarise_chains.py --toy sinusoid [--targets CASE/MODEL,...] [--no-hash]
    $PY -P scripts/golden/summarise_chains.py --toy galactic_binary [--targets ...] [--no-hash]

Reads ``wdm_paper_2/sinusoid_final/manifest.json`` (45 runs) or
``wdm_paper_2/final_chains/gb_likelihood_matrix/manifest.json`` (38 targets, 58 run
directories; pooled targets list their ``runs``). Only ``run_meta.json``, ``chain_0.bin`` (the
cold chain), ``diagnostics.json`` and ``noise_states.jsonl`` are opened, all read-only; nothing
is ever written under the chains root. Writes ``golden/chain_summaries/<toy>/<case>_<model>.json``
and merges them into ``golden/MANIFEST.json``.

Statistics (cold chain, first 30% of each chain dropped; periodic parameters unwrapped about
their circular mean; numbers identical to ``chain_stats.diagnostics`` for one chain):

* mean, std (ddof 1), 5/16/50/84/95% quantiles of the pooled kept samples, circular mean;
* ESS = min(batch means with floor(sqrt n) batches, Geyer initial sequence), per chain; a
  pooled target's ESS is the sum over its chains (the GB manifest's ``ess_rule``);
* MCSE of the mean = std / sqrt(ESS);
* split-R-hat: 4 segments of one chain; for a pooled target, two segments per chain (the
  first ``h`` samples and ``h`` samples from its midpoint, ``h`` = half the shortest chain), every
  chain unwrapped about the pooled circular mean (``collect.pooled_diagnostics``);
* counter totals from ``noise_states.jsonl`` and the -inf fractions per noise state and per row;
* GB only: ``P(primary)`` / ``P(reflection)`` = share of kept samples whose ``cos_colat`` has the
  same / opposite sign as the injection's (``primary_rule``).

The Geyer estimator is a verbatim copy of ``impulse.diagnostics`` at impulse 964283c (v2), so
this script needs NumPy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

if not sys.flags.safe_path:
    raise SystemExit("run with python -P (PYTHONSAFEPATH)")
HERE = Path(__file__).resolve().parent
REL = HERE.parents[1]
for _p in (str(HERE), str(REL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import manifest_merge  # noqa: E402
from scripts.golden.paths import GOLDEN, ROOT  # noqa: E402  (Section A)

ROOTS = {"sinusoid": "wdm_paper_2/sinusoid_final",
         "galactic_binary": "wdm_paper_2/final_chains/gb_likelihood_matrix"}
BURN_FRACTION = 0.3
SEGMENTS = 4
MIN_BATCH_SAMPLES = 4
QUANTILES = (0.05, 0.16, 0.5, 0.84, 0.95)
COUNTERS = ("noise_states", "pd_failed_states", "pd_failed_rows", "evaluations")
COUNTER_LOG = "noise_states.jsonl"
_LEGACY_KEYS = {"noise_evaluations": "noise_states", "pd_failures": "pd_failed_states"}
PRIMARY_RULE = "sign(cos_colat) == sign(cos_colat of the injection, run_meta anchor[1])"
CAVEATS = {
    ("galactic_binary", "gb_12mo", "band_raw"): "parameter rows bitwise identical to gb_12mo mid_raw "
    "(same seed 3007 and pooled flow): not independent evidence",
    ("galactic_binary", "gb_12mo", "mid_raw"): "parameter rows bitwise identical to gb_12mo band_raw "
    "(same seed 3007 and pooled flow): not independent evidence",
    ("sinusoid", "manuscript_high", "tri_exact"): "accepted at min ESS 2639 < 3000 "
    "(user ruling 2026-10-07)",
}


# --------------------------------------------------------------------------------------------
# impulse 964283c, impulse/diagnostics.py: Geyer initial-sequence ESS (verbatim)
# --------------------------------------------------------------------------------------------


def _next_fast_len(n: int) -> int:
    m = 1
    while m < 2 * n:
        m <<= 1
    return m


def _acf_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    x = x - x.mean()
    var = np.dot(x, x) / T
    if var == 0.0 or not np.isfinite(var):
        return np.full(T, np.nan)
    nfft = _next_fast_len(T)
    fx = np.fft.rfft(x, n=nfft)
    S = fx * np.conjugate(fx)
    acov = np.fft.irfft(S, n=nfft)[:T]
    acov = acov / T
    rho = acov / acov[0]
    return np.real(rho)


def _pair_sums_gamma(rho: np.ndarray) -> np.ndarray:
    tail = rho[1:]
    L = (tail.shape[0] // 2) * 2
    if L == 0:
        return np.array([], dtype=float)
    pairs = tail[:L].reshape(-1, 2)
    return pairs.sum(axis=1)


def _ips_tau_from_gamma(gamma: np.ndarray) -> float:
    if gamma.size == 0:
        return 1.0
    pos_mask = gamma > 0.0
    if not np.any(pos_mask):
        return 1.0
    Mstar = np.argmax(~pos_mask)
    if not pos_mask.all():
        gamma_use = gamma[:Mstar] if Mstar > 0 else np.array([], dtype=float)
    else:
        gamma_use = gamma
    return 1.0 + 2.0 * np.sum(gamma_use)


def _pava_monotone_nonincreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = y.astype(float)
    n = y.size
    if n == 0:
        return y
    if w is None:
        w = np.ones(n, dtype=float)
    yy = -y
    v = yy.copy()
    ww = w.copy()
    k = 0
    avg = []
    wsum = []
    for i in range(n):
        avg.append(v[i])
        wsum.append(ww[i])
        k += 1
        while k >= 2 and avg[k - 2] > avg[k - 1]:
            new_w = wsum[k - 2] + wsum[k - 1]
            new_avg = (wsum[k - 2] * avg[k - 2] + wsum[k - 1] * avg[k - 1]) / new_w
            avg[k - 2] = new_avg
            wsum[k - 2] = new_w
            avg.pop()
            wsum.pop()
            k -= 1
    out = np.empty(n, dtype=float)
    idx = 0
    for a, w_ in zip(avg, wsum):
        m = int(round(w_)) if np.allclose(w_, round(w_)) else int(w_)
        out[idx: idx + m] = a
        idx += m
    return -out


def _ims_tau_from_gamma(gamma: np.ndarray) -> float:
    if gamma.size == 0:
        return 1.0
    gamma_mon = _pava_monotone_nonincreasing(gamma)
    pos = np.where(gamma_mon > 0.0)[0]
    if pos.size == 0:
        return 1.0
    Mprime = pos[-1] + 1
    gamma_use = gamma_mon[:Mprime]
    return 1.0 + 2.0 * np.sum(gamma_use)


def autocorr_length_ips_ims(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    T, D = x.shape
    tau_ips = np.empty(D, dtype=float)
    tau_ims = np.empty(D, dtype=float)
    for j in range(D):
        rho = _acf_fft(x[:, j])
        if not np.isfinite(rho).all():
            tau_ips[j] = np.nan
            tau_ims[j] = np.nan
            continue
        gamma = _pair_sums_gamma(rho)
        tau_ips[j] = _ips_tau_from_gamma(gamma)
        tau_ims[j] = _ims_tau_from_gamma(gamma)
    return tau_ips, tau_ims


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    T, D = x.shape
    ess = np.empty(D, dtype=float)
    tau_ips, tau_ims = autocorr_length_ips_ims(x)
    for j in range(D):
        tau = min(tau_ips[j], tau_ims[j])
        if not np.isfinite(tau) or tau <= 0.0:
            ess[j] = np.nan
        else:
            ess[j] = T / tau
    return ess


# --------------------------------------------------------------------------------------------
# toy_models/galactic_binary/chain_stats.py (== the sinusoid copy), read-only parts
# --------------------------------------------------------------------------------------------


def batch_count(n: int) -> int:
    return math.isqrt(max(int(n), 0))


def ess_batch_means(x) -> float:
    y = np.asarray(x, float)
    b = batch_count(y.size)
    if y.size < MIN_BATCH_SAMPLES or b < 2:
        return float("nan")
    m = y.size // b
    y = y[: b * m]
    v = y.var(ddof=1)
    if v == 0.0:
        return 0.0
    vb = y.reshape(b, m).mean(axis=1).var(ddof=1)
    return float("inf") if vb == 0.0 else float(y.size * v / (m * vb))


def ess_geyer(x) -> float:
    y = np.asarray(x, float)
    if y.size < 2:
        return float("nan")
    return float(effective_sample_size(y[:, None])[0])


def split_rhat(x, n_splits: int = SEGMENTS) -> float:
    y = np.asarray(x, float)
    per = y.size // n_splits
    if per < 2:
        return float("nan")
    return rhat_of_segments(y[: per * n_splits].reshape(n_splits, per))


def rhat_of_segments(c: np.ndarray) -> float:
    """Split-R-hat of equal-length segments ``c`` (n_segments, per), chain_stats' formula."""
    per = c.shape[1]
    w = c.var(axis=1, ddof=1).mean()
    b = per * c.mean(axis=1).var(ddof=1)
    if w == 0.0:
        return float("inf") if b > 0.0 else float("nan")
    return float(np.sqrt(((per - 1) / per * w + b / per) / w))


def nan_min(values) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return min(finite) if finite else float("nan")


def nan_max(values) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return max(finite) if finite else float("nan")


def parameter_stats(x) -> dict:
    y = np.asarray(x, float)
    if y.size >= 2 and np.all(y == y[0]):
        return dict(ess=0.0, ess_batch=0.0, ess_geyer=0.0, rhat=float("inf"), frozen=True)
    if y.size == 0:
        return dict(ess=0.0, ess_batch=0.0, ess_geyer=0.0, rhat=float("nan"), frozen=False)
    batch, geyer = ess_batch_means(y), ess_geyer(y)
    return dict(ess=nan_min([batch, geyer]), ess_batch=batch, ess_geyer=geyer,
                rhat=split_rhat(y), frozen=False)


def circular_mean(x, low: float, high: float) -> float:
    period = high - low
    angle = 2 * np.pi * (np.asarray(x, float) - low) / period
    return float(low + period * np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi))


def unwrap(x, low: float, high: float, centre: float | None = None) -> np.ndarray:
    period = high - low
    if centre is None:
        centre = circular_mean(x, low, high)
    return centre + (x - centre + period / 2) % period - period / 2


def counter_totals(run_dir: Path) -> dict | None:
    path = run_dir / COUNTER_LOG
    if not path.is_file():
        return None
    lines = []
    for text in path.read_text().splitlines():
        try:
            lines.append(json.loads(text))
        except ValueError:
            continue
    if not lines:
        return None
    totals = {k: 0 for k in COUNTERS}
    seen, sessions, clean = set(), [], set()
    for i, line in enumerate(lines):
        for old, new in _LEGACY_KEYS.items():
            if old in line and new not in line:
                line[new] = line[old]
        for k in COUNTERS:
            totals[k] += int(line.get(k) or 0)
        session = line.get("session", f"legacy-{i}")
        if session not in seen:
            seen.add(session)
            sessions.append(session)
        if line.get("event", "end") == "end":
            clean.add(session)
    totals["counter_sessions"] = len(sessions)
    totals["counters_lower_bound"] = any(s not in clean for s in sessions)
    return totals


# --------------------------------------------------------------------------------------------
# targets and chains
# --------------------------------------------------------------------------------------------


def targets(toy: str, root: Path) -> dict[tuple[str, str], list[str]]:
    """(case, model) -> run directories relative to ``root``."""
    m = json.loads((root / "manifest.json").read_text())
    if toy == "sinusoid":
        return {(case, model): [rel] for case, d in m["runs"].items() for model, rel in d.items()}
    return {(case, model): [f"chains/{case}/{r}" for r in (e["runs"] if "runs" in e else [e["run"]])]
            for case, d in m.items() for model, e in d.items()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_chain(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    names = list(meta["parameter_names"])
    width = len(names) + 4
    data = np.fromfile(run_dir / "chain_0.bin", dtype="<f8")
    if data.size % width:
        raise ValueError(f"{run_dir}/chain_0.bin is not a whole number of {width}-column rows")
    rows = data.reshape(-1, width)
    return meta, names, rows


def periodic_of(toy: str, meta: dict, names: list[str]) -> dict[str, tuple[float, float]]:
    if toy == "galactic_binary":
        return {k: tuple(v) for k, v in (meta.get("periodic") or {}).items()}
    return {names[int(i)]: tuple(v) for i, v in (meta["target"].get("periodic") or {}).items()}


def target_of(toy: str, meta: dict) -> tuple[str, str]:
    t = meta if toy == "galactic_binary" else meta["target"]
    return t["case"], t["model"]


def summarise_target(toy: str, root: Path, case: str, model: str, run_rels: list[str], *,
                     hash_files: bool = True) -> dict:
    chains, per_chain = [], []
    names = periodic = None
    anchor_cc = None
    for rel in run_rels:
        run_dir = root / rel
        meta, nm, rows = read_chain(run_dir)
        if target_of(toy, meta) != (case, model):
            raise RuntimeError(f"{run_dir}: run_meta is {target_of(toy, meta)}, expected {(case, model)}")
        if names is None:
            names, periodic = nm, periodic_of(toy, meta, nm)
            if toy == "galactic_binary":
                anchor_cc = float(meta["anchor"][names.index("cos_colat")])
        elif nm != names:
            raise RuntimeError(f"{run_dir}: parameter names differ within a pooled target")
        n = rows.shape[0]
        kept = rows[int(BURN_FRACTION * n):, :len(names)]
        params = {}
        for j, name in enumerate(names):
            x = kept[:, j]
            if name in periodic and len(x):
                x = unwrap(x, *periodic[name])
            params[name] = parameter_stats(x)
        stored = None
        if (run_dir / "diagnostics.json").is_file():
            d = json.loads((run_dir / "diagnostics.json").read_text())
            stored = {"samples": d.get("samples"), "min_ess": d.get("min_ess"),
                      "max_rhat": d.get("max_rhat")}
        rec = {"run": rel, "rows": int(n), "kept": int(len(kept)),
               "param_block_sha256": hashlib.sha256(
                   np.ascontiguousarray(rows[:, :len(names)]).tobytes()).hexdigest(),
               "parameters": params,
               "min_ess": nan_min(v["ess"] for v in params.values()),
               "max_rhat": nan_max(v["rhat"] for v in params.values()),
               "stored_diagnostics": stored,
               "stale": stored is None or stored["samples"] != len(kept),
               "counters": counter_totals(run_dir)}
        if hash_files:
            rec["files"] = {f: {"bytes": (run_dir / f).stat().st_size, "sha256": _sha256(run_dir / f)}
                            for f in ("chain_0.bin", "run_meta.json", "diagnostics.json", COUNTER_LOG)
                            if (run_dir / f).is_file()}
        per_chain.append(rec)
        chains.append(kept)
    pooled = len(chains) > 1
    summary = {}
    for j, name in enumerate(names):
        cols = [c[:, j] for c in chains]
        allx = np.concatenate(cols)
        circ = None
        if name in periodic:
            lo, hi = periodic[name]
            circ = circular_mean(allx, lo, hi)
            cols = [unwrap(c, lo, hi, centre=circ) for c in cols]
            allx = np.concatenate(cols)
        ess = float(sum(pc["parameters"][name]["ess"] for pc in per_chain))
        if pooled:      # gb_likelihood_matrix/collect.py pooled_diagnostics, verbatim segments
            half = min(len(c) for c in cols) // 2
            segs = np.stack([s for c in cols
                             for s in (c[:half], c[len(c) // 2:len(c) // 2 + half])])
            rhat = rhat_of_segments(segs) if half >= 2 else float("nan")
        else:
            rhat = per_chain[0]["parameters"][name]["rhat"]
        std = float(np.std(allx, ddof=1))
        q = np.quantile(allx, QUANTILES)
        summary[name] = {"mean": float(np.mean(allx)), "std": std,
                         **{f"q{int(round(100 * p)):02d}": float(v) for p, v in zip(QUANTILES, q)},
                         "circular_mean": circ, "periodic": list(periodic[name]) if name in periodic else None,
                         "ess": ess, "mcse_mean": std / math.sqrt(ess) if ess > 0 else float("inf"),
                         "rhat": rhat}
    counters = None
    if any(pc["counters"] for pc in per_chain):
        counters = {k: sum((pc["counters"] or {}).get(k, 0) for pc in per_chain) for k in COUNTERS}
        counters["counters_lower_bound"] = any((pc["counters"] or {}).get("counters_lower_bound")
                                               for pc in per_chain)
    neg_inf = None
    if counters:
        neg_inf = {"per_noise_state": (counters["pd_failed_states"] / counters["noise_states"]
                                       if counters["noise_states"] else None),
                   "per_row": (counters["pd_failed_rows"] / counters["evaluations"]
                               if counters["evaluations"] else None)}
    out = {"schema": 1, "toy": toy, "case": case, "model": model, "pooled": pooled,
           "runs": run_rels, "burn_fraction": BURN_FRACTION, "quantiles": list(QUANTILES),
           "ess_rule": ("sum over chains of each chain's min(batch, Geyer) ESS; R-hat over the chains' "
                        "halves" if pooled else "min(batch means floor(sqrt n) batches, Geyer); "
                        "split-R-hat over 4 segments"),
           "samples": int(sum(len(c) for c in chains)), "parameter_names": names,
           "parameters": summary,
           "min_ess": nan_min(v["ess"] for v in summary.values()),
           "max_rhat": nan_max(v["rhat"] for v in summary.values()),
           "per_chain": per_chain, "counters": counters, "neg_inf_fraction": neg_inf,
           "caveats": [CAVEATS[(toy, case, model)]] if (toy, case, model) in CAVEATS else []}
    if toy == "galactic_binary":
        cc = np.concatenate([c[:, names.index("cos_colat")] for c in chains])
        s = np.sign(anchor_cc)
        out["modes"] = {"primary_rule": PRIMARY_RULE, "injection_cos_colat": anchor_cc,
                        "P_primary": float(np.mean(np.sign(cc) == s)),
                        "P_reflection": float(np.mean(np.sign(cc) == -s))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--toy", choices=sorted(ROOTS), required=True)
    ap.add_argument("--workspace", type=Path, default=ROOT)
    ap.add_argument("--targets", default="", help="comma list of CASE/MODEL (default: all)")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of the chain files")
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    root = (a.workspace / ROOTS[a.toy]).resolve()
    out_dir = GOLDEN / "chain_summaries" / a.toy
    if root in out_dir.resolve().parents or out_dir.resolve() == root:
        raise SystemExit("refusing to write under the chains root")
    todo = targets(a.toy, root)
    if a.targets:
        keep = {tuple(t.split("/")) for t in a.targets.split(",")}
        todo = {k: v for k, v in todo.items() if k in keep}
        if len(todo) != len(keep):
            raise SystemExit(f"unknown targets: {sorted(keep - set(todo))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for (case, model), runs in sorted(todo.items()):
        s = summarise_target(a.toy, root, case, model, runs, hash_files=not a.no_hash)
        rel = f"chain_summaries/{a.toy}/{case}_{model}.json"
        text = json.dumps(s, indent=1, sort_keys=True) + "\n"
        (GOLDEN / rel).write_text(text)
        files[rel] = manifest_merge.sha256_file(GOLDEN / rel)
        print(f"[summary] {a.toy} {case}/{model}: samples {s['samples']}, min ESS {s['min_ess']:.1f}, "
              f"max R-hat {s['max_rhat']:.4f}", flush=True)
    manifest_merge.merge_entries(GOLDEN / "MANIFEST.json", files,
                                 provenance={rel: {"script": "scripts/golden/summarise_chains.py",
                                                   "root": ROOTS[a.toy]} for rel in files},
                                 replace=a.replace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
