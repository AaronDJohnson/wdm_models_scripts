"""Lineage-A worker: the NumPy oracle (wdm_toys) that wrote the sinusoid references (spec §7.1).

    $PY -P sinusoid_wt.py <out_dir> wd-check --data-root <dir>
    $PY -P sinusoid_wt.py <out_dir> evaluate --jobs <jobs.json> --data-root <dir> --cache-src <dir>
    $PY -P sinusoid_wt.py <out_dir> intermediates --data-root <dir> --cache-src <dir>

PYTHONPATH = lineage A only (toy_models/vendor/wdm/src : wdm_paper_2/toy_models/src).
Writes only into <out_dir>. The exact model's cache_dir is a private copy inside
<out_dir>, re-hashed after use (WT FastCovariance np.save's a missing c_N into it).
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


def _load_common():
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("golden_worker_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load_common()
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
NF, NT, DT = 64, 256, 2.0 ** -10
COMMANDS: dict[str, Callable[[Path, list[str]], None]] = {}


def command(name: str):
    def register(fn):
        COMMANDS[name] = fn
        return fn
    return register


@command("wd-check")
def cmd_wd_check(out_dir: Path, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="sinusoid_wt wd-check")
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args(argv)
    import wdm
    report = {"wdm_file": C.check_wdm(required=True), "cases": {}}
    for case in CASES:
        with np.load(a.data_root / case / "toy_data.npz", allow_pickle=False) as z:
            d, w_d = z["d"], z["w_d"]
        w = wdm.forward_time(d, NF, NT, DT)
        report["cases"][case] = {"array_equal": bool(np.array_equal(w, w_d)),
                                 "max_abs": float(np.max(np.abs(w - w_d))),
                                 "shape": list(w.shape), "sha256_w": C.sha256_array(w)}
    C.write_json(out_dir / "wd_check.json", report)


# --- evaluate (Task A6) --------------------------------------------------------------------
ANALYSIS_FMIN = 4.0     # gpu/run.py::_ANALYSIS_FMIN == make_bundle.ANALYSIS_FMIN


@command("evaluate")
def cmd_evaluate(out_dir: Path, argv: list[str]) -> None:
    """lnlike at the given points, built as make_matrix_bundle.cpu_references does:
    wdm_toys.matrix.build(model, CONFIGS[case](), load(path, cfg)["d"], analysis_fmin=4.0,
    cache_dir=<private copy> for exact only), then lik.lnlike(point) one point at a time."""
    ap = argparse.ArgumentParser(prog="sinusoid_wt evaluate")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-src", type=Path, required=True)
    a = ap.parse_args(argv)
    from wdm_toys import matrix as M
    from wdm_toys.config import CONFIGS
    from wdm_toys.data import load
    C.check_wdm(required=True)
    cache = out_dir / "cache"
    cache_sha = C.copy_cache(a.cache_src, cache)
    current, lik, build_seconds = None, None, 0.0
    jobs_out = {}
    for job in C.read_json(a.jobs):
        case, model, tag = job["case"], job["model"], job["tag"]
        if (case, model) != current:
            lik = None                                   # release the previous model first
            cfg = CONFIGS[case]()
            data = np.asarray(load(str(a.data_root / case / "toy_data.npz"), cfg)["d"], dtype=float)
            kw = {"cache_dir": str(cache)} if model == "exact" else {}
            t0 = time.perf_counter()
            lik = M.build(model, cfg, data, analysis_fmin=ANALYSIS_FMIN, **kw)
            build_seconds, current = time.perf_counter() - t0, (case, model)
        with np.load(job["points"], allow_pickle=False) as z:
            theta = np.asarray(z["theta"], dtype=float)
        if job.get("wrap_phase", False):
            theta = C.wrap_phase_runner(theta)
        values = np.empty(len(theta))
        seconds = np.empty(len(theta))
        for i, point in enumerate(theta):
            t0 = time.perf_counter()
            values[i] = lik.lnlike(point)
            seconds[i] = time.perf_counter() - t0
        np.savez(out_dir / f"{tag}.npz", lnlike=values, seconds=seconds, theta=theta)
        jobs_out[tag] = {"case": case, "model": model, "n": int(len(theta)), "build_seconds": build_seconds,
                         "class": type(lik).__name__, "pd_failed_rows": int(getattr(lik, "pd_failed_rows", 0))}
    C.assert_cache_unchanged(cache, cache_sha)
    C.write_json(out_dir / "evaluate.json", {"wdm_file": C.check_wdm(required=True),
                                             "runtime": C.runtime_fingerprint(),
                                             "cache_sha256": cache_sha, "jobs": jobs_out})


# --- intermediates (Task A9) -----------------------------------------------------------------
MODELS = ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact",
          "mid_raw", "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad",
          "tri_exact", "exact")
ROW_SAMPLE = 10


def _marginal_parts(M, lik, theta) -> dict:
    """wdm_toys.matrix.MatrixWdm.lnlike + marginal_lnlike split into their terms. The
    arithmetic is copied line for line, so the recombined value equals lnlike bitwise."""
    from scipy.linalg import cho_factor, cho_solve
    theta = np.asarray(theta, dtype=float)
    n = len(lik.cfg.source_names)
    log10_A_n, alpha_c, alpha_d = theta[n:]
    h = M.signal(lik.cfg, *theta[:n])
    y = lik.y_data - (h if lik.edge == "pad" else lik.v.map(h))
    solver = lik.solver(alpha_c, alpha_d)
    A_n = 10.0 ** log10_A_n
    u = lik.u
    Ciy = solver.solve(y)
    quad = float(y @ Ciy)
    quad_C = quad
    logdet = solver.logdet()
    logdet_C = logdet
    n_eff = lik.n_cells
    quad_H = logdet_H = 0.0
    if u is not None and u.shape[1]:
        Ciu = solver.solve(u)
        H = u.T @ Ciu
        cf = cho_factor(0.5 * (H + H.T), lower=True, check_finite=False)
        b = u.T @ Ciy
        quad_H = float(b @ cho_solve(cf, b, check_finite=False))
        quad -= quad_H
        logdet_H = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        logdet += logdet_H
        n_eff -= u.shape[1]
    lnL = -0.5 * (quad / A_n + logdet + n_eff * np.log(A_n) + n_eff * M._LOG_2PI) + lik.log_jac
    return {"quad_C": quad_C, "logdet_C": float(logdet_C), "quad_H": quad_H, "logdet_H": logdet_H,
            "quad": quad, "logdet": float(logdet), "n_cells": int(lik.n_cells),
            "n_u": 0 if u is None else int(u.shape[1]), "n_eff": int(n_eff), "A_n": float(A_n),
            "log_jac": float(lik.log_jac), "lnL": float(lnL)}


def _whittle_parts(model: str, lik, theta) -> dict:
    """FreqWhittle.lnlike (analysis_fmin > 0 branch) / MatrixWhittle.lnlike, split; same arithmetic."""
    from wdm_toys.likelihood import _LOG_2PI
    r, (log10_A_n, alpha) = lik._split(theta)
    if model == "whittle_raw":
        A_n = 10.0 ** log10_A_n
        keep = lik.band.keep_rfft
        R = np.fft.rfft(r)[keep]
        lam = A_n * np.exp(alpha * lik._L[keep]) / (2.0 * lik.cfg.dt)
        quad = 2.0 / lik.cfg.N * float(np.sum(np.abs(R) ** 2 / lam))
        logdet = 2.0 * float(np.sum(np.log(lam)))
        n_keep = lik.band.n_keep
    else:
        x = r[lik.crop]
        if lik.w is not None:
            x = x * lik.w
        R = np.fft.rfft(x)[lik.keep]
        lam = lik.U * 10.0 ** log10_A_n * np.exp(alpha * lik.Lc) / (2.0 * lik.cfg.dt)
        quad = 2.0 / lik.Nc * float(np.sum(np.abs(R) ** 2 / lam))
        logdet = 2.0 * float(np.sum(np.log(lam)))
        n_keep = lik.n_keep_c
    lnL = -0.5 * (quad + logdet + n_keep * _LOG_2PI)
    out = {"quad": quad, "logdet": logdet, "n_keep": int(n_keep), "lnL": float(lnL),
           "lam": C.array_summary(lam)}
    if model == "whittle_cut":
        out["crop"] = [int(lik.crop.start), int(lik.crop.stop)]       # [CUT_START*Nf, (Nt-CUT_END)*Nf)
        out["Nc"] = int(lik.Nc)
    return out


def _exact_parts(lik, theta) -> dict:
    """FastFullCovariance.lnlike split (BandCovariance quad and logdet); same arithmetic."""
    import hashlib
    import json
    from wdm_toys.likelihood import _LOG_2PI
    r, (log10_A_n, alpha_c, alpha_d) = lik._split(theta)
    A_n = 10.0 ** log10_A_n
    quad_unit = lik.cov.quad(r, alpha_c, alpha_d)
    quad = quad_unit / A_n
    n = lik.band.n_keep
    logdet_unit = lik.cov.logdet(alpha_c, alpha_d)
    logdet = logdet_unit + n * float(np.log(A_n))
    lnL = -0.5 * (quad + logdet + n * _LOG_2PI)
    full_logdet = float(lik.cov.full.logdet(alpha_c, alpha_d))
    identity = dict(lik.cov.identity)
    return {"quad_unit": float(quad_unit), "logdet_unit": float(logdet_unit), "full_logdet": full_logdet,
            "weyl": float(lik.cov.full._weyl(alpha_c, alpha_d)), "c_N": float(lik.cov.full.c_N),
            "correction": float(logdet_unit) - full_logdet, "n_keep": int(n), "A_n": float(A_n),
            "quad": float(quad), "logdet": float(logdet), "lnL": float(lnL), "bandcov_identity": identity,
            "bandcov_key": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20],
            "bandcov_file": lik.cov.artifact_path.name}


def _fro(blocks) -> list[float]:
    return [0.0 if b is None else float(np.linalg.norm(b)) for b in blocks]


def _blocks_summary(diag, up) -> dict:
    """Per-block Frobenius norms, sha256 of all blocks (C order, diag then up), 50 entries."""
    import hashlib
    blocks = [("diag", k, b) for k, b in enumerate(diag)] + [("up", k, b) for k, b in enumerate(up)]
    digest = hashlib.sha256()
    for _, _, b in blocks:
        if b is not None:
            digest.update(np.ascontiguousarray(b).tobytes())
    usable = [(kind, k, b) for kind, k, b in blocks if b is not None and b.size]
    rng = np.random.default_rng(C.SAMPLE_SEED)
    samples = []
    for _ in range(50):
        kind, k, b = usable[int(rng.integers(len(usable)))]
        i, j = int(rng.integers(b.shape[0])), int(rng.integers(b.shape[1]))
        samples.append([kind, k, i, j, float(b[i, j])])
    return {"n_diag": len(diag), "n_up": len(up), "sha256": digest.hexdigest(),
            "diag_fro": _fro(diag), "up_fro": _fro(up),
            "diag_shapes": [None if b is None else list(b.shape) for b in diag], "samples": samples}


@command("intermediates")
def cmd_intermediates(out_dir: Path, argv: list[str]) -> None:
    """G6 at the manuscript_mid truth (alpha_c = -2, alpha_d = 1.2; A_n = 1 inside the blocks)."""
    ap = argparse.ArgumentParser(prog="sinusoid_wt intermediates")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-src", type=Path, required=True)
    a = ap.parse_args(argv)
    from wdm_toys import matrix as M
    from wdm_toys.config import CONFIGS
    from wdm_toys.data import load
    C.check_wdm(required=True)
    cache = out_dir / "cache"
    cache_sha = C.copy_cache(a.cache_src, cache)
    case = "manuscript_mid"
    cfg = CONFIGS[case]()
    d = np.asarray(load(str(a.data_root / case / "toy_data.npz"), cfg)["d"], dtype=float)
    ac, ad = float(cfg.alpha_c), float(cfg.alpha_d)
    report = {"case": case, "alpha_c": ac, "alpha_d": ad, "variances": {}, "maps": {}, "tri_blocks": {},
              "parts": {}}
    # G5 edge constants of the oracle (the wrap rule's columns; no frozen lineage computes the
    # wrap fractions themselves -- S1 pins those from the spec via wrap_cut_columns).
    report["edges"] = {"cut": [int(M.CUT_START), int(M.CUT_END)],
                       "edge_exact": [int(M.EDGE_START), int(M.EDGE_END)],
                       "pad_columns": int(M.PAD_COLUMNS),
                       "cut_columns": [int(c) for c in M.cut_columns(int(cfg.Nt))],
                       "edge_columns": [int(c) for c in M.edge_columns(int(cfg.Nt))]}
    arrays = {}
    g_raw, g_pad = M.make_variant(cfg, "raw").grid, M.make_variant(cfg, "pad").grid
    for base, fn in (("band", M.band_variance), ("mid", M.mid_variance)):
        v = fn(g_raw, ac, ad)
        arrays[f"{base}_variance_raw"] = v
        report["variances"][f"{base}_raw"] = C.array_summary(v)
        report["variances"][f"{base}_pad"] = C.array_summary(fn(g_pad, ac, ad))
    for model in MODELS:
        kw = {"cache_dir": str(cache)} if model == "exact" else {}
        lik = M.build(model, cfg, d, analysis_fmin=ANALYSIS_FMIN, **kw)
        truth = np.asarray(lik.truth, dtype=float)
        if model.startswith("whittle"):
            parts = _whittle_parts(model, lik, truth)
        elif model == "exact":
            parts = _exact_parts(lik, truth)
        else:
            parts = _marginal_parts(M, lik, truth)
            report["maps"][model] = {"n_cells": int(lik.n_cells), "log_jac": float(lik.log_jac),
                                     "y_data": C.array_summary(lik.y_data), "u": C.array_summary(lik.u)}
            if model == "band_pad":
                report["maps"]["Wp"] = C.array_summary(lik.Wp)
            if model.startswith("tri_"):
                report["tri_blocks"][model] = _blocks_summary(*lik.tri.blocks(ac, ad))
            if model == "band_exact":
                rows = lik._exact_rows(ac, ad)
                pick = np.linspace(0, rows.shape[0] - 1, ROW_SAMPLE).round().astype(np.int64)
                arrays["exact_rows_pick"] = pick
                arrays["exact_rows_sample"] = rows[pick]
                arrays["exact_rows_l2"] = np.linalg.norm(rows, axis=1)
                report["exact_rows"] = {**C.array_summary(rows), "n_edge": int(lik.e.size),
                                        "n_interior": int(lik.i.size),
                                        "edge_cells_sha256": C.sha256_array(lik.e)}
                del rows
        lnlike = float(lik.lnlike(truth))
        if parts["lnL"] != lnlike:
            raise AssertionError(f"{model}: split terms give {parts['lnL']!r}, lnlike gives {lnlike!r}")
        parts["lnlike"] = lnlike
        report["parts"][model] = parts
        del lik
    C.assert_cache_unchanged(cache, cache_sha)
    report["wdm_file"] = C.check_wdm(required=True)
    report["runtime"] = C.runtime_fingerprint()
    np.savez(out_dir / "intermediates.npz", **arrays)
    C.write_json(out_dir / "intermediates.json", report)


def main() -> None:
    out_dir, argv = C.start(("A",))
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"command must be one of {sorted(COMMANDS)}")
    COMMANDS[argv[0]](out_dir, argv[1:])


if __name__ == "__main__":
    main()
