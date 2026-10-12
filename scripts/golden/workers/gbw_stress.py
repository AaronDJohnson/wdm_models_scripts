"""G4: GB stress points per target, evaluated by lineage C on NumPy (and CPU-JAX).

Points (fixed order; the counters depend on it):

 0 ``anchor``          inject_glass + mu_fit (mu_freq for Whittle)
 1 ``noise_hi``        every noise parameter at the box corner +8
 2 ``noise_lo``        every noise parameter at -8
 3 ``noise_mixed``     channel A at +8, channel E at -8
 4-8 envelope models:  ``edge_h<k>``: anchor with harmonic k of both channels at +7.999 (k even)
                       or -7.999 (k odd); Whittle models: ``edge_A_hi`` (7.999, mu_E),
                       ``edge_E_lo`` (mu_A, -7.999), ``edge_cross`` (-7.999, 7.999),
                       ``level_up`` (mu + 1), ``level_down`` (mu - 1)
 9 ``source_out_lnA``  anchor with lnA = prior_hi[3] + 1   (outside the GLASS box)
10 ``source_out_f0``   anchor with f0T = prior_lo[0] - 1
11 ``scan``            tier ``full`` only: the first of 64 draws mu ~ U[-8, 8] (default_rng(20261012),
                       source at the injection) whose covariance fails a NumPy Cholesky in either
                       channel, else the 64th draw

Per point: ``lnl_ref`` (``make_lnlike``, NumPy; LinAlgError -> -inf, flagged), ``lnl_cache`` (the
NumPy noise-state cache class the sampler uses: FullCache / EdgeSchurCache / MatrixCache,
the whole point set in ONE call, so its counters are pinned), ``lnl_jax`` (``jax_models.build``),
the PD margin ``lambda_min / max|lambda|`` of each channel's covariance (dense assembled matrix;
for band/mid_exact also the Schur complement), tagged ``marginal`` when ``|margin| < 1e-10``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

EDGE = 7.999
MARGINAL = 1e-10
SCAN_DRAWS = 64
SCAN_SEED = 20261012


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR stress")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", choices=gbw_lineage_c.CASES, required=True)
    ap.add_argument("--tier", choices=("fast", "full"), default="fast")
    ap.add_argument("--backends", default="numpy,jax")
    ap.add_argument("--models", default="")
    return ap.parse_args(argv)


def corner_points(L, prep, model) -> list[tuple[str, np.ndarray]]:
    anc = prep.anchor(model)
    nn = anc.size - 8
    pts = [("anchor", anc.copy())]
    for name, fill in (("noise_hi", 8.0), ("noise_lo", -8.0)):
        p = anc.copy()
        p[8:] = fill
        pts.append((name, p))
    p = anc.copy()
    p[8:8 + nn // 2], p[8 + nn // 2:] = 8.0, -8.0
    pts.append(("noise_mixed", p))
    if L.config.is_stationary(model):
        mu = anc[8:]
        for name, v in (("edge_A_hi", [EDGE, mu[1]]), ("edge_E_lo", [mu[0], -EDGE]),
                        ("edge_cross", [-EDGE, EDGE]), ("level_up", mu + 1.0),
                        ("level_down", mu - 1.0)):
            p = anc.copy()
            p[8:] = v
            pts.append((name, p))
    else:
        ne = nn // 2
        for k in range(ne):
            p = anc.copy()
            p[8 + k] = p[8 + ne + k] = EDGE if k % 2 == 0 else -EDGE
            pts.append((f"edge_h{k}", p))
    p = anc.copy()
    p[3] = prep.prior_hi[3] + 1.0
    pts.append(("source_out_lnA", p))
    p = anc.copy()
    p[0] = prep.prior_lo[0] - 1.0
    pts.append(("source_out_f0", p))
    assert len(pts) == 11
    return pts


class Covariances:
    """Each channel's covariance of one model as the NumPy reference assembles it."""

    def __init__(self, L, prep, model):
        self.L, self.prep, self.model = L, prep, model
        self.k = prep.freq
        self.schur = None
        if model == "whittle_cut":
            self.W = L.matrix.whittle_map(prep, model)
        elif model not in ("whittle_raw", "exact"):
            edge = model.split("_")[1]
            self.V = L.matrix.variant(prep, edge)
            self.cov, self.diagonal = L.matrix.covariance_builder(prep, model, self.V, np)
            if model in L.matrix.SCHUR:
                self.approx, _ = L.matrix.covariance_builder(prep, model.split("_")[0] + "_raw",
                                                             self.V, np)
                self.rows_of = L.matrix.exact_rows_builder(prep, self.V, np)
                self.E = np.asarray(self.V.exact_rows)
                self.I = np.setdiff1d(np.arange(self.V.P), self.E)

    def matrix(self, mu_c, c):
        """('diag', v) or ('dense', S) or ('herm', M)."""
        p, k = self.prep, self.k
        if self.model == "whittle_raw":
            return "diag", p.sI[c, k] + np.exp(mu_c[0]) * p.sG[c, k]
        if self.model == "whittle_cut":
            return "diag", self.W.sI[c] + np.exp(mu_c[0]) * self.W.sG[c]
        if self.model == "exact":
            return "herm", self.L.likelihoods.full_covariance(mu_c, p.sI[c, k], p.sG[c, k],
                                                               p.basis_L, np)
        S = self.cov(mu_c, c)
        return ("diag", S) if self.diagonal else ("dense", 0.5 * (S + S.T))

    def schur_complement(self, mu_c, c):
        vI = self.approx(mu_c, c)[self.I]
        R = self.rows_of(mu_c, c)
        S = 0.5 * (R[:, self.E] + R[:, self.E].T) - (R[:, self.I] / vI[None, :]) @ R[:, self.I].T
        return 0.5 * (S + S.T)


def _margin(kind, M) -> float:
    if kind == "diag":
        return float(np.min(M) / np.max(np.abs(M)))
    ev = np.linalg.eigvalsh(M)
    return float(ev[0] / np.max(np.abs(ev)))


def _chol_ok(kind, M) -> bool:
    if kind == "diag":
        return bool(np.all(M > 0))
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def _split(theta, nn):
    mu = theta[8:]
    return [mu[: nn // 2], mu[nn // 2:]]


def scan_point(cov: Covariances, prep, model) -> tuple[np.ndarray, dict]:
    anc = prep.anchor(model)
    nn = anc.size - 8
    draws = np.random.default_rng(SCAN_SEED).uniform(-8.0, 8.0, size=(SCAN_DRAWS, nn))
    for i, mu in enumerate(draws):
        th = anc.copy()
        th[8:] = mu
        ok = [_chol_ok(*cov.matrix(m, c)) for c, m in enumerate(_split(th, nn))]
        if not all(ok):
            return th, {"draw": i, "found_non_pd": True}
    th = anc.copy()
    th[8:] = draws[-1]
    return th, {"draw": SCAN_DRAWS - 1, "found_non_pd": False}


def _cache_class(L, prep, model):
    if model == "exact":
        return L.fullcache.FullCache(prep)
    if model in L.matrix.SCHUR:
        return L.matrix.EdgeSchurCache(prep, model)
    if model in L.matrix.CACHED:
        return L.matrix.MatrixCache(prep, model)
    return None


def run(out_dir, argv) -> None:
    a = _parse(argv)
    backends = [b for b in a.backends.split(",") if b]
    L = gbw_lineage_c.load(a.workspace, jax="jax" in backends)
    pay = gbw_common.Payload(out_dir, lineage="C", worker="gb_legacy", subcommand="stress", args=argv,
                             base=L.workspace)
    prep_path = a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{a.case}.npz"
    prep_sha = pay.add_input(prep_path)
    chains_manifest = L.workspace / gbw_lineage_c.CHAINS / "manifest.json"
    pay.add_input(chains_manifest)
    models = sorted(gbw_lineage_c.targets(json.loads(chains_manifest.read_text()), a.case))
    models = [m for m in models if not a.models or m in a.models.split(",")]
    prep = L.likelihoods.Prepared(prep_path)
    np_be = L.likelihoods.numpy_backend()
    for model in models:
        cov = Covariances(L, prep, model)
        pts = corner_points(L, prep, model)
        scan_info = None
        if a.tier == "full":
            th, scan_info = scan_point(cov, prep, model)
            pts.append(("scan", th))
        X = np.stack([p for _, p in pts])
        nn = X.shape[1] - 8
        f_ref = L.likelihoods.make_lnlike(prep, model, np_be)
        ref, raised = [], []
        for th in X:
            try:
                ref.append(float(f_ref(th)))
                raised.append(False)
            except np.linalg.LinAlgError:
                ref.append(-np.inf)
                raised.append(True)
        cache = _cache_class(L, prep, model)
        lnl_cache = np.asarray(cache(X), dtype=np.float64) if cache is not None else None
        counters_cache = cache.counters() if cache is not None else None
        lnl_jax, counters_jax = None, None
        if "jax" in backends:
            J = L.jax_models.build(prep, model)
            if hasattr(J, "reset_counters"):
                J.reset_counters()
            lnl_jax = np.asarray(J(X), dtype=np.float64)
            counters_jax = J.counters() if hasattr(J, "counters") else None
            del J
        margins = {}
        points = []
        for i, (name, th) in enumerate(pts):
            key = th[8:].tobytes()
            if key not in margins:
                per = []
                schur = []
                for c, mu_c in enumerate(_split(th, nn)):
                    kind, M = cov.matrix(mu_c, c)
                    per.append(_margin(kind, M))
                    if model in L.matrix.SCHUR:
                        schur.append(_margin("dense", cov.schur_complement(mu_c, c)))
                margins[key] = (per, schur)
            per, schur = margins[key]
            margin = min(per)
            source_out = name.startswith("source_out")
            rec = {"index": i, "name": name, "theta": th.tolist(), "lnl_ref": ref[i],
                   "ref_raised": raised[i], "pd_margin_channels": per, "pd_margin": margin,
                   "marginal": abs(margin) < MARGINAL, "source_out": source_out,
                   "expected_neg_inf": bool(source_out or margin < 0)}
            if schur:
                rec["schur_margin_channels"] = schur
            if lnl_cache is not None:
                rec["lnl_cache"] = float(lnl_cache[i])
            if lnl_jax is not None:
                rec["lnl_jax"] = float(lnl_jax[i])
            rec["mask_consistent"] = bool(rec["marginal"]
                                          or np.isneginf(ref[i]) == rec["expected_neg_inf"])
            points.append(rec)
        out = {"schema": 1, "case": a.case, "model": model, "tier": a.tier,
               "n_points": len(points), "points": points, "scan": scan_info,
               "counters_cache": counters_cache, "counters_jax": counters_jax,
               "cache_class": None if cache is None else type(cache).__name__,
               "backends": backends, "marginal_threshold": MARGINAL,
               "prepared_sha256": prep_sha, "build": gbw_common.build_info()}
        pay.write_json(f"galactic_binary/stress/{a.case}/{model}.{a.tier}.json", out)
        n_inf = sum(np.isneginf(p["lnl_ref"]) for p in points)
        n_marg = sum(p["marginal"] for p in points)
        print(f"[stress] {a.case}/{model}: {len(points)} points, -inf {n_inf}, marginal {n_marg}, "
              f"cache {counters_cache}", flush=True)
        del cov, f_ref, cache
        gc.collect()
    pay.close()
