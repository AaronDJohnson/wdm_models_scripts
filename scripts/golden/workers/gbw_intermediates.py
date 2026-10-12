"""G6 (GB part): per-module intermediates at the anchor of one case, from the frozen code.

``galactic_binary/intermediates_<case>.npz`` holds small arrays verbatim;
``galactic_binary/intermediates_<case>.json`` holds fingerprints (``gbw_common.fingerprint``)
of the large ones and scalars. Anchor: ``inject_glass`` with ``mu_fit`` (envelope models)
or ``mu_freq`` (Whittle). Contents:

* every variant (raw, cut, exact, pad): ``V.C``, ``V.U``, ``V.b``, ``V.n`` fingerprints, the
  store-column slice, ``P``, grid geometry; cells of the store data for raw and pad;
  ``V.exact_rows``; ``PadMask`` ``E``, ``Kpos``, ``AE`` (fingerprint + row 0), ``k0``, ``R`` grid;
* Eq. (45): ``K^I``, ``K^G`` fingerprints and diagonals, ``half``, ``basis_half``;
* ``band_psd`` (band, mid); the band/mid variance vectors at the anchor;
* the dense tri_raw and {band,mid,tri}_pad covariances at the anchor (fingerprint + diagonal);
* the exact edge rows ``1/2 Re(U_E M' U^H)`` at the anchor (fingerprint + 3 rows);
* ``M'`` per channel (fingerprint, ``q0``, ``log|M'|``) and ``ghat`` at lags ``0..n-1``;
* the gbonfly store template at the injection;
* ``(quad, logdet)`` per channel for every target model of the case, re-summed and checked
  against ``make_lnlike`` at the anchor.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

LOG_2PI = float(np.log(2.0 * np.pi))


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR intermediates")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", choices=gbw_lineage_c.CASES, required=True)
    return ap.parse_args(argv)


def quad_logdet(L, prep, model: str, V: dict, mu_env, mu_freq):
    """``(2, 2)`` [[quad_A, logdet_A], [quad_E, logdet_E]] and the re-summed lnL at the anchor,
    following each family's code path in ``matrix.make_lnlike`` / ``fullcache``."""
    np_be = L.likelihoods.numpy_backend()
    st = L.matrix.store(prep)
    p8 = prep.inject
    out = np.empty((2, 2))
    total = 0.0
    if model == "whittle_raw":
        k = prep.freq
        r = prep.z - np.asarray(prep.template(p8, np))
        for c in range(2):
            m = prep.sI[c, k] + np.exp(mu_freq[c]) * prep.sG[c, k]
            out[c] = np.sum(np.abs(r[c, k]) ** 2 / m), np.sum(np.log(m))
            total = total - out[c, 0] - out[c, 1]
        return out, total
    if model == "whittle_cut":
        W = L.matrix.whittle_map(prep, model)
        d = (W.A @ st.z.T + W.B @ np.conj(st.z).T).T
        h = np.asarray(L.matrix.store_template(prep, p8, np))
        hc = h @ W.A.T + np.conj(h) @ W.B.T
        for c in range(2):
            m = W.sI[c] + np.exp(mu_freq[c]) * W.sG[c]
            out[c] = np.sum(np.abs(d[c] - hc[c]) ** 2 / m), np.sum(np.log(m))
            total = total - out[c, 0] - out[c, 1]
        return out, total
    if model == "exact":
        k = prep.freq
        block = prep.NT // 2 if L.fullcache.is_periodic(prep) else None
        h = np.asarray(prep.template(p8, np))[:, k]
        for c in range(2):
            q0, logdet, y, factor = L.fullcache.noise_state(mu_env[c], prep.sI[c, k], prep.sG[c, k],
                                                            prep.basis_L, prep.z[c, k], block, np)
            t = L.fullcache.forward(factor, h[c], np)
            out[c] = q0 - 2.0 * np.real(np.vdot(h[c], y)) + np.real(np.vdot(t, t)), logdet
            total = total - out[c, 0] - out[c, 1]
        return out, total
    v = V[model.split("_")[1]]
    cov, diagonal = L.matrix.covariance_builder(prep, model, v, np)
    d = L.matrix.cells(v, st.z)
    h = np.asarray(L.matrix.store_template(prep, p8, np))[:, v.cols]
    r = d - np.real(h @ v.C.T)
    for c in range(2):
        if diagonal:
            var = cov(mu_env[c], c)
            out[c] = np.sum(r[c] ** 2 / var), np.sum(np.log(var))
        else:
            out[c] = L.matrix._dense_quad_logdet(cov(mu_env[c], c), r[c], np_be)
        total = total - 0.5 * (out[c, 0] + out[c, 1] + v.P * LOG_2PI)
    return out, total


def run(out_dir, argv) -> None:
    a = _parse(argv)
    L = gbw_lineage_c.load(a.workspace)
    pay = gbw_common.Payload(out_dir, lineage="C", worker="gb_legacy", subcommand="intermediates",
                             args=argv,
                             base=L.workspace)
    case = a.case
    prep_path = a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{case}.npz"
    pay.add_input(prep_path)
    chains_manifest = L.workspace / gbw_lineage_c.CHAINS / "manifest.json"
    pay.add_input(chains_manifest)
    models = sorted(gbw_lineage_c.targets(json.loads(chains_manifest.read_text()), case))
    prep = L.likelihoods.Prepared(prep_path)
    st = L.matrix.store(prep)
    mu_env, mu_freq = prep.mu_fit, prep.mu_freq
    arrays, fps, scalars = {}, {}, {}

    V = {e: L.matrix.variant(prep, e) for e in ("raw", "cut", "exact", "pad")}
    for e, v in V.items():
        for name, arr in (("C", v.C), ("U", v.U), ("b", v.b), ("n", v.n)):
            fps[f"V_{e}_{name}"] = gbw_common.fingerprint(arr)
        scalars[f"V_{e}"] = {"P": v.P, "cols": [v.cols.start, v.cols.stop], "NT": v.grid.NT,
                             "N": v.grid.N, "g0": v.grid.g0, "G": v.grid.G,
                             "bands": v.grid.bands.tolist()}
    for e in ("raw", "pad"):
        arrays[f"V_{e}_cells"] = L.matrix.cells(V[e], st.z)
    fps["V_cut_cells"] = gbw_common.fingerprint(L.matrix.cells(V["cut"], st.z))
    arrays["V_exact_rows"] = np.asarray(V["exact"].exact_rows, dtype=np.int64)
    m = V["pad"].mask
    arrays["pad_E"], arrays["pad_Kpos"] = m.E, m.Kpos
    arrays["pad_AE_row0"] = m.AE[0]
    fps["pad_AE"] = gbw_common.fingerprint(m.AE)
    scalars["pad_mask"] = {"k0": m.k0, "n_rows": int(m.E.size), "R_NT": m.grid.NT, "R_nB": m.grid.nB,
                           "R_g0": m.grid.g0, "R_G": m.grid.G, "R_bands": m.grid.bands.tolist()}

    K = L.matrix.tri_kernels(prep, V["raw"])
    fps["KI"], fps["KG"] = gbw_common.fingerprint(K.KI), gbw_common.fingerprint(K.KG)
    fps["half"] = gbw_common.fingerprint(K.half)
    arrays["KI_diag"] = np.stack([np.diagonal(K.KI[c]) for c in range(2)])
    arrays["KG_diag"] = np.stack([np.diagonal(K.KG[c]) for c in range(2)])
    arrays["basis_half"] = K.basis_half
    del K
    for how in ("band", "mid"):
        sI, sG = L.matrix.band_psd(prep, V["raw"].grid, how)
        arrays[f"bandpsd_{how}_sI"], arrays[f"bandpsd_{how}_sG"] = sI, sG
        cov, diagonal = L.matrix.covariance_builder(prep, f"{how}_raw", V["raw"], np)
        assert diagonal
        arrays[f"var_{how}_raw"] = np.stack([cov(mu_env[c], c) for c in range(2)])
    for model, edge in (("tri_raw", "raw"), ("band_pad", "pad"), ("mid_pad", "pad"), ("tri_pad", "pad")):
        cov, _ = L.matrix.covariance_builder(prep, model, V[edge], np)
        diag = []
        for c in range(2):
            S = cov(mu_env[c], c)
            fps[f"cov_{model}_c{c}"] = gbw_common.fingerprint(S)
            diag.append(np.diagonal(S).copy())
        arrays[f"cov_{model}_diag"] = np.stack(diag)
        del cov
        gc.collect()
    nE = V["exact"].exact_rows.size
    if nE:
        rows_of = L.matrix.exact_rows_builder(prep, V["exact"], np)
        sel = [0, nE // 2, nE - 1]
        for c in range(2):
            R = rows_of(mu_env[c], c)
            fps[f"exact_rows_c{c}"] = gbw_common.fingerprint(R)
            arrays[f"exact_rows_c{c}_sel"] = R[sel]
        scalars["exact_rows_selected"] = sel
    k = prep.freq
    block = prep.NT // 2 if L.fullcache.is_periodic(prep) else None
    scalars["exact_block"] = block
    for c in range(2):
        q0, logdet = L.fullcache.noise_state(mu_env[c], prep.sI[c, k], prep.sG[c, k], prep.basis_L,
                                             prep.z[c, k], block, np, with_factor=False)
        scalars[f"Mprime_c{c}"] = {"q0": float(q0), "logdet": float(logdet)}
        fps[f"Mprime_c{c}"] = gbw_common.fingerprint(
            L.likelihoods.full_covariance(mu_env[c], prep.sI[c, k], prep.sG[c, k], prep.basis_L, np))
        arrays[f"ghat_c{c}"] = L.likelihoods.ghat_at(mu_env[c], prep.basis_L,
                                                      np.arange(k.stop - k.start), np)
    arrays["store_template"] = np.asarray(L.matrix.store_template(prep, prep.inject, np))

    np_be = L.likelihoods.numpy_backend()
    ql, checks = [], {}
    for model in models:
        parts, total = quad_logdet(L, prep, model, V, mu_env, mu_freq)
        ref = float(L.likelihoods.make_lnlike(prep, model, np_be)(prep.anchor(model)))
        if not abs(total - ref) <= 1e-9 * max(1.0, abs(ref)):
            raise RuntimeError(f"{case}/{model}: re-summed lnL {total!r} != make_lnlike {ref!r}")
        ql.append(parts)
        checks[model] = {"lnl_resummed": float(total), "lnl_make_lnlike": ref,
                         "abs_diff": abs(float(total) - ref)}
        gc.collect()
    arrays["quad_logdet"] = np.stack(ql)
    meta = {"schema": 1, "case": case, "anchor": {"inject_glass": prep.inject, "mu_fit": mu_env,
                                                   "mu_freq": mu_freq},
            "quad_logdet_models": models, "lnl_at_anchor": checks, "scalars": scalars,
            "fingerprints": fps, "arrays": sorted(arrays), "build": gbw_common.build_info()}
    pay.write_npz(f"galactic_binary/intermediates_{case}.npz", arrays)
    pay.write_json(f"galactic_binary/intermediates_{case}.json", meta)
    pay.close()
    print(f"[intermediates] {case}: {len(arrays)} arrays, {len(fps)} fingerprints, "
          f"{len(models)} models re-summed", flush=True)
