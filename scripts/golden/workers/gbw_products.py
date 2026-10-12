"""G5 (GB part): data products of the three prepared cases, from the frozen code.

Writes ``galactic_binary/products.json`` and ``galactic_binary/psd_window.npz``:

* gbonfly template sums at the injection (BW = 4096, ``orbit_cache(T, 0, 4096)``, prepared
  units) on the analysis window and on the store: ``sum |h|^2`` and ``Re <z, h>``;
* ``matrix.describe`` of the 15 paper models and the per-channel counts table (spec 4.9);
* ``snr_full`` recomputed (``prepare.snr_full`` at ``mu_fit``) and the report's value;
* the stored prior box and ``gbonfly.ucb_prior`` on the source band (bitwise flags);
* ``inject_glass`` against ``gbonfly.params_from_native`` of the segment's native injection;
* the envelope: ``C`` (``orthonormal_coefficients``), Table III's ``C^T mu_fit`` per channel,
  the prior half-widths ``min_i 8 sum_j |C_ji|`` and ``basis_L == ghat_grid(T)``;
* the instrument and GLASS foreground PSDs (``prepare.psd_components``) on the window bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

NATIVE_KEYS = ("f0", "fdot", "amp", "phi0", "colat", "lon", "psi", "cos_iota")


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR products")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    return ap.parse_args(argv)


def paper_models(matrix) -> list[str]:
    return [m for m in matrix.MODELS if "win" not in m]


def run(out_dir, argv) -> None:
    a = _parse(argv)
    L = gbw_lineage_c.load(a.workspace)
    pay = gbw_common.Payload(out_dir, lineage="C", worker="gb_legacy", subcommand="products",
                             args=argv,
                             base=L.workspace)
    data = a.data_dir.resolve() / "galactic_binary"
    models = paper_models(L.matrix)
    assert len(models) == 15, models
    cases, psd = {}, {}
    for case in gbw_lineage_c.CASES:
        months = int(case[3:-2])
        prep_path = data / "prepared" / f"{case}.npz"
        report_path = data / "prepared" / f"{case}_report.json"
        seg_path = data / "segments" / f"segment_snr7_{months}mo_3bands.npz"
        for p in (prep_path, report_path, seg_path):
            pay.add_input(p)
        report = json.loads(report_path.read_text())
        prep = L.likelihoods.Prepared(prep_path)
        arr = prep.arrays
        p8 = prep.inject
        hw = np.asarray(prep.template(p8, np))
        hs = np.asarray(L.matrix.store_template(prep, p8, np))
        template = {
            "window": {"sum_h2": float(np.sum(np.abs(hw) ** 2)),
                       "re_zh": float(np.real(np.vdot(prep.z, hw))), "bins": int(prep.n_bins)},
            "store": {"sum_h2": float(np.sum(np.abs(hs) ** 2)),
                      "re_zh": float(np.real(np.vdot(arr["store_z"], hs))), "bins": int(arr["store_n"])},
            "BW": int(L.config.TEMPLATE_BW), "backend": "gbonfly.compat.template_on_window (fourier)",
            "orbit": "gbonfly.orbit_cache(T, 0.0, TEMPLATE_BW)",
            "units": "template_on_window(...) / (DT * sigma0[:, None])"}
        describe = {m: L.matrix.describe(prep, m) for m in models}
        counts = {"whittle": describe["whittle_raw"]["P"], "whittle_cut": describe["whittle_cut"]["P"],
                  "wdm": describe["band_raw"]["P"], "cut": describe["band_cut"]["P"],
                  "exact_edge_rows": describe["band_exact"]["exact_rows"],
                  "pad_mask_rows": describe["band_pad"]["mask_rows"],
                  "lambda_R": describe["band_pad"]["lambda_cells"]}
        block = prep.NT // 2 if L.fullcache.is_periodic(prep) else None
        d = dict(sI=prep.sI, sG=prep.sG, basis_L=prep.basis_L, resid=arr["resid"],
                 injected=arr["injected"])
        snr = L.prepare.snr_full(d, prep.mu_fit, prep.freq, block)
        src = report["source_band_bins"]
        df = float(arr["df"])
        prior = L.gbonfly.ucb_prior(T=prep.T, fmin=src[0] * df, fmax=(src[0] + src[1] - 1) * df)
        with np.load(seg_path, allow_pickle=False) as seg:
            native = [float(seg[f"inject_{k}"]) for k in NATIVE_KEYS]
        pfn = L.gbonfly.params_from_native(*native, prep.T)
        period = float(arr["envelope_period"])
        C = L.envelope.orthonormal_coefficients(prep.T, L.config.N_HARM, period)
        half = [8.0 * float(np.sum(np.abs(C[:, i]))) for i in range(C.shape[1])]
        freqs = (prep.bin_offset + np.arange(prep.n_bins)) * df
        inst, gal = L.prepare.psd_components(freqs, np.asarray(arr["fg_params"], float))
        S_ref = np.asarray(arr["S_ref"], float)[:, None]
        psd[f"{case}__freqs"], psd[f"{case}__S_inst"], psd[f"{case}__S_fg"] = freqs, inst, gal
        cases[case] = {
            "T": prep.T, "N": prep.N, "NT": prep.NT, "DT": prep.DT, "df": df,
            "bin_offset": prep.bin_offset, "n_bins": prep.n_bins,
            "freq": [prep.freq.start, prep.freq.stop], "store": [int(arr["store_offset"]), int(arr["store_n"])],
            "bands": [int(arr["first_band"]), int(arr["first_band"]) + int(arr["n_bands"]) - 1],
            "sigma0": prep.sigma0, "S_ref": arr["S_ref"], "matrix_cut": arr["matrix_cut"],
            "cut_columns": list(L.matrix.cut_columns(prep)), "periodic": block is not None,
            "exact_block": block, "fg_params": arr["fg_params"], "mu_fit": prep.mu_fit,
            "mu_freq": prep.mu_freq, "template": template, "describe": describe, "counts": counts,
            "snr_full": {"recomputed": snr, "report": report["snr_full"],
                         "equal": snr == report["snr_full"]},
            "prior": {"lo": prep.prior_lo, "hi": prep.prior_hi, "source_band_bins": src,
                      "equal_ucb_prior": bool(np.array_equal(prior.lo, prep.prior_lo)
                                              and np.array_equal(prior.hi, prep.prior_hi))},
            "inject": {"inject_glass": prep.inject, "native": dict(zip(NATIVE_KEYS, native)),
                       "params_from_native": pfn,
                       "equal": bool(np.array_equal(pfn, prep.inject))},
            "envelope": {"period": period, "C": C, "physical_A": C.T @ prep.mu_fit[0],
                         "physical_E": C.T @ prep.mu_fit[1], "half_widths": half,
                         "prior_half_width": min(half),
                         "basis_L_equals_ghat_grid": bool(np.array_equal(
                             L.envelope.ghat_grid(prep.T), prep.basis_L))},
            "psd_check": {"max_rel_sI": float(np.max(np.abs(inst / S_ref - prep.sI) / prep.sI)),
                          "max_rel_sG": float(np.max(np.abs(gal / S_ref - prep.sG) / prep.sG))},
        }
        print(f"[products] {case}: window {template['window']['sum_h2']:.10f} / "
              f"{template['window']['re_zh']:.10f}; snr {snr!r}; counts {counts}", flush=True)
    constants = {"TEMPLATE_BW": L.config.TEMPLATE_BW, "MU_BOX": L.config.MU_BOX, "C_F": L.config.C_F,
                 "C_T": L.config.C_T, "GHAT_L": L.config.GHAT_L, "N_HARM": L.config.N_HARM,
                 "N_BANDS": L.config.N_BANDS, "DT": L.config.DT,
                 "ENVELOPE_PERIOD": L.config.ENVELOPE_PERIOD, "ANALYSIS_NT": L.config.ANALYSIS_NT,
                 "CUT": {k: list(v) for k, v in L.matrix.CUT.items()},
                 "PAD_COLUMNS": L.matrix.PAD_COLUMNS, "PAD_EDGE_TOL": L.matrix.PAD_EDGE_TOL,
                 "GLASS_PARAMS": list(L.config.GLASS_PARAMS), "ENV_NAMES": list(L.config.ENV_NAMES),
                 "PERIODIC": {str(k): list(v) for k, v in L.config.PERIODIC.items()},
                 "NOISE_MOVE_FRACTION": L.config.NOISE_MOVE_FRACTION,
                 "gbonfly_version": L.gbonfly.__version__}
    pay.write_json("galactic_binary/products.json",
                   {"schema": 1, "models": models, "constants": constants, "cases": cases,
                    "build": gbw_common.build_info()})
    pay.write_npz("galactic_binary/psd_window.npz", psd)
    pay.close()
