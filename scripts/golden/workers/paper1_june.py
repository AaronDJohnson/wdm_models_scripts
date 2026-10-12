"""Lineage-D golden worker: Paper I oracle values from the June ``wdm`` package (read-only).

    $PY -P scripts/golden/workers/paper1_june.py OUT_DIR --workspace DIR

Run through ``lineage.run_worker("D", "paper1_june", ...)`` (PYTHONPATH = workspace root and
``wdm_paper_1/notes``). Section A's worker contract runs first (``_common.start(("D",))``) and
``_common.check_wdm(required=True)`` after the imports: ``wdm`` must resolve to
``<workspace>/wdm`` (the June package, whose
``phi_centered(Nt, Nf, dt)`` takes Nt FIRST and is Hz-normalised, amplitude 1/sqrt(dF)).

Inputs, recorded verbatim in the outputs:

* PSD grids (Nf, Nt, dt) = (8, 16, 1.0), (16, 32, 0.5); one-sided S on ``rfftfreq(N, dt)``:
  flat 2.3; linear 1 + 3|f|/(0.5/dt); steep 1/(|f| + 0.02); quadratic 1 + 40(|f| - 0.25)^2;
  random = ``np.random.default_rng(0).uniform(0.5, 5.0, N//2 + 1)``, a FRESH generator per grid
  (not the shared stream of ``paper_algorithms._test``);
* data grids (8,16,1), (16,32,.5), (8,32,1.7), (32,64,1), (4,8,1); x = one shared
  ``default_rng(0)``, ``standard_normal(N)`` per grid in that order (as ``_test``).

Outputs (``golden/wdm/``):

* ``alg4_reference.npz``: per (grid, psd): ``S_in``; Lambda from the literal Alg 3
  (``paper_algorithms.lambda_from_psd``; full blocks at 8x16, fingerprints at 16x32 in the JSON);
  S recovered by literal Alg 4 (``psd_from_lambda``), ``stationary.lambda_to_psd_B``,
  ``stationary.lambda_to_psd_exact``, ``psd_backward_offdiag.lambda_to_psd_offdiag``,
  ``inverse_psd_new_section.inverse_psd_new`` (all fed the literal Lambda) and the June
  diagonal-only ``stationary.lambda_to_psd``; the June window ``phi_centered(Nt, Nf, dt)``;
* ``alg4_reference.json``: definitions, source sha256, relative errors, Lambda fingerprints;
* ``paper_june_alg12.npz``: literal Alg 1 ``forward`` / Alg 2 ``inverse`` / package
  ``transform.forward_time`` on the data grids, and the Alg 2 leak vector at (8, 16, 0.5)
  (unit inactive DC pixel ``w[0, 1] = 1``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

PSD_GRIDS = ((8, 16, 1.0), (16, 32, 0.5))
DATA_GRIDS = ((8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (32, 64, 1.0), (4, 8, 1.0))
PSD_NAMES = ("flat", "linear", "steep", "quadratic", "random")
FULL_LAMBDA_GRIDS = ((8, 16, 1.0),)
PSD_DEFINITIONS = {"flat": "2.3", "linear": "1 + 3|f|/(0.5/dt)", "steep": "1/(|f| + 0.02)",
                   "quadratic": "1 + 40(|f| - 0.25)^2",
                   "random": "np.random.default_rng(0).uniform(0.5, 5.0, N//2 + 1), fresh per grid"}


def gkey(Nf, Nt, dt) -> str:
    return f"g{Nf}x{Nt}dt{str(dt).replace('.', 'p')}"


def psd_values(name: str, Nf: int, Nt: int, dt: float) -> np.ndarray:
    N = Nf * Nt
    f = np.fft.rfftfreq(N, dt)
    if name == "flat":
        return np.full_like(f, 2.3)
    if name == "linear":
        return 1.0 + 3.0 * np.abs(f) / (0.5 / dt)
    if name == "steep":
        return 1.0 / (np.abs(f) + 0.02)
    if name == "quadratic":
        return 1.0 + 40.0 * (np.abs(f) - 0.25) ** 2
    if name == "random":
        return np.random.default_rng(0).uniform(0.5, 5.0, size=N // 2 + 1)
    raise ValueError(name)


def relerr(S, ref, sl) -> float:
    return float(np.max(np.abs(S[sl] - ref[sl])) / np.max(np.abs(ref[sl])))


def _load_common():
    """Section A's worker contract (``workers/_common.py``), loaded by file path (A3)."""
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("golden_worker_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    C = _load_common()
    out_dir, rest = C.start(("D",))          # out_dir under $TMPDIR, -P, no bytecode, lineage D
    ap = argparse.ArgumentParser(prog="paper1_june.py OUT_DIR")
    ap.add_argument("--workspace", type=Path, required=True)
    a = ap.parse_args(rest)
    sys.path.append(str(Path(__file__).resolve().parent))
    import gbw_common

    gbw_common.require_safe_interpreter()
    gbw_common.install_write_guard(out_dir, tempfile.gettempdir())
    ws = a.workspace.resolve()
    import wdm
    from wdm import paper_algorithms as pa
    from wdm import stationary, transform
    from wdm.filter import phi_centered
    import inverse_psd_new_section
    import psd_backward_offdiag

    C.check_wdm(required=True)               # wdm resolves under lineage D's <ROOT>/wdm
    gbw_common.check_origin(wdm, ws / "wdm")
    gbw_common.check_origin(psd_backward_offdiag, ws / "wdm_paper_1" / "notes")
    gbw_common.check_origin(inverse_psd_new_section, ws / "wdm_paper_1" / "notes")
    pay = gbw_common.Payload(out_dir, lineage="D", worker="paper1_june", subcommand="alg4",
                             args=rest, base=ws)
    sources = {}
    for rel in ("wdm/paper_algorithms.py", "wdm/stationary.py", "wdm/filter.py", "wdm/transform.py",
                "wdm_paper_1/notes/psd_backward_offdiag.py",
                "wdm_paper_1/notes/inverse_psd_new_section.py"):
        sources[rel] = pay.add_input(ws / rel)

    arrays, errors, lam_fp, lam_vs_fast = {}, {}, {}, {}
    for Nf, Nt, dt in PSD_GRIDS:
        g = gkey(Nf, Nt, dt)
        N = Nf * Nt
        sl = slice(Nt // 2, N // 2 - Nt // 2)
        arrays[f"{g}__phi_june"] = phi_centered(Nt, Nf, dt)
        for name in PSD_NAMES:
            S = psd_values(name, Nf, Nt, dt)
            d, o, na = pa.lambda_from_psd(S, Nf, Nt, dt)
            df, of, _ = stationary.build_lambda_blocks_full_active(Nf, Nt, dt, S)
            lam_vs_fast[f"{g}__{name}"] = {
                "diag": max(float(np.max(np.abs(d[m] - df[m]))) for m in range(Nf + 1)),
                "offd": max(float(np.max(np.abs(o[m] - of[m]))) for m in range(Nf))}
            outs = {"alg4": pa.psd_from_lambda(d, o, na, Nf, Nt, dt),
                    "B": stationary.lambda_to_psd_B(d, o, na, Nf, Nt, dt),
                    "exact": stationary.lambda_to_psd_exact(d, o, na, Nf, Nt, dt),
                    "offdiag": psd_backward_offdiag.lambda_to_psd_offdiag(d, o, na, Nf, Nt, dt),
                    "new_section": inverse_psd_new_section.inverse_psd_new(d, o, na, Nf, Nt, dt),
                    "diag_only": stationary.lambda_to_psd(d, o, na, Nf, Nt, dt)}
            key = f"{g}__{name}"
            arrays[f"{key}__S_in"] = S
            for k, v in outs.items():
                arrays[f"{key}__S_{k}"] = np.asarray(v, dtype=np.float64)
            errors[key] = {k: relerr(np.asarray(v), S, sl) for k, v in outs.items()}
            flat = np.concatenate([b.ravel() for b in d] + [b.ravel() for b in o])
            lam_fp[key] = gbw_common.fingerprint(flat)
            if (Nf, Nt, dt) in FULL_LAMBDA_GRIDS:
                for m, b in enumerate(d):
                    arrays[f"{key}__lam_diag_{m:02d}"] = b
                for m, b in enumerate(o):
                    arrays[f"{key}__lam_offd_{m:02d}"] = b
        for m, n_act in enumerate(na):
            arrays[f"{g}__n_active_{m:02d}"] = np.asarray(n_act, dtype=np.int64)

    alg12 = {}
    rng = np.random.default_rng(0)
    for Nf, Nt, dt in DATA_GRIDS:
        g = gkey(Nf, Nt, dt)
        x = rng.standard_normal(Nf * Nt)
        w = pa.forward(x, Nf, Nt, dt)
        alg12[f"{g}__x"] = x
        alg12[f"{g}__w_alg1"] = w
        alg12[f"{g}__x_alg2"] = pa.inverse(w, Nf, Nt, dt)
        alg12[f"{g}__w_package"] = transform.forward_time(x, Nf, Nt, dt)
    leak = np.zeros((9, 16))
    leak[0, 1] = 1.0
    alg12["leak__w"] = leak
    alg12["leak__x_alg2"] = pa.inverse(leak, 8, 16, 0.5)

    meta = {"schema": 1, "lineage": "D", "wdm_file": wdm.__file__,
            "phi_centered_signature": "phi_centered(Nt, Nf, dt, d=4); Hz-normalised, amp = 1/sqrt(dF)",
            "psd_grids": [list(g) for g in PSD_GRIDS], "data_grids": [list(g) for g in DATA_GRIDS],
            "psd_definitions": PSD_DEFINITIONS, "interior_slice": "slice(Nt // 2, N // 2 - Nt // 2)",
            "lambda_fed_to_every_alg4_oracle": "paper_algorithms.lambda_from_psd (literal Alg 3)",
            "full_lambda_grids": [gkey(*g) for g in FULL_LAMBDA_GRIDS],
            "relerr_interior": errors, "lambda_literal_vs_fast": lam_vs_fast,
            "lambda_fingerprints": lam_fp, "sources_sha256": sources,
            "leak_max_abs_x": float(np.max(np.abs(alg12["leak__x_alg2"]))),
            "build": gbw_common.build_info()}
    pay.write_npz("wdm/alg4_reference.npz", arrays)
    pay.write_npz("wdm/paper_june_alg12.npz", alg12)
    pay.write_json("wdm/alg4_reference.json", meta)
    pay.close()
    worst = max(v for e in errors.values() for k, v in e.items() if k != "diag_only")
    print(f"[paper1_june] {len(arrays)} arrays; worst exact-oracle interior relerr {worst:.2e}; "
          f"leak {meta['leak_max_abs_x']!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
