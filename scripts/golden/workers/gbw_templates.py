"""F-C (S6): the gbonfly store templates at given source rows, for the release's test-only
gbonfly adapter (spec section 7.6, S6: "templates computed on demand in the lineage-C
subprocess and streamed in; nothing large is stored in golden/").

    gb_legacy.py OUT_DIR templates --jobs JOBS.json --workspace DIR --data-dir DIR

``JOBS.json``: ``[{"tag", "case", "points"}]``, ``points`` an ``.npz`` holding ``p8`` (n, 8),
the GLASS source rows exactly as the likelihood receives them (unwrapped).  Per job the
worker writes ``<tag>.npz`` with ``h`` (n, 2, store_n) complex: ``matrix.store_template(prep,
p8, np)`` -- gbonfly ``template_on_window`` on the store window, BW = ``TEMPLATE_BW``,
divided by ``DT * sigma0`` -- the template every lineage-C matrix model reads (and, on the
window bins, the window template ``Prepared.template`` of whittle_raw/exact, bit for bit).
``templates.json`` records the runtime and, per job, the case, row count and the prepared
file's sha256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c


def _parse(argv):
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR templates")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    return ap.parse_args(argv)


def run(out_dir, argv) -> None:
    a = _parse(argv)
    jobs = json.loads(a.jobs.read_text())
    L = gbw_lineage_c.load(a.workspace)
    out_dir = Path(out_dir)
    preps, info = {}, {}
    for job in jobs:
        tag, case = job["tag"], job["case"]
        if case not in gbw_lineage_c.CASES:
            raise SystemExit(f"job {tag}: unknown case {case!r}")
        if case not in preps:
            path = a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{case}.npz"
            preps[case] = (L.likelihoods.Prepared(path), gbw_common.sha256_file(path))
        prep, sha = preps[case]
        with np.load(job["points"], allow_pickle=False) as z:
            P8 = np.atleast_2d(np.asarray(z["p8"], dtype=np.float64))
        if P8.shape[1] != 8:
            raise SystemExit(f"job {tag}: p8 has {P8.shape[1]} columns, expected 8")
        h = np.stack([np.asarray(L.matrix.store_template(prep, p, np)) for p in P8])
        np.savez(out_dir / f"{tag}.npz", h=h, p8=P8)
        info[tag] = {"case": case, "n": int(len(P8)), "store_n": int(h.shape[-1]), "prepared_sha256": sha,
                     "evaluator": "matrix.store_template(Prepared(path), p8, np)  (gbonfly, fourier)"}
        print(f"[templates] {tag}: {case} n={len(P8)}", flush=True)
    (out_dir / "templates.json").write_text(json.dumps(
        {"lineage": "C", "runtime": gbw_common.build_info(), "jobs": info}, indent=1, sort_keys=True) + "\n")
