"""Lineage-C worker (test-only): the frozen GB proposal Fisher and starting covariance.

    lineage.run_worker("C", "gb_legacy", "proposals", "--workspace", W, "--data-dir", D, "--case", C)

Writes ``<out_dir>/proposals.json``: ``gpu/proposals.SourceFisher`` (gbonfly templates) at the
injection and at ``inject + OFFSET``, and per model ``noise_fisher`` and
``initial_covariance``.  The release's ``galactic_binary.proposals`` with ``backend="fourier"``
is compared with it (``tests/golden/test_s5_gb_proposals_legacy.py``, legacy marker).
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

import gbw_common
import gbw_lineage_c

#: A point off the integer carrier bin (``f0 T`` = 7884.3 at 3 months) for the Fisher.
OFFSET = (0.3, 0.02, 0.05, 0.01, 0.03, 0.04, 0.05, 1e-6)


def run(out_dir, argv) -> None:
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR proposals")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", default="gb_3mo", choices=gbw_lineage_c.CASES)
    ap.add_argument("--models", nargs="+", default=["whittle_raw", "band_raw"])
    a = ap.parse_args(argv)
    L = gbw_lineage_c.load(a.workspace)
    prop = importlib.import_module("gpu.proposals")
    gbw_common.check_origin(prop, L.gb_dir / "gpu")
    prep = L.likelihoods.Prepared(a.data_dir.resolve() / "galactic_binary" / "prepared" / f"{a.case}.npz")
    fisher = prop.SourceFisher(prep)
    point = prep.inject + np.asarray(OFFSET)
    out = {"case": a.case, "point": point.tolist(), "fisher_inject": fisher(prep.inject).tolist(),
           "fisher_point": fisher(point).tolist(), "noise_fisher": {}, "initial_covariance": {}}
    for model in a.models:
        out["noise_fisher"][model] = prop.noise_fisher(prep, model).tolist()
        out["initial_covariance"][model] = prop.initial_covariance(prep, model, fisher).tolist()
    (Path(out_dir) / "proposals.json").write_text(json.dumps(out) + "\n")
