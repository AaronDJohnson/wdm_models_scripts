"""Lineage-C worker: the frozen ``prepare.prepare`` + ``prepare.add_matrix`` on the SHIPPED
segments and stores, written into ``<out_dir>`` (``<case>.npz``, ``<case>_report.json``).

    lineage.run_worker("C", "gb_legacy", "prepared", "--workspace", W, "--data-dir", D, "--case", C ...)

The legacy module reads its inputs from ``config.SEGMENTS`` / ``config.STORES``; this worker
points those entries (in its own process only) at ``<data-dir>/galactic_binary/segments`` --
byte-identical copies of the frozen files (``data/MANIFEST.json``) -- and writes nothing else.
``scripts/galactic_binary/prepare.py --from-shipped --engine legacy`` compares the result with
the shipped prepared files (measured 2026-10-13: every key ``array_equal`` except
``matrix_store``, a path, and ``nt_star``, which the shipped files carry from an earlier
``calc_Nt``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gbw_lineage_c


def run(out_dir, argv) -> None:
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR prepared")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--case", nargs="+", required=True, choices=gbw_lineage_c.CASES)
    a = ap.parse_args(argv)
    L = gbw_lineage_c.load(a.workspace)
    seg = a.data_dir.resolve() / "galactic_binary" / "segments"
    seconds = {}
    for case in a.case:
        months = int(case[3:-2])
        L.config.SEGMENTS[case] = seg / f"segment_snr7_{months}mo_3bands.npz"
        L.config.STORES[case] = seg / f"store_snr7_{months}mo_3bands_pm3.npz"
        t0 = time.perf_counter()
        L.prepare.prepare(case, out_dir=Path(out_dir))
        L.prepare.add_matrix(case, out_dir=Path(out_dir))
        seconds[case] = time.perf_counter() - t0
    (Path(out_dir) / "prepared.json").write_text(json.dumps({"cases": a.case, "seconds": seconds}) + "\n")
