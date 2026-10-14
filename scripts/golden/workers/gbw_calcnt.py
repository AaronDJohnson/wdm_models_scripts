"""Lineage C, S5: the legacy ``prepare.calc_Nt`` (Sec. VII A's NT* on the fixed band
``config.NT_BAND``) for the release's port to be checked against.

    lineage.run_worker("C", "gb_legacy", "calcnt", "--workspace", ROOT, "--case", "gb_3mo", ...)

The shipped prepared files' ``nt_star`` predates ``calc_Nt`` (it is the earlier, tiling-dependent
estimate, 253.63 at 3 months), so the legacy function itself is the oracle.  It reads the frozen
segment (``config.SEGMENTS``) and fits ``mu`` with JAX; it writes nothing but this worker's
payload (``galactic_binary/calc_nt.json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gbw_common
import gbw_lineage_c


def run(out_dir, argv) -> None:
    ap = argparse.ArgumentParser(prog="gb_legacy.py OUT_DIR calcnt")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--case", action="append", choices=gbw_lineage_c.CASES, required=True)
    a = ap.parse_args(argv)
    L = gbw_lineage_c.load(a.workspace, jax=False)
    pay = gbw_common.Payload(out_dir, lineage="C", worker="gb_legacy", subcommand="calcnt",
                             args=argv, base=L.workspace)
    out = {}
    for case in a.case:
        pay.add_input(L.config.SEGMENTS[case])
        out[case] = L.prepare.calc_Nt(case)
        print(f"[calcnt] {case}: NT* {out[case]['nt_star']!r} -> {out[case]['nt_valid']}", flush=True)
    pay.write_json("galactic_binary/calc_nt.json", {"schema": 1, "cases": out,
                                                    "build": gbw_common.build_info()})
    pay.close()
