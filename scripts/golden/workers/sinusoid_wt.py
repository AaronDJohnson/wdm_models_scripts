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


def main() -> None:
    out_dir, argv = C.start(("A",))
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"command must be one of {sorted(COMMANDS)}")
    COMMANDS[argv[0]](out_dir, argv[1:])


if __name__ == "__main__":
    main()
