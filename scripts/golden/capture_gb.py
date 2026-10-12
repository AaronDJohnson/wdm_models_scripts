"""Capture the galactic-binary goldens (lineage C) with the absolute paths filled in.

    $PY -m scripts.golden.capture_gb rows --case gb_3mo [--tier fast] [--backends numpy,jax] [--models M,..]
    $PY -m scripts.golden.capture_gb products
    $PY -m scripts.golden.capture_gb intermediates --case gb_3mo
    $PY -m scripts.golden.capture_gb stress --case gb_3mo [--tier fast] [--backends numpy,jax]

Everything after the subcommand that this script does not know is passed to the worker
(``workers/gb_legacy.py`` -> ``workers/gbw_<subcommand>.py``); ``--workspace`` and
``--data-dir`` are always supplied as absolute paths (the worker runs in $TMPDIR/golden_run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.golden import capture_payload
from scripts.golden.paths import DATA, ROOT  # (Section A)

SUBCOMMANDS = ("rows", "products", "intermediates", "stress")


def main(argv=None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subcommand", choices=SUBCOMMANDS)
    ap.add_argument("--workspace", type=Path, default=ROOT)   # the root lineage C imports from
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--replace", action="store_true")
    a, rest = ap.parse_known_args(argv)
    worker_args = [a.subcommand, "--workspace", str(a.workspace.resolve()),
                   "--data-dir", str(DATA.resolve()), *rest]
    files = capture_payload.capture("C", "gb_legacy", worker_args, timeout=a.timeout,
                                    replace=a.replace)
    print(f"captured {len(files)} galactic-binary golden file(s): {a.subcommand} {' '.join(rest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
