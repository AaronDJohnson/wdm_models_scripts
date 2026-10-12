"""Capture the Paper I lineage-D oracle values (June ``wdm``) into golden/wdm/.

    $PY -m scripts.golden.capture_paper1 [--workspace DIR] [--replace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.golden import capture_payload
from scripts.golden.paths import ROOT  # (Section A)


def main(argv=None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", type=Path, default=ROOT)   # the root lineage D imports from
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    files = capture_payload.capture("D", "paper1_june", ["--workspace", str(a.workspace.resolve())],
                                    timeout=600.0, replace=a.replace)
    print(f"captured {len(files)} Paper I golden file(s): {sorted(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
