"""Capture the Paper I lineage-D oracle values (June ``wdm``) into golden/wdm/.

    $PY -P scripts/golden/capture_paper1.py [--workspace DIR] [--replace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not sys.flags.safe_path:
    raise SystemExit("run with python -P (PYTHONSAFEPATH)")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_payload  # noqa: E402  (also puts $REL on sys.path)
from scripts.golden.paths import ROOT  # noqa: E402  (Section A)


def main(argv=None) -> int:
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
