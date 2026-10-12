"""Lineage-C golden worker: the frozen galactic-binary code, evaluated read-only.

    $PY -P scripts/golden/workers/gb_legacy.py OUT_DIR SUBCOMMAND [options]

Run through ``lineage.run_worker("C", "gb_legacy", SUBCOMMAND, ...)``. ``SUBCOMMAND`` names
a module ``gbw_<SUBCOMMAND>.py`` in this directory exposing ``run(out_dir, argv)``
(rows, products, intermediates, stress; ``evaluate`` from Task B12). Section A's worker
contract runs first (``_common.start(("C",))``: out_dir under $TMPDIR, ``-P``, no bytecode,
``WDM_GOLDEN_LINEAGE == "C"``) and last (``_common.check_wdm()``: lineage C never imports a
``wdm`` package). Every Python-level write outside ``OUT_DIR`` and $TMPDIR raises
(``gbw_common.install_write_guard``).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import tempfile
from pathlib import Path


def _load_common():
    """Section A's worker contract (``workers/_common.py``), loaded by file path (A3)."""
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("golden_worker_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    C = _load_common()
    out_dir, rest = C.start(("C",))          # out_dir under $TMPDIR, -P, no bytecode, lineage C
    if not rest or not re.fullmatch(r"[a-z]+", rest[0]):
        raise SystemExit("usage: gb_legacy.py OUT_DIR SUBCOMMAND [options]")
    sub = rest[0]
    sys.path.append(str(Path(__file__).resolve().parent))
    import gbw_common

    gbw_common.require_safe_interpreter()
    gbw_common.install_write_guard(out_dir, tempfile.gettempdir())
    module = importlib.import_module(f"gbw_{sub}")
    module.run(out_dir, rest[1:])
    C.check_wdm()                            # lineage C must never import a wdm package
    return 0


if __name__ == "__main__":
    sys.exit(main())
