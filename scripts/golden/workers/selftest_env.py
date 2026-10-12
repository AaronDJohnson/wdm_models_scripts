"""Harness self-test worker: report how a lineage subprocess sees its world (Task A3).

    $PY -P selftest_env.py <out_dir> [--import MODULE]... [--fail]
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


def _load_common():
    path = Path(__file__).resolve().with_name("_common.py")
    spec = importlib.util.spec_from_file_location("golden_worker_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load_common()


def main() -> None:
    out_dir, argv = C.start(("A", "B", "C", "D", "R"))
    if "--fail" in argv:
        raise SystemExit("selftest asked to fail")
    imported = {}
    rest = list(argv)
    while "--import" in rest:
        i = rest.index("--import")
        name = rest[i + 1]
        del rest[i:i + 2]
        imported[name] = str(Path(importlib.import_module(name).__file__).resolve())
    report = {
        "cwd": str(Path.cwd().resolve()),
        "sys_path": list(sys.path),
        "executable": sys.executable,
        "flags": {"safe_path": bool(sys.flags.safe_path),
                  "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                  "no_user_site": bool(sys.flags.no_user_site)},
        "env": {k: os.environ.get(k) for k in C.ENV_KEYS},
        "imported": imported,
        "wdm_file": C.check_wdm(),
        "jax_imported": "jax" in sys.modules,
    }
    C.write_json(out_dir / "selftest.json", report)


if __name__ == "__main__":
    main()
