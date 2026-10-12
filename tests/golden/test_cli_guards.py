"""The harness scripts import cleanly without PYTHONSAFEPATH; only their ``main()`` refuses.

Spec §1 guard 5 (no cwd injection) belongs to the command line: a module-level guard turned a
plain ``import`` (pytest collection) into ``SystemExit`` and made the bare root ``pytest`` end
with INTERNALERROR. Every script keeps the guard at the top of ``main()`` (as Section A always
did), imports its siblings as ``scripts.golden.*`` and never edits ``sys.path``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
#: every scripts/golden module with a command-line ``main()``.
SECTION_A = ("capture", "measure_tolerances", "pin_external", "sinusoid_goldens")
SECTION_B = ("measure_tolerances_gb", "summarise_chains", "capture_payload", "capture_gb",
             "capture_paper1", "ship_gb_data")

PROBE = r"""
import importlib, json, sys
before = list(sys.path)
exits = {}
for name in sys.argv[1:]:
    mod = importlib.import_module(f"scripts.golden.{name}")
    try:
        mod.main(["--help"])          # guard first: no argparse, nothing read or written
        exits[name] = "returned"
    except SystemExit as exc:
        exits[name] = exc.code if isinstance(exc.code, (int, type(None))) else str(exc.code)
print(json.dumps({"safe_path": bool(sys.flags.safe_path), "path_unchanged": sys.path == before,
                  "exits": exits}))
"""


def _probe(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    env.update(PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", PROBE, *SECTION_A, *SECTION_B], env=env,
                          cwd=str(tmp_path), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_scripts_import_without_safe_path_and_main_refuses(tmp_path):
    info = _probe(tmp_path)
    assert info["safe_path"] is False
    assert info["path_unchanged"] is True                       # no scripts/golden on sys.path
    for name in SECTION_A + SECTION_B:
        assert "python -P" in str(info["exits"][name]), (name, info["exits"][name])


@pytest.mark.parametrize("name", SECTION_B)
def test_section_b_scripts_do_not_touch_sys_path(name):
    source = (REL / "scripts" / "golden" / f"{name}.py").read_text()
    assert "sys.path" not in source, name
    assert "\nimport manifest_merge" not in source and "\nimport capture_payload" not in source, name
