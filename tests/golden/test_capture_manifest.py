"""capture.write_manifest never silently re-approves a golden (final review I1; spec §7.1).

``--manifest-only`` may only ADD new files; a ``--family F`` capture may re-pin only files under
``golden/F/``; every other changed or removed entry is refused before anything is written,
unless ``--replace``. ``build`` is kept unless ``--replace``. All on a scratch copy of golden/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.golden import capture, measure_tolerances
from scripts.golden.goldens_io import golden_files
from scripts.golden.paths import GOLDEN, REL
from scripts.golden.workers._common import runtime_fingerprint

pytestmark = pytest.mark.golden

GB_FILE = "galactic_binary/stress/gb_3mo/exact.fast.json"
SIN_FILE = "sinusoid/tolerances_measured.json"
SENTINEL_BUILD = {"python": "sentinel"}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    real = (GOLDEN / "MANIFEST.json").read_bytes()
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN, golden, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    m = json.loads((golden / "MANIFEST.json").read_text())
    m["build"] = SENTINEL_BUILD                       # proves `build` is kept, not re-stamped
    (golden / "MANIFEST.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    monkeypatch.setattr(capture, "GOLDEN", golden)
    yield golden
    assert (GOLDEN / "MANIFEST.json").read_bytes() == real      # the real manifest is never touched


def _manifest(golden: Path) -> dict:
    return json.loads((golden / "MANIFEST.json").read_text())


def _tamper_gb(golden: Path) -> None:
    """The review's reproduction: shift lnl_ref, lnl_cache and lnl_jax by +0.5 nat."""
    path = golden / GB_FILE
    r = json.loads(path.read_text())
    for p in r["points"]:
        for key in ("lnl_ref", "lnl_cache", "lnl_jax"):
            if isinstance(p.get(key), float):
                p[key] += 0.5
    path.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")


def test_manifest_only_cli_refuses_a_changed_gb_golden(scratch):
    _tamper_gb(scratch)
    before = (scratch / "MANIFEST.json").read_bytes()
    code = ("import sys; from pathlib import Path; from scripts.golden import capture; "
            "capture.GOLDEN = Path(sys.argv[1]); sys.exit(capture.main(['--manifest-only']))")
    env = dict(os.environ, PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-P", "-c", code, str(scratch)], env=env, cwd=str(scratch.parent),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert GB_FILE in proc.stderr and "--replace" in proc.stderr, proc.stderr
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_manifest_only_refuses_a_removed_golden(scratch):
    (scratch / GB_FILE).unlink()
    before = (scratch / "MANIFEST.json").read_bytes()
    with pytest.raises(SystemExit, match=r"--replace") as exc:
        capture.write_manifest()
    assert GB_FILE in str(exc.value)
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_manifest_only_refuses_a_changed_sinusoid_golden(scratch):
    """No family given: even the sinusoid files may not be re-pinned."""
    (scratch / SIN_FILE).write_text((scratch / SIN_FILE).read_text() + " ")
    before = (scratch / "MANIFEST.json").read_bytes()
    with pytest.raises(SystemExit, match=SIN_FILE):
        capture.write_manifest()
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_manifest_only_adds_new_files_and_changes_nothing_else(scratch):
    before = _manifest(scratch)
    (scratch / "galactic_binary" / "new_golden.json").write_text("{}\n")
    capture.write_manifest()
    after = _manifest(scratch)
    added = set(after["files"]) - set(before["files"])
    assert added == {"galactic_binary/new_golden.json"}
    assert {k: v for k, v in after["files"].items() if k not in added} == before["files"]
    assert after["build"] == SENTINEL_BUILD
    assert {k: after[k] for k in after if k != "files"} == {k: before[k] for k in before if k != "files"}


def test_unchanged_manifest_only_run_rewrites_identical_bytes(scratch):
    before = (scratch / "MANIFEST.json").read_bytes()
    capture.write_manifest()
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_family_capture_repins_only_its_own_files(scratch):
    (scratch / SIN_FILE).write_text((scratch / SIN_FILE).read_text() + " ")
    old = _manifest(scratch)
    sources = old["sources"]["sinusoid"]
    tolerances = old["tolerances"]["sinusoid"]
    capture.write_manifest(family="sinusoid", sources=sources, tolerances=tolerances)
    after = _manifest(scratch)
    assert after["files"][SIN_FILE] == golden_files(scratch)[SIN_FILE] != old["files"][SIN_FILE]
    assert after["build"] == SENTINEL_BUILD
    # a sinusoid capture may not re-approve a galactic-binary golden
    _tamper_gb(scratch)
    before = (scratch / "MANIFEST.json").read_bytes()
    with pytest.raises(SystemExit, match=GB_FILE):
        capture.write_manifest(family="sinusoid", sources=sources, tolerances=tolerances)
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_family_capture_refuses_before_the_family_runs(scratch, monkeypatch):
    calls = []
    monkeypatch.setitem(capture.FAMILIES, "sinusoid", (lambda tier: calls.append(tier) or {},
                                                       lambda: {"file": SIN_FILE, "status": "x"}))
    monkeypatch.setattr(capture.fth, "verify", lambda *a, **k: [])
    _tamper_gb(scratch)
    before = (scratch / "MANIFEST.json").read_bytes()
    with pytest.raises(SystemExit, match=GB_FILE):
        capture.capture("sinusoid", "fast")
    assert calls == []
    assert (scratch / "MANIFEST.json").read_bytes() == before


def test_family_capture_keeps_the_approved_tolerance_summary(scratch, monkeypatch):
    """A sinusoid re-capture writes the same MANIFEST tolerance summary as the measurement script's
    own command line (status, approval date, decision labels and the approved gates; Task F4), not
    just ``{"file", "status"}``."""
    old = _manifest(scratch)
    report = json.loads((scratch / SIN_FILE).read_text())
    monkeypatch.setitem(capture.FAMILIES, "sinusoid", (lambda tier: old["sources"]["sinusoid"], lambda: report))
    monkeypatch.setattr(capture.fth, "verify", lambda *a, **k: [])
    monkeypatch.setattr(capture, "guard_state", lambda: ())
    capture.capture("sinusoid", "fast")
    after = _manifest(scratch)
    assert after["tolerances"]["sinusoid"] == measure_tolerances.manifest_summary(report)
    assert after["tolerances"]["sinusoid"] == old["tolerances"]["sinusoid"]
    assert after["tolerances"]["sinusoid"]["approved"] == report["approved"]


def test_replace_reapproves_everything_and_restamps_build(scratch):
    _tamper_gb(scratch)
    (scratch / SIN_FILE).unlink()
    capture.write_manifest(replace=True)
    after = _manifest(scratch)
    assert after["files"] == golden_files(scratch)
    assert after["build"] == runtime_fingerprint()
