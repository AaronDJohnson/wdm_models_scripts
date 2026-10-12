"""B2: the section-B worker helpers (write guard, fingerprints, row tiers, npz determinism)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden

REL = Path(__file__).resolve().parents[2]
WORKERS = REL / "scripts" / "golden" / "workers"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gbw_common = _load("gbw_common", WORKERS / "gbw_common.py")

GUARD_SCRIPT = r"""
import os, sys
sys.path.append(sys.argv[1])
import gbw_common
allowed, outside = sys.argv[2], sys.argv[3]
gbw_common.install_write_guard(allowed)
open(os.path.join(allowed, "ok.txt"), "w").write("fine")
os.mkdir(os.path.join(allowed, "sub"))
results = []
for attempt in (lambda: open(os.path.join(outside, "bad.txt"), "w"),
                lambda: open(os.path.join(outside, "bad.txt"), "ab"),
                lambda: os.open(os.path.join(outside, "bad2.txt"), os.O_WRONLY | os.O_CREAT),
                lambda: os.mkdir(os.path.join(outside, "baddir")),
                lambda: os.rename(os.path.join(allowed, "ok.txt"), os.path.join(outside, "moved.txt"))):
    try:
        attempt()
        results.append("allowed")
    except PermissionError:
        results.append("blocked")
print(",".join(results))
print(open(os.path.join(outside, "existing.txt")).read())
"""


def test_write_guard_blocks_outside_and_allows_inside(tmp_path):
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "existing.txt").write_text("read me")
    proc = subprocess.run([sys.executable, "-P", "-c", GUARD_SCRIPT, str(WORKERS), str(allowed),
                           str(outside)], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["blocked,blocked,blocked,blocked,blocked", "read me"]
    assert sorted(p.name for p in outside.iterdir()) == ["existing.txt"]
    assert (allowed / "ok.txt").read_text() == "fine" and (allowed / "sub").is_dir()


def test_select_rows_tiers():
    assert gbw_common.select_rows(1, "fast") == [0]
    assert gbw_common.select_rows(10, "fast") == [0, 5, 9]
    full = gbw_common.select_rows(309500, "full")
    assert len(full) == 32 and {0, 154750, 309499} <= set(full) and full == sorted(full)
    assert full == gbw_common.select_rows(309500, "full")          # deterministic
    assert gbw_common.select_rows(20, "full") == list(range(20))  # pool smaller than 29
    with pytest.raises(ValueError):
        gbw_common.select_rows(10, "medium")


def test_fingerprint_is_deterministic_and_complete():
    a = np.random.default_rng(1).standard_normal((40, 30))
    f1, f2 = gbw_common.fingerprint(a), gbw_common.fingerprint(a.copy())
    assert f1 == f2 and f1["shape"] == [40, 30] and len(f1["idx"]) == 50
    assert f1["values"] == a.reshape(-1)[f1["idx"]].tolist()
    assert f1["fro"] == float(np.linalg.norm(a.reshape(-1)))
    c = a + 1j * a[::-1]
    fc = gbw_common.fingerprint(c)
    assert fc["values_im"] == c.reshape(-1)[fc["idx"]].imag.tolist()
    b = a.copy()
    b[3, 4] = np.nextafter(b[3, 4], 1.0)
    assert gbw_common.fingerprint(b)["sha256"] != f1["sha256"]


def test_write_npz_is_byte_deterministic(tmp_path):
    arrays = {"b": np.arange(5.0), "a": np.eye(3) + 0.5j}
    gbw_common.write_npz(tmp_path / "x.npz", arrays)
    first = (tmp_path / "x.npz").read_bytes()
    gbw_common.write_npz(tmp_path / "x.npz", arrays)
    assert (tmp_path / "x.npz").read_bytes() == first
    with np.load(tmp_path / "x.npz") as z:
        assert sorted(z.files) == ["a", "b"] and np.array_equal(z["a"], arrays["a"])


def test_payload_index_lists_every_file(tmp_path):
    pay = gbw_common.Payload(tmp_path, lineage="C", worker="gb_legacy", subcommand="t", args=["x"])
    pay.write_json("galactic_binary/t.json", {"v": np.float64(1.5), "w": np.arange(2)})
    pay.write_npz("galactic_binary/t.npz", {"a": np.ones(3)})
    index = json.loads(pay.close().read_text())
    assert sorted(index["files"]) == ["galactic_binary/t.json", "galactic_binary/t.npz"]
    assert json.loads((tmp_path / "payload/galactic_binary/t.json").read_text()) == {"v": 1.5, "w": [0, 1]}
    with pytest.raises(ValueError):
        pay.write_json("../escape.json", {})
