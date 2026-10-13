"""scripts/sinusoid/regenerate_data.py (spec sections 5.1 ``data.regenerate`` and 6): the
comparison rules and the CLI's guards.  The full-size draw (the dense 16384 x 16384 voice
factor, ~2 GB) never runs here: the CLI test replaces ``data.regenerate`` by the shipped
arrays, perturbed; the real run is the user's (approval-gated, see the module docstring)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.golden.paths import REL
from scripts.sinusoid import regenerate_data as R
from wdm_models.sinusoid import data

pytestmark = pytest.mark.golden
CASE = "manuscript_mid"


def _draw(scale: float = 0.0, *, h_ulp: bool = False, seed: int = 1, nan: bool = False) -> dict:
    """The shipped case with ``n`` scaled by ``1 + scale`` (and ``d = h + n``)."""
    s = data.load(CASE)
    n = s.n * (1.0 + scale)
    if nan:
        n = n.copy()
        n[7] = np.nan
    h = np.nextafter(s.h, np.inf) if h_ulp else s.h.copy()
    return {"d": h + n, "h": h, "n": n, "seed": seed}


def test_the_gate_is_one_part_in_1e12_with_h_bitwise():
    shipped = data.load(CASE)
    rep = R.compare(_draw(1e-13), shipped)
    assert rep["ok"] and rep["problems"] == [] and rep["h_bitwise"]
    assert 5e-14 < rep["n_rel"] <= 1e-13 * (1 + 1e-6) and rep["rtol"] == R.N_RTOL == 1e-12
    same = R.compare(_draw(0.0), shipped)
    assert same["ok"] and same["n_max_abs"] == same["d_max_abs"] == 0.0
    for draw, word in ((_draw(1e-11), "n differs"), (_draw(h_ulp=True), "h is not"),
                       (_draw(nan=True), "non-finite"), (_draw(seed=2), "seed")):
        bad = R.compare(draw, shipped)
        assert not bad["ok"] and any(word in p for p in bad["problems"]), (word, bad["problems"])
    short = {k: (v[:-1] if isinstance(v, np.ndarray) else v) for k, v in _draw().items()}
    assert any("shape" in p for p in R.compare(short, shipped)["problems"])


@pytest.mark.parametrize("scale,code", [(1e-13, 0), (1e-10, 1)])
def test_the_cli_writes_outside_the_tree_and_exits_on_the_gate(tmp_path, monkeypatch, capsys, scale, code):
    calls = []

    def fake_regenerate(case, *, seed):
        calls.append((case, seed))
        return _draw(scale, seed=seed)

    monkeypatch.setattr(R.data, "regenerate", fake_regenerate)
    assert R.main(["--case", CASE, "--out", str(tmp_path)]) == code
    assert calls == [(CASE, 1)]
    rep = json.loads((tmp_path / f"{CASE}_regenerate_report.json").read_text())
    assert rep["case"] == CASE and rep["ok"] is (code == 0) and rep["seconds"] >= 0.0
    with np.load(tmp_path / f"{CASE}_regenerated.npz") as z:
        assert sorted(z.files) == ["d", "h", "n", "seed"] and int(z["seed"]) == 1
    verdict = "PASS" if code == 0 else "FAIL"
    assert f"[regenerate_data] {verdict}: {CASE}" in capsys.readouterr().out


def _cli(*args, safe_path=True):
    drop = ("PYTHONPATH",) if safe_path else ("PYTHONPATH", "PYTHONSAFEPATH")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, *args], cwd=str(REL), env=env, capture_output=True, text=True)


def test_the_cli_refuses_the_release_tree_a_bad_case_and_a_cwd_path():
    p = _cli("-P", "-m", "scripts.sinusoid.regenerate_data", "--case", CASE, "--out", str(REL / "regen"))
    assert p.returncode == 2 and "outside the release tree" in p.stderr
    assert not (REL / "regen").exists()
    p = _cli("-P", "-m", "scripts.sinusoid.regenerate_data", "--case", "manuscript_top", "--out", "/tmp/x")
    assert p.returncode == 2 and "invalid choice" in p.stderr
    p = _cli("-c", "from scripts.sinusoid.regenerate_data import main; main([])", safe_path=False)
    assert p.returncode == 1 and "python -P" in p.stderr
