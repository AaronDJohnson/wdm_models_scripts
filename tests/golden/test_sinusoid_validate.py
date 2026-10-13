"""scripts/sinusoid/validate.py: golden readers, gates, fail-closed run and CLI (spec
sections 6, 7.2 G1/G2, 7.3).  A stub likelihood stands in for the registry wherever a
defect must be injected; one CLI run uses the real NumPy release on two cheap targets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scripts.golden.paths import GOLDEN, REL
from scripts.sinusoid import validate as V
from wdm_models.sampling.validate import COUNTER_KEYS, ValidationError, build_fingerprint, same_build

pytestmark = pytest.mark.golden
EDGE_EXACT = ("band_exact", "mid_exact", "tri_exact")


def recording_build() -> bool:
    rec = json.loads((GOLDEN / "sinusoid" / "legacy_eval" / "A_references_fast.json").read_text())
    return same_build(build_fingerprint("numpy"), rec["worker"]["runtime"])


class GoldenStub:
    """Returns each row's golden value (G1 and G2 of one target), with a defect on demand."""

    def __init__(self, case, model, defect=None):
        self.table = {}
        for kind in ("g1", "g2"):
            pts = V._points(kind, case, model, "full", GOLDEN)
            for th, ll in zip(pts["theta"], pts["reference"], strict=True):
                self.table[th.tobytes()] = float(ll)
        self.ndim, self.defect = pts["theta"].shape[1], defect
        self.states = 0 if model.startswith("whittle") else 1      # FourierWhittle factors nothing
        self.reset_counters()

    def reset_counters(self):
        self.c = dict.fromkeys(COUNTER_KEYS, 0)

    def counters(self):
        return dict(self.c, cache_hits=0)

    def __call__(self, Z):
        v = np.array([self.table[np.asarray(z, float).tobytes()] for z in Z])
        if self.defect == "nan":
            v[0] = np.nan
        elif self.defect == "offset":
            v[0] += 1e-3
        elif self.defect == "nudge":
            v = v + 5e-9
        elif self.defect == "flip":
            v[np.flatnonzero(v == -np.inf)] = -1e6
        bad = int(np.sum(v == -np.inf))
        self.c.update(evaluations=self.c["evaluations"] + len(Z),
                      noise_states=self.c["noise_states"] + self.states * len(Z),
                      pd_failed_rows=self.c["pd_failed_rows"] + bad, pd_failed_states=self.c["pd_failed_states"] + bad)
        self.last_failed = np.zeros(len(Z), bool) if self.defect == "last_failed" else v == -np.inf
        return v


def stub_builder(defect=None):
    def build(toy, case, model, **kw):
        assert toy == "sinusoid" and kw["backend"] == "numpy"
        if defect == "raise":
            raise RuntimeError("no such model")
        return GoldenStub(case, model, defect)
    return build


def test_select_refuses_unknown_names_and_an_empty_selection():
    assert V.select(None, None)[:2] == [("manuscript_low", "whittle_raw"), ("manuscript_low", "whittle_cut")]
    assert len(V.select(None, None)) == 45
    assert V.select(["manuscript_high", "manuscript_mid"], ["exact"]) == [("manuscript_mid", "exact"),
                                                                       ("manuscript_high", "exact")]
    with pytest.raises(ValidationError, match="unknown"):
        V.select(["manuscript_mid"], ["band_win"])
    with pytest.raises(ValidationError, match="empty selection"):
        V.select([], None)


def test_approved_gates_are_the_user_approved_table():
    ref = V.approved_gates("numpy", "references")
    assert (ref["whittle_raw"].atol, ref["whittle_raw"].rtol) == (1e-9, 0.0)
    assert (ref["band_pad"].atol, ref["band_pad"].rtol) == (1e-9, 1e-12)
    assert (ref["tri_exact"].atol, ref["tri_exact"].rtol) == (1e-8, 1e-12)
    assert (ref["exact"].atol, ref["exact"].rtol) == (1e-7, 1e-12)
    rows = V.approved_gates("numpy", "chain_rows")
    assert (rows["band_raw"].atol, rows["exact"].atol, rows["exact"].rtol) == (1e-5, 1e-4, 0.0)
    jx = V.approved_gates("jax", "references")
    assert (jx["tri_exact"].atol, jx["tri_exact"].rtol, jx["exact"].atol) == (1e-5, 1e-8, 1e-4)


def _golden_copy(tmp_path):
    root = tmp_path / "golden"
    shutil.copytree(GOLDEN / "sinusoid", root / "sinusoid")
    return root


def test_unapproved_or_incomplete_gates_are_refused(tmp_path):
    root = _golden_copy(tmp_path)
    path = root / "sinusoid" / "tolerances_measured.json"
    tol = json.loads(path.read_text())
    path.write_text(json.dumps(dict(tol, status="PROPOSED")))
    with pytest.raises(ValidationError, match="not approved"):
        V.approved_gates("numpy", "references", root)
    del tol["approved"]["numpy"]["references"]["tri_pad"]
    path.write_text(json.dumps(tol))
    with pytest.raises(ValidationError, match="tri_pad"):
        V.approved_gates("numpy", "references", root)


def test_fast_tier_points_are_part_one_selection():
    """The A6 selection (sinusoid_goldens.fast_indices): truth, first tight, first wide, then
    the first -inf point (edge-exact) or the first stress corner (index 25)."""
    files = json.loads((GOLDEN / "sinusoid" / "references" / "INDEX.json").read_text())["files"]
    for case, model in V.select(None, None):
        pts = V.reference_points(case, model, "fast")
        assert pts["index"].tolist() == files[f"{case}_{model}_reference.npz"]["fast_index"]
        assert pts["index"].tolist() == ([0, 1, 2, 4] if model in EDGE_EXACT else [0, 1, 2, 25])
        assert (pts["reference"][-1] == -np.inf) == (model in EDGE_EXACT)
        assert pts["near"].tolist() == [True, True, False, False]
        full = V.reference_points(case, model, "full")
        assert len(full["index"]) == (27 if model.startswith("whittle") else 30)


def test_chain_rows_are_the_fast_positions_with_the_phase_wrapped():
    for case, model in V.select(["manuscript_mid"], None):
        z = np.load(GOLDEN / "sinusoid" / "chain_rows" / f"{case}_{model}.npz")
        pts = V.chain_rows(case, model, "fast")
        assert pts["index"].tolist() == z["row_index"][[0, 1, 19]].tolist()
        assert np.array_equal(pts["reference"], z["lnlike"][[0, 1, 19]]) and pts["near"].all()
        lo, hi = -np.pi, np.pi
        assert np.array_equal(pts["theta"][:, 2], lo + np.mod(z["theta"][[0, 1, 19], 2] - lo, hi - lo))
        assert np.array_equal(np.delete(pts["theta"], 2, axis=1), np.delete(z["theta"][[0, 1, 19]], 2, axis=1))
        assert len(V.chain_rows(case, model, "full")["index"]) == 20


@pytest.mark.parametrize("model", ["whittle_raw", "band_exact"])
def test_a_golden_stub_passes_g1(model):
    rep = V.run(cases=["manuscript_mid"], models=[model], goldens=("g1",), builder=stub_builder())
    t = rep["targets"][f"manuscript_mid/{model}"]
    assert rep["ok"] and t["ok"] and rep["n_points"] == 4 and rep["failed_targets"] == []
    assert t["g1"]["max_abs"] == 0.0 and t["g1"]["inf_mismatch"] == 0
    assert t["g1"]["counters"]["pd_failed_rows"] == (1 if model == "band_exact" else 0)
    assert rep["recording_build"] is recording_build() and ("lineage_A" in t["g1"]) is recording_build()


@pytest.mark.parametrize("defect,message", [("nan", "NaN"), ("offset", "over the gate"), ("flip", "-inf mask"),
                                            ("last_failed", "last_failed"), ("raise", "build failed")])
def test_every_defect_fails_the_run(defect, message):
    rep = V.run(cases=["manuscript_mid"], models=["band_exact"], goldens=("g1",), builder=stub_builder(defect))
    assert not rep["ok"] and rep["failed_targets"] == ["manuscript_mid/band_exact"]
    assert any(message in p for p in rep["problems"]), rep["problems"]


def test_g2_is_gated_against_the_chain_and_against_lineage_a():
    """G2 gates the chain's own lnlike at the approved chain-row atol (1e-5); on the recording
    build the release is ALSO compared with lineage A's re-evaluation of the same rows at the
    model's NumPy reference gate (band_raw: atol 1e-9).  A stub off by 5e-9 passes the first
    and, there, fails the second."""
    clean = V.run(cases=["manuscript_mid"], models=["band_raw"], goldens=("g2",), builder=stub_builder())
    g2 = clean["targets"]["manuscript_mid/band_raw"]["g2"]
    assert clean["ok"] and g2["max_abs"] == 0.0
    off = V.run(cases=["manuscript_mid"], models=["band_raw"], goldens=("g2",), builder=stub_builder("nudge"))
    g2 = off["targets"]["manuscript_mid/band_raw"]["g2"]
    assert g2["passed"].all() and g2["max_abs"] == pytest.approx(5e-9, rel=1e-3)
    if recording_build():
        assert clean["targets"]["manuscript_mid/band_raw"]["g2"]["lineage_A"]["max_abs"] == pytest.approx(
            2.9103830456733704e-11)                                    # chain vs lineage A: inside 1e-9
        assert not off["ok"] and any("NumPy reference gate against lineage A" in p for p in off["problems"])
    else:
        assert off["ok"] and "lineage_A" not in g2


def test_a_missing_or_altered_reference_is_refused(tmp_path):
    root = _golden_copy(tmp_path)
    ref = root / "sinusoid" / "references" / "manuscript_mid_mid_raw_reference.npz"
    ref.write_bytes(ref.read_bytes() + b"\0")
    with pytest.raises(ValidationError, match="sha256"):
        V.run(cases=["manuscript_mid"], models=["mid_raw"], golden=root, builder=stub_builder())
    ref.unlink()
    with pytest.raises(ValidationError, match="missing reference"):
        V.run(cases=["manuscript_mid"], models=["mid_raw"], golden=root, builder=stub_builder())
    (root / "sinusoid" / "chain_rows" / "manuscript_mid_band_raw.npz").unlink()
    with pytest.raises(ValidationError, match="missing golden file"):
        V.run(cases=["manuscript_mid"], models=["band_raw"], goldens=("g2",), golden=root, builder=stub_builder())


def _cli(*args, cwd=REL, safe_path=True):
    """The CLI as ``source env/activate.sh`` leaves it: only the release root on PYTHONPATH
    (validate.py puts the three package sources on sys.path itself)."""
    drop = ("PYTHONPATH",) if safe_path else ("PYTHONPATH", "PYTHONSAFEPATH")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(PYTHONPATH=str(REL), PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_X64="true")
    return subprocess.run([sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True)


def test_cli_validates_two_cheap_targets_and_writes_the_report(tmp_path):
    out = tmp_path / "report.json"
    p = _cli("-P", "-m", "scripts.sinusoid.validate", "--backend", "numpy", "--tier", "fast",
             "--case", "manuscript_mid", "--model", "whittle_raw", "--model", "band_raw", "--report", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert rep["ok"] is True and rep["n_targets"] == 2 and rep["n_points"] == 14
    assert rep["selection"] == {"cases": ["manuscript_mid"], "models": ["whittle_raw", "band_raw"]}
    for name in ("wdm", "domainia", "wdm_models"):                   # the path bootstrap found the release
        assert rep["fingerprint"]["packages"][name]["file"].startswith(str(REL / "packages" / name / "src"))
    assert "[validate] PASS: 2 targets, 14 points" in p.stdout


def test_cli_fails_closed(tmp_path):
    empty = tmp_path / "golden"
    empty.mkdir()
    p = _cli("-P", "-m", "scripts.sinusoid.validate", "--case", "manuscript_mid", "--model", "band_raw",
             "--golden-root", str(empty))
    assert p.returncode == 1 and "cannot validate: missing golden file" in p.stderr
    p = _cli("-P", "-m", "scripts.sinusoid.validate", "--report", str(REL / "report.json"))
    assert p.returncode == 2 and "outside the release tree" in p.stderr
    assert not (REL / "report.json").exists()
    p = _cli("-P", "-m", "scripts.sinusoid.validate", "--model", "band_win")
    assert p.returncode == 2 and "invalid choice" in p.stderr
    p = _cli("-c", "from scripts.sinusoid.validate import main; main([])", safe_path=False)
    assert p.returncode == 1 and "python -P" in p.stderr
