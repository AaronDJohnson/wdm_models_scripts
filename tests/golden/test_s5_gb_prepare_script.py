"""F-B5: ``scripts/galactic_binary/prepare.py --from-shipped`` -- the comparison policies, the
CLI's gate and guards (with a stub engine), the release engine (F-A's ``prepare`` port: the S5
"prepared regeneration" gate; 6 and 12 months ``slow``) and the legacy engine on 3 months."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from scripts.galactic_binary import prepare as PR
from scripts.golden.paths import DATA, REL, legacy_available

pytestmark = pytest.mark.golden
PREPARED = DATA / "galactic_binary" / "prepared"
#: ``calc_Nt`` of the release port (measured 2026-10-13; the legacy function gives
#: 246.86598221011158 / 444.3893998481366 / 925.7765186242784, F-A8's gate 1e-6 relative).
NT_STAR = {"gb_3mo": 246.86598304856258, "gb_6mo": 444.38939984813663, "gb_12mo": 925.7765186111524}


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def _s5(record_property, item: str, observed: float, gate: float) -> None:
    record_property("s5", json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


def test_the_prepared_policy():
    from wdm_models.galactic_binary.prepared import PAPER_KEYS

    shipped = _npz(PREPARED / "gb_3mo.npz")
    assert set(shipped) == set(PAPER_KEYS) | set(PR.LEGACY_ONLY)
    same = PR.compare_prepared(shipped, shipped)
    assert same["ok"] and same["keys"]["mu_fit"] == 0.0 and same["keys"]["mu_freq"] == 0.0
    assert set(same["keys"]) == set(PAPER_KEYS) - {"nt_star"} and "nt_star" in same["info"]
    assert sum(v == "equal" for v in same["keys"].values()) == 50
    assert same["legacy_only"] == {k: "same file name" if k == "matrix_store" else "equal" for k in PR.LEGACY_ONLY}

    def changed(**kw):
        new = dict(shipped)
        for k, v in kw.items():
            if v is None:
                del new[k]
            else:
                new[k] = v
        return PR.compare_prepared(new, shipped)

    assert changed(mu_fit=shipped["mu_fit"] + 5e-7)["ok"]
    assert not changed(mu_fit=shipped["mu_fit"] + 2e-6)["ok"]
    assert not changed(mu_freq=shipped["mu_freq"] + 2e-6)["ok"]
    assert changed(nt_star=np.float64(246.86598304856258))["ok"]
    release = changed(**{k: None for k in PR.LEGACY_ONLY})                 # the release port's file
    assert release["ok"] and set(release["legacy_only"].values()) == {"not regenerated"}
    assert changed(matrix_store=np.array("/elsewhere/store_snr7_3mo_3bands_pm3.npz"))["ok"]
    z = shipped["z"].copy()
    z[0, 7] = np.nextafter(z[0, 7].real, np.inf) + 1j * z[0, 7].imag
    for bad, word in ((changed(z=z), "z is not array_equal"),
                      (changed(store_z=None), "store_z is only in the shipped"),
                      (changed(NT=np.int32(240)), "NT is not array_equal"),
                      (changed(extra=np.zeros(1)), "extra is not a prepared key"),
                      (changed(rho_win=shipped["rho_win"] * 2), "rho_win (legacy-only) is not"),
                      (changed(matrix_store=np.array("/elsewhere/other.npz")), "matrix_store (legacy-only)")):
        assert not bad["ok"] and any(word in p for p in bad["problems"]), (word, bad["problems"])


def test_the_report_policy():
    shipped = json.loads((PREPARED / "gb_3mo_report.json").read_text())
    assert PR.compare_report(shipped, shipped)["ok"]
    near = dict(shipped, snr_full=shipped["snr_full"] * (1 + 1e-10), nt_star=246.87,
                segment="galactic_binary/segments/segment_snr7_3mo_3bands.npz",
                foreground=dict(shipped["foreground"], model="GLASS (foreground.glass_psd)"))
    rep = PR.compare_report(near, shipped)
    assert rep["ok"] and rep["info"]["nt_star"] == {"rebuilt": 246.87, "shipped": shipped["nt_star"]}
    assert rep["info"]["foreground_model"]["rebuilt"] == "GLASS (foreground.glass_psd)"
    params = list(shipped["foreground"]["params_lnA_lnf1_alpha_lnfk_lnf2"])
    params[0] += 1e-12
    for bad in (dict(shipped, snr_full=shipped["snr_full"] * (1 + 1e-8)), dict(shipped, window_bins=[0, 1]),
                dict(shipped, foreground=dict(shipped["foreground"], params_lnA_lnf1_alpha_lnfk_lnf2=params))):
        assert not PR.compare_report(bad, shipped)["ok"]


def _stub_engine(scale: float = 0.0, *, fail: bool = False):
    def rebuild(cases, out_dir, data_root):
        if fail:
            raise RuntimeError("no port")
        paths = {}
        for case in cases:
            a = _npz(data_root / "galactic_binary" / "prepared" / f"{case}.npz")
            a["mu_fit"] = a["mu_fit"] + scale
            np.savez(out_dir / f"{case}.npz", **a)
            shutil.copyfile(data_root / "galactic_binary" / "prepared" / f"{case}_report.json",
                            out_dir / f"{case}_report.json")
            paths[case] = out_dir / f"{case}.npz"
        return paths
    return rebuild


@pytest.mark.parametrize("scale,code", [(5e-7, 0), (2e-6, 1)])
def test_the_prepare_cli_gates_on_the_policy(tmp_path, monkeypatch, capsys, scale, code):
    monkeypatch.setitem(PR.ENGINES, "release", _stub_engine(scale))
    report = tmp_path / "report.json"
    assert PR.main(["--from-shipped", "--case", "gb_3mo", "--out", str(tmp_path / "out"),
                    "--report", str(report)]) == code
    rep = json.loads(report.read_text())
    assert rep["engine"] == "release" and rep["ok"] is (code == 0)
    assert abs(rep["cases"]["gb_3mo"]["npz"]["keys"]["mu_fit"] - scale) < 1e-12
    assert ("PASS" if code == 0 else "FAIL") in capsys.readouterr().out


def test_the_prepare_cli_fails_closed_and_refuses_the_release_tree(tmp_path, monkeypatch):
    monkeypatch.setitem(PR.ENGINES, "release", _stub_engine(fail=True))
    report = tmp_path / "report.json"
    assert PR.main(["--from-shipped", "--case", "gb_3mo", "--out", str(tmp_path), "--report", str(report)]) == 1
    rep = json.loads(report.read_text())
    assert rep["ok"] is False and rep["error"] == "rebuild raised RuntimeError: no port"

    def no_report(cases, out_dir, data_root):         # a rebuilt case without its <case>_report.json
        _stub_engine()(cases, out_dir, data_root)
        for case in cases:
            (out_dir / f"{case}_report.json").unlink()
        return {case: out_dir / f"{case}.npz" for case in cases}

    monkeypatch.setitem(PR.ENGINES, "release", no_report)
    report.write_text(json.dumps({"ok": True}))        # a stale passing report is overwritten
    assert PR.main(["--from-shipped", "--case", "gb_3mo", "--out", str(tmp_path / "out"),
                    "--report", str(report)]) == 1
    rep = json.loads(report.read_text())
    assert rep["ok"] is False and rep["error"].startswith("comparison raised FileNotFoundError")
    with pytest.raises(SystemExit) as e:
        PR.main(["--from-shipped", "--out", str(REL / "tmp_out")])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        PR.main(["--case", "gb_3mo"])                      # --from-shipped is required
    assert e.value.code == 2


@pytest.mark.parametrize("case", ["gb_3mo", pytest.param("gb_6mo", marks=pytest.mark.slow),
                                  pytest.param("gb_12mo", marks=pytest.mark.slow)])
def test_the_release_engine_regenerates_the_shipped_case(case, tmp_path, record_property):
    """F-A's ``prepare.prepare`` (``mu`` and ``calc_Nt`` refitted) + ``prepare.add_matrix`` on the
    shipped segment and store give the shipped file under the policy (spec section 7.3: the
    deterministic keys ``array_equal``, ``mu_fit`` <= 1e-6); ``nt_star`` is the fixed-band
    ``calc_Nt``'s.  ~5 s at 3 months (6 / 12 months: ~8 / ~19 s, ``slow``)."""
    pytest.importorskip("binary_galactica", reason="the prepare port's prior and fits need binary_galactica (JAX)")
    report = tmp_path / "report.json"
    code = PR.main(["--from-shipped", "--case", case, "--out", str(tmp_path / "out"), "--report", str(report)])
    rep = json.loads(report.read_text())
    assert "error" not in rep, rep.get("error")
    c = rep["cases"][case]
    keys = c["npz"]["keys"]
    _s5(record_property, f"G5/prepared/{case}/deterministic_keys",
        sum(v != "equal" for k, v in keys.items() if k not in PR.POLICY["fit_atol"]), 0.0)
    _s5(record_property, f"G5/prepared/{case}/fits", max(keys["mu_fit"], keys["mu_freq"]), 1e-6)
    _s5(record_property, f"G5/prepared/{case}/snr_full", c["report"]["keys"]["snr_full"], 1e-9)
    assert code == 0, c["npz"]["problems"] + c["report"]["problems"]
    assert set(c["npz"]["legacy_only"].values()) == {"not regenerated"}
    assert abs(c["npz"]["info"]["nt_star"]["rebuilt"] / NT_STAR[case] - 1.0) <= 1e-6


@pytest.mark.legacy
@pytest.mark.skipif(not legacy_available(), reason="frozen legacy trees not present (CI/Colab)")
def test_the_legacy_engine_reproduces_the_shipped_3_month_case(tmp_path, record_property):
    """The frozen ``prepare.py`` on the shipped segment and store gives the shipped file: every
    compared key ``array_equal`` (the eight legacy-only keys too), ``mu_fit`` exactly,
    ``nt_star`` 246.866 (shipped 253.633)."""
    report = tmp_path / "report.json"
    assert PR.main(["--from-shipped", "--engine", "legacy", "--case", "gb_3mo", "--out", str(tmp_path / "out"),
                    "--report", str(report)]) == 0
    c = json.loads(report.read_text())["cases"]["gb_3mo"]
    assert c["npz"]["keys"]["mu_fit"] == 0.0 and c["npz"]["keys"]["mu_freq"] == 0.0
    assert c["npz"]["info"]["nt_star"]["rebuilt"] == 246.86598221011158
    assert c["npz"]["info"]["nt_star"]["shipped"] == 253.63256112042106
    assert sum(v == "equal" for v in c["npz"]["keys"].values()) == len(c["npz"]["keys"]) - 2 == 50
    assert set(c["npz"]["legacy_only"].values()) == {"equal", "same file name"}
    assert c["report"]["keys"]["snr_full"] == 0.0
    _s5(record_property, "G5/prepared/gb_3mo/legacy_engine", len(c["npz"]["problems"]), 0.0)
