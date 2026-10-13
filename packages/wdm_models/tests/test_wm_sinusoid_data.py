"""E-A6: the shipped sinusoid data (spec §5.1 data.py) and the data-root helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import wdm
from domainia.assets import ArtifactError, ArtifactMissing
from domainia.covariance import ExactTimeCovariance
from domainia.data import TimeSeries
from domainia.domains import TimeDomain
from wdm_models.common import validation as V
from wdm_models.sinusoid import config as C
from wdm_models.sinusoid import data as D
from wdm_models.sinusoid.psd import as_loglinear

DATA = Path(__file__).resolve().parents[3] / "data"
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
pytestmark = pytest.mark.golden


def _arrays(case: str) -> dict:
    with np.load(DATA / "sinusoid" / case / "toy_data.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def test_the_default_data_root_is_the_release_data(monkeypatch):
    monkeypatch.delenv(V.DATA_ENV, raising=False)
    assert V.default_data_root() == DATA
    assert V.resolve_data_root(None) == DATA
    assert V.resolve_data_root(str(DATA)) == DATA
    monkeypatch.setenv(V.DATA_ENV, "/nonexistent/wdm_data")
    assert V.default_data_root() == Path("/nonexistent/wdm_data")


def test_the_pins_are_the_manifest_entries():
    manifest = json.loads((DATA / "MANIFEST.json").read_text())["files"]
    for case in CASES:
        assert manifest[f"sinusoid/{case}/toy_data.npz"] == D.DATA_SHA256[case]
        assert V.manifest_sha256(DATA, f"sinusoid/{case}/toy_data.npz") == D.DATA_SHA256[case]
    with pytest.raises(ArtifactError, match="not in"):
        V.manifest_sha256(DATA, "sinusoid/manuscript_chirp/toy_data.npz")


def test_load_every_case():
    for case in CASES:
        s = D.load(case)
        cfg = C.config(case)
        assert s.case == case and s.cfg == cfg and s.seed == 1
        assert np.array_equal(s.truth("tri_raw"), cfg.truth("tri_raw")) and s.truth("whittle_raw").shape == (5,)
        for name in ("d", "h", "n"):
            v = getattr(s, name)
            assert v.shape == (16384,) and v.dtype == np.float64 and not v.flags.writeable
        assert s.w_d.shape == (65, 256)
        assert np.array_equal(s.w_d, wdm.forward_time(s.d, 64, 256, 2.0 ** -10))
        assert s.sha256 == D.DATA_SHA256[case]
        assert s.snr == pytest.approx(15.0, abs=1e-11)
        ts = s.series()
        assert isinstance(ts, TimeSeries) and ts.domain == TimeDomain(16384, 2.0 ** -10)
        assert np.array_equal(np.asarray(ts.values), s.d)
    assert D.load("manuscript_mid", data_root=DATA).sha256 == D.DATA_SHA256["manuscript_mid"]


def test_load_is_fail_closed_on_the_file(tmp_path):
    shutil.copytree(DATA / "sinusoid" / "manuscript_mid", tmp_path / "sinusoid" / "manuscript_mid")
    shutil.copy(DATA / "MANIFEST.json", tmp_path / "MANIFEST.json")
    assert D.load("manuscript_mid", data_root=tmp_path).sha256 == D.DATA_SHA256["manuscript_mid"]
    path = tmp_path / "sinusoid" / "manuscript_mid" / "toy_data.npz"
    body = bytearray(path.read_bytes())
    body[-100] ^= 1
    path.write_bytes(bytes(body))
    with pytest.raises(ArtifactError, match="sha256"):
        D.load("manuscript_mid", data_root=tmp_path)
    path.unlink()
    with pytest.raises(ArtifactMissing):
        D.load("manuscript_mid", data_root=tmp_path)
    with pytest.raises(KeyError, match="manuscript_chirp"):
        D.load("manuscript_chirp", data_root=tmp_path)


def test_a_manifest_that_disagrees_with_the_pin_is_refused(tmp_path):
    shutil.copytree(DATA / "sinusoid" / "manuscript_mid", tmp_path / "sinusoid" / "manuscript_mid")
    m = json.loads((DATA / "MANIFEST.json").read_text())
    m["files"]["sinusoid/manuscript_mid/toy_data.npz"] = "0" * 64
    (tmp_path / "MANIFEST.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="MANIFEST"):
        D.load("manuscript_mid", data_root=tmp_path)


def test_the_payload_checks():
    case = "manuscript_mid"
    D.check_payload(case, _arrays(case))
    bad = _arrays(case)
    bad["w_d"][3, 7] = np.nextafter(bad["w_d"][3, 7], np.inf)
    with pytest.raises(ValueError, match="w_d"):
        D.check_payload(case, bad)
    bad = _arrays(case)
    bad["h"][0] = np.nextafter(bad["h"][0], np.inf)
    with pytest.raises(ValueError, match="signal"):
        D.check_payload(case, bad)
    bad = _arrays(case)
    bad["n"][5] = 0.0
    with pytest.raises(ValueError, match="noise"):
        D.check_payload(case, bad)
    for key, value in (("A_h", 0.3), ("fdot", 0.5), ("fit_fdot", 1.0), ("f_min", 2.0)):
        bad = _arrays(case)
        bad["_cfg_vals"][list(bad["_cfg_keys"]).index(key)] = value
        with pytest.raises(ValueError, match=key):
            D.check_payload(case, bad)
    with pytest.raises(ValueError, match="manuscript_low"):
        D.check_payload("manuscript_low", _arrays(case))


def test_regenerate_draws_the_wt_voices_on_a_small_grid():
    """``regenerate`` is WT ``data.generate``: ``n = G z`` with the voices in WT order (cosine
    k = 0..N/2, then sine k = 1..N/2-1) and ``z = default_rng(seed).standard_normal(N)``,
    ``d = h + n``.  At the manuscript size it agrees with the shipped ``n`` only to ~1e-13
    relative (not bitwise; the dense factor is 2 GB): ``scripts/sinusoid/regenerate_data.py``."""
    cfg = C.SinusoidConfig(Nf=8, Nt=16, dt=1 / 16, A_h=1.0, f_0=3.5, A_n=2.0, f_min=0.5)
    out = D.regenerate(cfg, seed=7)
    assert set(out) == {"d", "h", "n", "seed"} and out["seed"] == 7
    exact = ExactTimeCovariance(as_loglinear(cfg), cfg.N, cfg.dt, form="dense")
    G = exact.voice_factor((cfg.alpha_c, cfg.alpha_d), cfg.A_n)
    z = np.random.default_rng(7).standard_normal(cfg.N)
    assert np.array_equal(out["n"], G @ z)
    assert np.array_equal(out["d"], out["h"] + out["n"])
    Sigma = cfg.A_n * exact.sigma_tilde((cfg.alpha_c, cfg.alpha_d))
    assert np.max(np.abs(G @ G.T - Sigma)) <= 1e-12 * np.max(np.abs(Sigma))
    assert np.array_equal(D.regenerate(cfg, seed=7)["n"], out["n"])
