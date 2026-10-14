"""F-B4: ``scripts/galactic_binary/{fit_foreground,build_segments}.py`` -- the CLI guards and
both CLIs end to end on a small synthetic Sangria record (with ``SegmentSpec.paper`` and the
fit's record length substituted).  The real Sangria runs are approval-gated and never run here."""

from __future__ import annotations

import json

import pytest

from scripts.galactic_binary import build_segments as BS
from scripts.galactic_binary import fit_foreground as FF
from scripts.golden.paths import REL

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    pytest.importorskip("binary_galactica")
    from wdm.backend import enable_x64

    enable_x64()
    from wdm_models.galactic_binary.pipeline import foreground_fit as F
    from wdm_models.galactic_binary.pipeline import synthetic as syn

    root = tmp_path_factory.mktemp("cli")
    h5 = syn.write_synthetic_sangria(root / "synthetic.h5", backend="fourier")
    return h5, root, syn, F


def _small_resolve(syn, F):
    def run(h5, months, snr, **kw):
        return F.resolve(h5, months, snr, T=syn.SYN_T, map_iterations=5, map_ntemps=2,
                         sampler_cls=syn.PolishOnlySampler, log=lambda s: None, **kw)
    return run


def test_the_pipeline_clis_refuse_bad_arguments(tmp_path, monkeypatch):
    for main, extra in ((BS.main, []), (FF.main, [])):
        with pytest.raises(SystemExit) as e:
            main(["--out", str(tmp_path)] + extra)          # --sangria-h5 is required
        assert e.value.code == 2
        with pytest.raises(SystemExit) as e:
            main(["--sangria-h5", str(tmp_path / "missing.h5"), "--out", str(tmp_path)])
        assert e.value.code == 2
        with pytest.raises(SystemExit) as e:
            main(["--sangria-h5", __file__, "--out", str(REL / "tmp_out")])
        assert e.value.code == 2
    monkeypatch.setattr(FF, "_impulse_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        FF.main(["--sangria-h5", __file__, "--out", str(tmp_path)])
    assert e.value.code == 2


def test_fit_foreground_cli_on_the_synthetic_record(synthetic, tmp_path, monkeypatch, capsys):
    h5, root, syn, F = synthetic
    monkeypatch.setattr(FF, "_impulse_available", lambda: True)
    monkeypatch.setattr(FF, "resolve", _small_resolve(syn, F))
    out = tmp_path / "fit"
    assert FF.main(["--sangria-h5", str(h5), "--out", str(out), "--months", "3", "--backend", "fourier"]) == 0
    assert (out / "foreground_3mo_snr7.npz").is_file() and (out / "foreground_3mo_snr7.json").is_file()
    # a synthetic fit is not the shipped one: gated with the shipped backend, reported with onfly
    assert FF.main(["--sangria-h5", str(h5), "--out", str(tmp_path / "f2"), "--months", "3", "--backend",
                    "fourier", "--compare-shipped"]) == 1
    assert FF.main(["--sangria-h5", str(h5), "--out", str(tmp_path / "f3"), "--months", "3", "--backend",
                    "onfly", "--compare-shipped"]) == 0
    assert "audit only: backend onfly" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        FF.main(["--sangria-h5", str(h5), "--out", str(tmp_path / "f4"), "--snr", "5", "--compare-shipped"])
    assert e.value.code == 2


def test_build_segments_cli_on_the_synthetic_record(synthetic, tmp_path, monkeypatch, capsys):
    h5, root, syn, F = synthetic
    fg = tmp_path / "fg"
    F.write(_small_resolve(syn, F)(h5, 3, 7.0, backend="fourier"), fg)
    spec = BS.S.SegmentSpec(3, syn.SYN_T, 32, 256)
    monkeypatch.setattr(BS.S.SegmentSpec, "paper", classmethod(lambda cls, months: spec))
    out, report = tmp_path / "out", tmp_path / "report.json"
    common = ["--sangria-h5", str(h5), "--months", "3", "--backend", "fourier", "--foreground-dir", str(fg)]
    assert BS.main(common + ["--out", str(out), "--report", str(report)]) == 0
    assert (out / "segment_snr7_3mo_3bands.npz").is_file()
    assert json.loads(report.read_text())["problems"] == []
    assert BS.main(common + ["--out", str(out), "--store", "--segments-dir", str(out)]) == 0
    assert (out / "store_snr7_3mo_3bands_pm3.npz").is_file()
    assert BS.main(common + ["--out", str(out)]) == 1                     # refuses to overwrite
    assert "FileExistsError" in capsys.readouterr().err
    # against the shipped (real) segment: gated (and failing) with fourier, an audit with onfly
    assert BS.main(common + ["--out", str(tmp_path / "o2"), "--compare-shipped"]) == 1
    onfly = ["--sangria-h5", str(h5), "--months", "3", "--backend", "onfly", "--foreground-dir", str(fg)]
    assert BS.main(onfly + ["--out", str(tmp_path / "o3"), "--compare-shipped"]) == 0
