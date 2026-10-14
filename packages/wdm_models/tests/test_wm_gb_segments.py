"""F-B3: ``galactic_binary.pipeline.segments`` -- every piece that needs no Sangria file
against the shipped segments and stores (window geometry, the instrument draws bit for bit,
the resolved-binary selection, the injection), and ``build_segment`` / ``build_store`` end to
end on a small synthetic Sangria record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wdm_models.galactic_binary.pipeline import segments as S
from wdm_models.galactic_binary.pipeline import synthetic as syn

REL = Path(__file__).resolve().parents[3]
GB = REL / "data" / "galactic_binary"
MONTHS = (3, 6, 12)


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def _segment(months: int) -> dict[str, np.ndarray]:
    return _npz(GB / "segments" / f"segment_snr7_{months}mo_3bands.npz")


def _store(months: int) -> dict[str, np.ndarray]:
    return _npz(GB / "segments" / f"store_snr7_{months}mo_3bands_pm3.npz")


def _s5(record_property, item: str, observed: float, gate: float) -> None:
    record_property("s5", json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_paper_geometry_is_the_shipped_one(months, record_property):
    """``window``/``store_window`` of ``SegmentSpec.paper`` are the shipped files' provenance
    windows; the injection, the generator bandwidth and the seeds are the shipped ones."""
    seg, st = _segment(months), _store(months)
    prov, sprov = json.loads(str(seg["provenance_json"])), json.loads(str(st["provenance_json"]))
    spec = S.SegmentSpec.paper(months)
    assert (spec.T, spec.NT, spec.bw, spec.n_bands) == (S.YEAR * months / 12, prov["window"]["NT"],
                                                         int(seg["inject_bw"]), 3)
    assert S.window(spec) == prov["window"]
    assert S.store_window(spec) == sprov["window"]
    assert tuple(float(seg[f"inject_{k}"]) for k in S.INJECT_KEYS) == S.INJECT_NATIVE
    assert float(seg["f_centre"]) == S.F_CENTRE and prov["subtraction_bw"] == sprov["subtraction_bw"] == spec.bw
    assert prov["instrument_noise"]["seed"] == [S.INSTRUMENT_SEED, months]
    assert sprov["instrument_seed_extra_bins"] == [S.INSTRUMENT_SEED, months, S.OUTER_SEED]
    assert sprov["store_margin_bands"] == S.STORE_MARGIN
    assert prov["bandwidth_check"]["max_error_snr"] == S.MAX_ERROR_SNR
    for item in ("window", "store_window"):
        _s5(record_property, f"G5/segments/gb_{months}mo/{item}", 0.0, 0.0)


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_instrument_noise_is_the_shipped_draw(months, record_property):
    """``simulate_instrument`` reproduces the segment's instrument bins (seed ``[20260929, m]``)
    and the store's (seed ``[20260929, m, 1]`` with the segment's bins spliced in) bit for bit."""
    seg, st = _segment(months), _store(months)
    spec = S.SegmentSpec.paper(months)
    inner = S.simulate_instrument(seg["freqs"], spec.T, [S.INSTRUMENT_SEED, months])
    assert np.array_equal(inner, np.stack([seg["instrument_A"], seg["instrument_E"]]))
    w = S.store_window(spec)
    k = slice(w["bin_lo"] - w["store_lo"], w["bin_lo"] - w["store_lo"] + w["n_bins"])
    outer = S.simulate_instrument(st["freqs"], spec.T, [S.INSTRUMENT_SEED, months, S.OUTER_SEED])
    outer[:, k] = inner
    assert np.array_equal(outer, np.stack([st["instrument_A"], st["instrument_E"]]))
    assert np.array_equal(st["freqs"], (w["store_lo"] + np.arange(w["store_n"])) / spec.T)
    for item in ("instrument_segment", "instrument_store"):
        _s5(record_property, f"G5/segments/gb_{months}mo/{item}", 0.0, 0.0)


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_resolved_binary_selection_is_the_shipped_one(months, record_property):
    seg, st = _segment(months), _store(months)
    prov, sprov = json.loads(str(seg["provenance_json"])), json.loads(str(st["provenance_json"]))
    spec = S.SegmentSpec.paper(months)
    fg = S.load_foreground(GB / "foreground" / f"foreground_{months}mo_snr7.npz", spec.T)
    assert np.array_equal(fg["fg_params"], seg["fg_params"]) and fg["snr_threshold"] == 7.0
    assert len(fg["resolved"]) == prov["resolved_binaries"]
    w = S.store_window(spec)
    sel = S.select_reaching(fg["resolved"], T=spec.T, bin_lo=w["bin_lo"], n_bins=w["n_bins"], bw=spec.bw)
    assert np.array_equal(sel, seg["resolved_subtracted_indices"])
    assert np.array_equal(fg["row"][sel], seg["resolved_subtracted_row"])
    assert np.array_equal(fg["component"][sel], seg["resolved_subtracted_component"])
    doubled = S.select_reaching(fg["resolved"], T=spec.T, bin_lo=w["bin_lo"], n_bins=w["n_bins"], bw=2 * spec.bw)
    assert len(doubled) == prov["bandwidth_check"]["resolved_reaching_window"]
    store = S.select_reaching(fg["resolved"], T=spec.T, bin_lo=w["store_lo"], n_bins=w["store_n"], bw=spec.bw)
    assert len(store) == sprov["resolved_reaching_store"]
    _s5(record_property, f"G5/segments/gb_{months}mo/selection", 0.0, 0.0)


def test_the_release_default_backend_is_onfly():
    """User ruling 2026-10-11: ``onfly`` everywhere in the pipeline; only the comparisons with
    shipped products pass ``pipeline.SHIPPED_BACKEND`` (``fourier``) explicitly."""
    import inspect

    from wdm_models.galactic_binary.pipeline import SHIPPED_BACKEND
    from wdm_models.galactic_binary.pipeline import foreground_fit as F

    for fn in (F.Generator, F.resolve, S.generator, S.subtract_resolved, S.build_segment, S.build_store,
               syn.write_synthetic_sangria):
        assert inspect.signature(fn).parameters["backend"].default == "onfly", fn
    assert SHIPPED_BACKEND == "fourier"


def test_load_foreground_refuses_another_duration():
    with pytest.raises(ValueError, match="is not"):
        S.load_foreground(GB / "foreground" / "foreground_3mo_snr7.npz", S.YEAR / 2)


@pytest.fixture(scope="module")
def bg_x64():
    pytest.importorskip("binary_galactica", reason="the waveforms need binary_galactica (JAX)")
    from wdm.backend import enable_x64

    enable_x64()


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_fourier_injection_is_the_shipped_one(months, bg_x64, record_property):
    """binary_galactica ``fourier`` at the generator bandwidth reproduces the shipped injection
    (gbonfly) on the segment and the store to <= 1e-12 of its peak (measured <= 1.3e-16); the
    injection's SNRs are the shipped ones; ``onfly`` (the release default) is reported."""
    from wdm_models.galactic_binary.prepare import psd_components

    seg, st = _segment(months), _store(months)
    spec = S.SegmentSpec.paper(months)
    w = S.store_window(spec)
    for backend in ("fourier", "onfly"):
        bins, h = S.generator(spec.T, spec.bw, backend=backend)(np.array(S.INJECT_NATIVE))
        for name, lo, n, z in (("segment", w["bin_lo"], w["n_bins"], seg), ("store", w["store_lo"], w["store_n"], st)):
            got = np.zeros((2, n), complex)
            S.add_to_window(got, bins, h, bin_lo=lo)
            old = np.stack([z["injected_A"], z["injected_E"]])
            rel = float(np.max(np.abs(got - old)) / np.max(np.abs(old)))
            if backend == "fourier":
                assert rel <= 1e-12, (name, rel)
                _s5(record_property, f"G5/segments/gb_{months}mo/injection_{name}", rel, 1e-12)
            else:
                assert 1e-6 < rel < 1e-3, (name, rel)                # a different algorithm: audit only
        if backend == "fourier":
            psd = lambda f: np.sum(psd_components(f, seg["fg_params"]), axis=0)  # noqa: E731
            full = S.full_support_snr(bins, h, T=spec.T, psd=psd)
            assert abs(full - float(seg["inject_full_support_snr"])) <= 1e-12 * full
            inj = np.zeros((2, w["n_bins"]), complex)
            S.add_to_window(inj, bins, h, bin_lo=w["bin_lo"])
            snr = S.snr_function(spec.T, psd(seg["freqs"]))(inj)
            assert abs(snr - float(seg["inject_snr"])) <= 1e-12 * snr


def test_validate_bandwidth_error_and_add_to_window():
    S.validate_bandwidth_error(0.1)
    for bad in (0.1000001, np.nan, np.inf):
        with pytest.raises(ValueError, match="bandwidth convergence"):
            S.validate_bandwidth_error(bad)
    win = np.zeros((2, 4), complex)
    S.add_to_window(win, np.array([-1, 0, 3, 4]), np.ones((2, 4), complex), bin_lo=0)
    assert win[0].tolist() == [1, 0, 0, 1]


def test_compare_npz_flags_each_kind_of_difference():
    seg = _segment(3)
    assert S.compare_npz(seg, seg)["ok"]
    bad = dict(seg)
    bad["instrument_A"] = seg["instrument_A"] * (1 + 2.0 ** -52)
    bad["data_E"] = seg["data_E"] * (1 + 1e-11)
    del bad["inject_snr"]
    rep = S.compare_npz(bad, seg)
    assert not rep["ok"] and rep["exact"]["instrument_A"] is False and rep["close"]["data_E"] > 1e-12
    assert any("inject_snr is only in the shipped" in p for p in rep["problems"])


# --- the builders on the synthetic record --------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory, bg_x64):
    from wdm_models.galactic_binary.pipeline import foreground_fit as F

    root = tmp_path_factory.mktemp("segments")
    h5 = syn.write_synthetic_sangria(root / "synthetic.h5", backend="fourier")
    res = F.resolve(h5, 3, 7.0, T=syn.SYN_T, backend="fourier", map_iterations=5, map_ntemps=2,
                    sampler_cls=syn.PolishOnlySampler, log=lambda s: None)
    fg = S.load_foreground(F.write(res, root / "fit"), syn.SYN_T)
    spec = S.SegmentSpec(3, syn.SYN_T, 32, 256)
    lines: list[str] = []
    seg = S.build_segment(h5, spec, root / "out", foreground=fg, backend="fourier", log=lines.append)
    store = S.build_store(h5, spec, seg, root / "out", foreground=fg, backend="fourier", log=lines.append)
    return dict(h5=h5, fg=fg, spec=spec, seg=seg, store=store, root=root, lines=lines)


def test_a_built_segment_and_store_have_the_shipped_schema(built):
    seg, st = _npz(built["seg"]), _npz(built["store"])
    assert built["seg"].name == "segment_snr7_3mo_3bands.npz"
    assert built["store"].name == "store_snr7_3mo_3bands_pm3.npz"
    assert set(seg) == set(_segment(3)) and set(st) == set(_store(3))
    assert str(seg["waveform"]) == "binary_galactica:fourier"
    prov = json.loads(str(seg["provenance_json"]))
    assert prov["window"] == S.window(built["spec"]) == {
        "T": syn.SYN_T, "NT": 32, "source_band": 10, "first_band": 9, "last_band": 11, "bin_lo": 132, "n_bins": 57}
    assert prov["raw_convention"] == "not verified (no catalogue file given)"
    assert prov["bandwidth_check"]["passed"] and prov["resolved_reaching_window"] == 1
    for key in ("data", "residual", "injected", "instrument"):
        assert seg[f"{key}_A"].shape == (57,) and st[f"{key}_A"].shape == (153,)
    assert np.array_equal(seg["data_A"], seg["residual_A"] + seg["injected_A"])
    assert np.array_equal(seg["resolved_subtracted_indices"], [0])          # the 1.05 mHz binary
    assert any("1 resolved binaries subtracted" in s for s in built["lines"])


def test_the_store_keeps_the_segment_verbatim_and_draws_its_own_outer_noise(built):
    seg, st = _npz(built["seg"]), _npz(built["store"])
    w = S.store_window(built["spec"])
    k = slice(w["bin_lo"] - w["store_lo"], w["bin_lo"] - w["store_lo"] + w["n_bins"])
    for key in ("residual", "injected", "instrument", "data"):
        for c in "AE":
            assert np.array_equal(st[f"{key}_{c}"][k], seg[f"{key}_{c}"]), (key, c)
    outer = S.simulate_instrument(st["freqs"], syn.SYN_T, [S.INSTRUMENT_SEED, 3, S.OUTER_SEED])
    mask = np.ones(st["freqs"].size, bool)
    mask[k] = False
    assert np.array_equal(st["instrument_A"][mask], outer[0, mask])
    assert (int(st["inner_bin_offset"]), int(st["inner_n_bins"])) == (132, 57)


def test_the_builders_are_deterministic_and_refuse_to_overwrite(built):
    spec, fg, h5, root = built["spec"], built["fg"], built["h5"], built["root"]
    again = S.build_segment(h5, spec, root / "again", foreground=fg, backend="fourier", log=lambda s: None)
    assert S.compare_npz(_npz(again), _npz(built["seg"]), rtol=0.0)["ok"]
    with pytest.raises(FileExistsError):
        S.build_segment(h5, spec, root / "out", foreground=fg, backend="fourier", log=lambda s: None)
    other = S.SegmentSpec(3, syn.SYN_T, 64, 256)
    with pytest.raises(ValueError, match="analysis window"):
        S.build_store(h5, other, built["seg"], root / "other", foreground=fg, backend="fourier", log=lambda s: None)


def test_a_too_narrow_generator_fails_the_doubling_check(built):
    """At BW 32 doubling the bandwidth moves the subtraction by SNR 0.19 > 0.1 (256: 0.022)."""
    small = S.SegmentSpec(3, syn.SYN_T, 32, 32)
    with pytest.raises(ValueError, match="bandwidth convergence error SNR=0.1885"):
        S.build_segment(built["h5"], small, built["root"] / "narrow", foreground=built["fg"],
                        backend="fourier", log=lambda s: None)


def test_read_ae_crop_and_the_raw_convention_check(built, tmp_path):
    import h5py
    from scipy.fft import rfft

    h5 = built["h5"]
    with h5py.File(h5, "r") as f:
        d, b = f["obs/tdi"][:].reshape(-1), f["sky/mbhb/tdi"][:].reshape(-1)
    x, y, z = (d[k] - b[k] for k in "XYZ")
    want = np.stack([S.DT * rfft((z - x) / np.sqrt(2)), S.DT * rfft((x - 2 * y + z) / np.sqrt(6))])
    got = S.read_ae_crop(h5, T=syn.SYN_T)
    assert np.array_equal(got, want)
    bins = np.arange(100, 400)
    cat = tmp_path / "catalog.npz"
    np.savez(cat, Tobs=syn.SYN_T, freqs=bins / syn.SYN_T, channels_mbhb_sub_real=got[:, bins].real,
             channels_mbhb_sub_imag=got[:, bins].imag)
    rep = S.verify_raw_convention(h5, cat)
    assert max(rep["full_year_relative_AE_error"]) < 1e-15
    np.savez(cat, Tobs=syn.SYN_T, freqs=bins / syn.SYN_T, channels_mbhb_sub_real=1.001 * got[:, bins].real,
             channels_mbhb_sub_imag=got[:, bins].imag)
    with pytest.raises(ValueError, match="does not reproduce"):
        S.verify_raw_convention(h5, cat)
    with pytest.raises(ValueError, match="too short"):
        S.read_ae_crop(h5, T=2 * syn.SYN_T)
