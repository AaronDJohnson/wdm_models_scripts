"""F-B2: ``galactic_binary.pipeline.foreground_fit`` -- the fit's helpers against the shipped
foreground fits, the resolve loop on a small synthetic Sangria record, the MAP fit's wiring,
and (with impulse) the shipped MAP bit for bit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wdm_models.galactic_binary import foreground as FGM
from wdm_models.galactic_binary.foreground import glass_psd
from wdm_models.galactic_binary.pipeline import foreground_fit as F
from wdm_models.galactic_binary.pipeline import synthetic as syn

REL = Path(__file__).resolve().parents[3]
FG = REL / "data" / "galactic_binary" / "foreground"
MONTHS = (3, 6, 12)


def _shipped(months: int) -> dict[str, np.ndarray]:
    with np.load(FG / f"foreground_{months}mo_snr7.npz", allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def _s5(record_property, item: str, observed: float, gate: float) -> None:
    """One ``s5`` junit property (the S5 report's convention, as ``s3``/``s4``)."""
    record_property("s5", json.dumps({"item": item, "observed": float(observed), "gate": float(gate)}))


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_binned_periodogram_and_psds_are_the_shipped_ones(months, record_property):
    """``blocks`` gives the shipped block centres and sizes (fit band and smoothing band);
    ``glass_psd`` at the shipped MAP is the shipped model and ``s_inst`` the shipped
    instrument PSD, bit for bit."""
    z = _shipped(months)
    T = float(z["T"])
    assert T == F.YEAR * months / 12 and float(z["snr_threshold"]) == 7.0
    edges = F.blocks(T)
    assert np.array_equal(np.sqrt(edges[:-1] * edges[1:]) / T, z["fit_f"])
    assert np.array_equal(np.diff(edges), z["fit_M"])
    smooth = F.blocks(T, band=F.SMOOTH_BAND)
    assert np.array_equal(np.sqrt(smooth[:-1] * smooth[1:]) / T, z["confusion_f"])
    assert np.array_equal(glass_psd(z["fit_f"], z["fg_params"]), z["fit_S_model"])
    assert np.array_equal(F.s_inst(z["fit_f"]), z["fit_S_inst"])
    assert np.array_equal(F.GLASS_PRIOR, z["fg_prior"]) and tuple(z["fg_names"]) == F.GLASS_NAMES
    assert F.GLASS_PRIOR is FGM.GLASS_PRIOR and F.GLASS_START is FGM.GLASS_START     # one source
    assert tuple(z["native_names"]) == F.NATIVE and np.array_equal(z["fit_band"], F.FIT_BAND)
    for item in ("fit_blocks", "fit_S_model", "fit_S_inst"):
        _s5(record_property, f"G5/foreground/gb_{months}mo/{item}", 0.0, 0.0)


@pytest.mark.golden
@pytest.mark.parametrize("months", MONTHS)
def test_the_shipped_map_is_a_fixed_point_of_the_polish(months):
    """Nelder-Mead from the shipped MAP stays there at 6 and 12 months; at 3 months (``ln f2``
    on its prior edge) it moves by <= 1e-3 and never lowers the likelihood."""
    z = _shipped(months)
    lnlike, lnprior = F.glass_objective(z["fit_Pbar"], z["fit_M"], z["fit_f"])
    x = F.polish(z["fg_params"], lnlike, lnprior)
    if months == 3:
        assert np.max(np.abs(x - z["fg_params"])) <= 1e-3
        assert lnlike(x) >= lnlike(z["fg_params"])
    else:
        assert np.array_equal(x, z["fg_params"])
    assert lnprior(z["fg_params"]) == 0.0 and lnprior(F.GLASS_PRIOR[:, 1] + 1.0) == -np.inf


def test_binned_power_and_block_mean_are_the_block_averages(rng):
    T = 4096.0
    d = rng.standard_normal((2, 2000)) + 1j * rng.standard_normal((2, 2000))
    edges = F.blocks(T, band=(0.05, 0.4), dlnf=0.05, min_bins=8)
    Pbar, M = F.binned_power(d, T, edges)
    for b in range(len(M)):
        k = np.arange(edges[b], edges[b + 1])
        want = np.sum(np.abs(d[0, k]) ** 2 + np.abs(d[1, k]) ** 2) / (T / 2) / (2 * len(k))
        assert np.isclose(Pbar[b], want, rtol=1e-13) and M[b] == len(k)
    assert np.all(np.diff(edges) >= 8) and edges[0] == int(np.ceil(0.05 * T))
    S = rng.random(edges[-1] - edges[0])
    assert np.allclose(F.block_mean(S, edges), [S[a - edges[0]:b - edges[0]].mean()
                                                for a, b in zip(edges[:-1], edges[1:])], rtol=1e-14)


def test_accumulate_and_band_snr2_use_the_positive_bins():
    out = np.zeros((2, 10), complex)
    h = np.ones((2, 2, 4), complex)
    F.accumulate(out, np.array([-2, 7]), h, -1.0)          # bins -2..1 and 7..10: clipped, summed
    assert np.array_equal(out[0], -np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], complex))
    T = 5.0 * 16                                           # 16 samples: positive bins 1..7
    snr2 = F.band_snr2(np.array([-1, 6]), np.ones((2, 4)), T, lambda f: np.ones_like(f))
    assert np.allclose(snr2, 4.0 / T * np.array([2, 2]))   # bins {1, 2} and {6, 7}


def test_the_map_fit_wiring(monkeypatch):
    """``map_fit_binned``: GLASS's start clipped into the box, ladder ``geomspace(1, 50, n)``,
    proposal covariance ``diag((0.01 width)^2)``, ``save_freq = iterations``, seed 0; then the
    polish of the best cold sample (here the start)."""
    seen = {}

    class Recording(syn.PolishOnlySampler):
        def __init__(self, ndim, lnlike, lnprior, **kw):
            super().__init__(ndim, lnlike, lnprior, **kw)
            seen["self"] = self

    z = _shipped(6)
    x, sd, fit = F.map_fit_binned(z["fit_Pbar"], z["fit_M"], z["fit_f"], iterations=7, ntemps=3,
                                  sampler_cls=Recording)
    s = seen["self"]
    width = F.GLASS_PRIOR[:, 1] - F.GLASS_PRIOR[:, 0]
    assert s.ndim == 5 and s.num_iterations == 7
    assert s.kwargs["seed"] == 0 and s.kwargs["ntemps"] == 3 and s.kwargs["save_freq"] == 7
    assert s.kwargs["verbose"] is False
    assert np.array_equal(s.kwargs["ladder"], np.geomspace(1.0, 50.0, 3))
    assert np.array_equal(s.kwargs["sample_cov"], np.diag((0.01 * width) ** 2))
    assert np.array_equal(s.x0, np.clip(F.GLASS_START, F.GLASS_PRIOR[:, 0] + 1e-9, F.GLASS_PRIOR[:, 1] - 1e-9))
    lnlike, lnprior = F.glass_objective(z["fit_Pbar"], z["fit_M"], z["fit_f"])
    assert np.array_equal(x, F.polish(s.x0, lnlike, lnprior)) and np.array_equal(sd, np.zeros(5))
    assert fit["lnlike"] == lnlike(x) and np.array_equal(fit["S_model"], glass_psd(z["fit_f"], x))


def test_map_fit_without_impulse_says_what_to_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "impulse":
            raise ImportError("hidden")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    z = _shipped(6)
    with pytest.raises(ImportError, match=r"impulse-mcmc v2 964283c"):
        F.map_fit_binned(z["fit_Pbar"], z["fit_M"], z["fit_f"], iterations=2)


def _impulse_stub(version):
    """A stand-in ``impulse`` module: ``version`` as ``__version__`` (None: no attribute, like
    impulse-mcmc 1.0.0, whose ``__init__`` is ``from .base import PTSampler``)."""
    import types

    mod = types.ModuleType("impulse")

    class PTSampler:
        def __init__(self, *a, **k):
            raise AssertionError("constructed")

    mod.PTSampler = PTSampler
    if version is not None:
        mod.__version__ = version
    return mod


@pytest.mark.parametrize("version,metadata", [(None, "1.0.0"), ("1.0.0", None), ("1.2", "1.2"), (None, None),
                                              ("v2", None)])
def test_map_fit_refuses_impulse_1x_before_a_typeerror(monkeypatch, version, metadata):
    """P4P: PyPI's impulse-mcmc 1.0.0 imports and has a PTSampler, but not v2's constructor (a
    TypeError later): the guard refuses it -- and an unknown version -- with what to install."""
    import importlib.metadata as md
    import sys

    def fake(name):
        if metadata is None:
            raise md.PackageNotFoundError(name)
        return metadata

    monkeypatch.setitem(sys.modules, "impulse", _impulse_stub(version))
    monkeypatch.setattr(md, "version", fake)
    with pytest.raises(ImportError, match=r"impulse-mcmc v2 964283c") as err:
        F.impulse_sampler()
    assert "git+https://github.com/AaronDJohnson/impulse_mcmc@964283c" in str(err.value)
    z = _shipped(6)
    with pytest.raises(ImportError, match=r"not impulse v2"):
        F.map_fit_binned(z["fit_Pbar"], z["fit_M"], z["fit_f"], iterations=2)


def test_impulse_v2_is_accepted(monkeypatch):
    import sys

    stub = _impulse_stub("2.0.0")
    monkeypatch.setitem(sys.modules, "impulse", stub)
    assert F.impulse_sampler() is stub.PTSampler


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.parametrize("months", MONTHS)
def test_the_impulse_map_is_the_shipped_map_bit_for_bit(months, record_property):
    """With impulse (v2 964283c) the legacy settings (100000 iterations, 8 temperatures, seed 0)
    on the shipped binned periodogram give the shipped ``fg_params`` and ``fg_sd`` exactly
    (~30-40 s per duration)."""
    pytest.importorskip("impulse", reason="impulse is the [sampling] extra (installed with S9)")
    z = _shipped(months)
    x, sd, _ = F.map_fit_binned(z["fit_Pbar"], z["fit_M"], z["fit_f"], iterations=100000, ntemps=8)
    assert np.array_equal(x, z["fg_params"]) and np.array_equal(sd, z["fg_sd"])
    _s5(record_property, f"G5/foreground/gb_{months}mo/map", float(np.max(np.abs(x - z["fg_params"]))), 0.0)


# --- the resolve loop on the synthetic record ----------------------------------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    pytest.importorskip("binary_galactica", reason="the waveforms need binary_galactica (JAX)")
    from wdm.backend import enable_x64

    enable_x64()
    root = tmp_path_factory.mktemp("sangria")
    h5 = syn.write_synthetic_sangria(root / "synthetic.h5", backend="fourier")
    lines: list[str] = []
    res = F.resolve(h5, 3, 7.0, T=syn.SYN_T, backend="fourier", map_iterations=5, map_ntemps=2,
                    sampler_cls=syn.PolishOnlySampler, log=lines.append)
    return h5, res, lines, root


def test_the_catalogue_readers(synthetic):
    h5, *_ = synthetic
    labels = F.labels(h5)
    assert labels.tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [2, 0]]
    native = F.native_of(h5, np.arange(7))
    want = np.array([s[1:] for s in syn.SYN_SOURCES])
    want[4:6, 1] = np.float32(want[4:6, 1])                 # igb FrequencyDerivative is float32
    assert np.allclose(native, want, rtol=1e-15, atol=1e-15)        # cos(arccos(0)) = 6e-17
    assert np.array_equal(F.read_column(h5, "Amplitude"), want[:, 2])


def test_read_gb_ae_is_the_dt_rfft_of_the_summed_components(synthetic):
    import h5py
    from scipy.fft import rfft

    h5, *_ = synthetic
    with h5py.File(h5, "r") as f:
        xyz = {k: sum(f[f"sky/{c}/tdi"][:].reshape(-1)[k] for c in F.COMPONENTS) for k in "XYZ"}
    want = np.stack([F.DT * rfft((xyz["Z"] - xyz["X"]) / np.sqrt(2)),
                     F.DT * rfft((xyz["X"] - 2 * xyz["Y"] + xyz["Z"]) / np.sqrt(6))])
    got = F.read_gb_ae(h5, syn.SYN_T)
    assert np.max(np.abs(got - want)) <= 1e-15 * np.max(np.abs(want))
    with pytest.raises(ValueError, match="timestamps"):
        F.read_gb_ae(h5, syn.SYN_T, dt=4.0)


def test_the_generator_batches_equal_one_source_at_a_time(synthetic):
    """The vectorised GLASS vector of ``Generator.batches`` is ``params_from_native``'s: each
    batched spectrum equals binary_galactica's single-source call to 1e-12 of its peak
    (``jit(vmap)`` against eager; measured 1.5e-13)."""
    import binary_galactica as bg

    h5, *_ = synthetic
    T = syn.SYN_T
    native = F.native_of(h5, np.arange(7))
    gen = F.Generator(T, backend="fourier")
    orbit = bg.orbit_cache(T, 0.0, 64)
    seen = 0
    for p, qmin, h in gen.batches(native, np.full(7, 64)):
        for j, i in enumerate(p):
            w = bg.ucb_waveform(bg.params_from_native(*native[i], T=T), T=T, BW=64, positions=orbit,
                                backend="fourier")
            h1 = np.sqrt(T) * np.stack([np.asarray(w["A"]), np.asarray(w["E"])])
            assert int(w["qmin"]) == int(qmin[j])
            assert np.max(np.abs(h[j] - h1)) <= 1e-12 * np.max(np.abs(h1))
            seen += 1
    assert seen == 7 and gen.bandwidth(native[:1]).tolist() == [64]          # GLASS heuristic, safety 2


def test_resolve_drops_the_confused_binary_and_converges(synthetic):
    h5, res, lines, _ = synthetic
    assert res["n_catalogue"] == 7 and res["n_candidates"] == 4 and res["backend"] == "fourier"
    assert [(h["iteration"], h["n_resolved"], h["n_dropped"]) for h in res["history"]] == [(1, 4, 1), (2, 3, 0)]
    assert res["resolved_label"].tolist() == [[0, 0], [2, 0], [1, 0]]       # ascending f0
    assert np.all(res["resolved_snr"] >= 7.0) and np.all(res["resolved_snr"] < res["resolved_snr_inst"])
    assert np.array_equal(res["resolved_native"], F.native_of(h5, np.array([0, 6, 4])))
    assert 2.0 < res["proxy_check_max"] < 3.0              # dgb row 2, below the cut, reaches 2.29
    level = res["history"][-1]["confusion_over_instrument"]
    assert all(2.0 < level[f"{c} mHz"] < 4.0 for c in (1, 2, 3))
    assert any("1 of 4 fall below SNR 7" in s for s in lines)


def test_write_has_the_shipped_schema(synthetic):
    _, res, _, root = synthetic
    path = F.write(res, root / "fit")
    assert path.name == "foreground_3mo_snr7.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        assert str(z["waveform_backend"]) == "fourier"
    assert keys == set(_shipped(3)) | {"waveform_backend"}
    summary = json.loads(path.with_suffix(".json").read_text())
    shipped = json.loads((FG / "foreground_3mo_snr7.json").read_text())
    assert set(shipped) <= set(summary) and summary["n_resolved"] == 3
    assert F.compare_fit(np.load(path), np.load(path))["ok"]
    bad = dict(np.load(path))
    bad["resolved_component"] = bad["resolved_component"][::-1]
    assert any("resolved_component" in p for p in F.compare_fit(bad, np.load(path))["problems"])
