"""F-B1: ``galactic_binary.pipeline.synthetic`` -- the small Sangria-shaped record the pipeline
tests run on, and the sampler stand-in for the foreground MAP."""

from __future__ import annotations

import numpy as np
import pytest

from wdm_models.galactic_binary.pipeline import SHIPPED_BACKEND
from wdm_models.galactic_binary.pipeline import synthetic as syn


def test_the_shipped_backend_is_fourier():
    assert SHIPPED_BACKEND == "fourier"


@pytest.fixture(scope="module")
def h5(tmp_path_factory):
    pytest.importorskip("binary_galactica", reason="the waveforms need binary_galactica (JAX)")
    from wdm.backend import enable_x64

    enable_x64()
    return syn.write_synthetic_sangria(tmp_path_factory.mktemp("syn") / "synthetic.h5", backend="fourier")


def test_the_record_has_sangrias_layout(h5):
    import h5py

    with h5py.File(h5, "r") as f:
        for name in ("obs/tdi", "sky/mbhb/tdi", "sky/dgb/tdi", "sky/igb/tdi", "sky/vgb/tdi"):
            rec = f[name]
            assert rec.shape == (syn.SYN_N, 1) and rec.dtype.names == ("t", "X", "Y", "Z")
            assert np.array_equal(rec[:].reshape(-1)["t"], np.arange(syn.SYN_N) * syn.SYN_DT)
        sizes = {c: f[f"sky/{c}/cat"].shape for c in ("dgb", "igb", "vgb")}
        assert sizes == {"dgb": (4, 1), "igb": (2, 1), "vgb": (1, 1)}
        assert f["sky/igb/cat"].dtype["FrequencyDerivative"] == np.float32
        assert {"Distance", "Name"} <= set(f["sky/vgb/cat"].dtype.names)
        cat = f["sky/dgb/cat"][:].reshape(-1)
    assert np.allclose(np.pi / 2 - cat["EclipticLatitude"], [s[5] for s in syn.SYN_SOURCES[:4]], rtol=0, atol=1e-15)


def test_the_noise_free_components_are_their_sources(h5):
    """The igb and vgb TDI (no confusion noise) transform back to ``sqrt(T) (X, Y, Z)`` of
    their sources at the generator bandwidth (to 1e-12 of the peak)."""
    import binary_galactica as bg
    import h5py

    T = syn.SYN_T
    with h5py.File(h5, "r") as f:
        x = f["sky/vgb/tdi"][:].reshape(-1)["X"]
    got = syn.SYN_DT * np.fft.rfft(x)
    _, *native = syn.SYN_SOURCES[-1]
    w = bg.ucb_waveform(bg.params_from_native(*native, T=T), T=T, BW=512,
                        positions=bg.orbit_cache(T, 0.0, 512), backend="fourier")
    want = np.zeros_like(got)
    want[np.asarray(w["freq_bin_indices"], int)] = np.sqrt(T) * np.asarray(w["X"])
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


def test_the_polish_only_sampler_has_impulses_surface():
    s = syn.PolishOnlySampler(2, lambda x: -float(x @ x), lambda x: 0.0, ntemps=3, seed=0)
    s.sample(np.array([1.0, 2.0]), num_iterations=10)
    chain = s.load_chain()
    assert chain["samples"].shape == (1, 1, 2) and chain["lnprob"].tolist() == [[-5.0]]
    assert s.num_iterations == 10 and s.kwargs == {"ntemps": 3, "seed": 0}
