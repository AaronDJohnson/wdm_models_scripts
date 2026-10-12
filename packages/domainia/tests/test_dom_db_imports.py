"""D-B7: the section's public names, and no JAX import from maps, spectra or variance."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

EXPORTS = {
    "domainia.maps": ["BinSelect", "Compose", "CropTime", "FourierCrop", "FourierToWDM",
                      "FourierZeroPad", "Identity", "LinearMap", "PushedCovariance", "Restrict",
                      "SupportsDense", "SupportsMatvec", "TimeToFourier", "TimeToWDM", "WDMToTime",
                      "ZeroPadTime", "crop_maps", "domain_size", "on_backend", "pad_maps"],
    "domainia.spectra": ["Envelope", "GenericPSD", "LogLinearPSD", "SeparablePSD",
                         "TabulatedSpectrum"],
    "domainia.variance": ["BandAveragedVariance", "MidpointVariance"],
}


def test_public_names():
    import importlib

    for mod, names in EXPORTS.items():
        m = importlib.import_module(mod)
        assert sorted(m.__all__) == sorted(names), mod
        for n in names:
            assert hasattr(m, n), (mod, n)


def test_importing_the_section_does_not_import_jax():
    code = ("import sys, domainia.maps, domainia.spectra, domainia.variance; "
            "assert 'jax' not in sys.modules, sorted(k for k in sys.modules if k.startswith('jax'))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(PKG / "src"), str(PKG.parent / "wdm" / "src"), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
