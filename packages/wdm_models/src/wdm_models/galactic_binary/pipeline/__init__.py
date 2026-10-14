r"""The galactic-binary data pipeline (spec §5.2 ``pipeline/``): from the Sangria HDF5 to the
shipped segments, stores and foreground fits.

* :mod:`.foreground_fit` -- the manuscript's resolve procedure on every Sangria galactic
  binary (SNR >= 7) and the GLASS foreground MAP (``toy_models/galactic_binary/fit_foreground.py``);
* :mod:`.segments` -- the 3-band analysis segments and their +-3-band stores
  (``build_segments.py`` with the ``wdm_gb.prepare_durations`` generator inlined).

Both need the Sangria HDF5 (``LDC2_sangria_training_v2.h5``, cited, not shipped) and the
``[pipeline]`` extra (``h5py``); the MAP fit also needs ``impulse`` (the ``[sampling]`` extra).
The end-to-end runs take minutes to hours and are approval-gated (spec §6): the scripts
``scripts/galactic_binary/{fit_foreground,build_segments}.py`` document them.  Nothing here
imports ``jax``, ``binary_galactica``, ``h5py`` or ``impulse`` at module import (the
dependency-direction probe imports every ``wdm_models`` module with JAX absent).

**Waveform backend.**  The release default is ``"onfly"`` everywhere (user ruling 2026-10-11).
The shipped products were made with gbonfly, which binary_galactica's ``"fourier"`` backend
reproduces to ~1e-16 relative; reproducing them therefore passes ``backend="fourier"``
explicitly (:data:`SHIPPED_BACKEND`).
"""

from __future__ import annotations

__all__ = ["SHIPPED_BACKEND"]

#: The binary_galactica backend that reproduces the shipped segments, stores and fits (gbonfly).
SHIPPED_BACKEND = "fourier"
