"""Where the likelihood runs: the device kind the strategy columns key on, and the run
metadata every validation report records (spec sections 3.2, 3.4 and 6).

``device_kind()`` is ``"cpu"`` without JAX or on XLA-CPU and ``"gpu"`` on a CUDA/ROCm
device; the sinusoid's historical JAX column (``options.resolve``) differs between the two
exactly where WG did (``gpu_sinusoid/wdmtoysgpu/matrix.py:262-263, 279``: exact-row chunk
64 vs 192, rows of one call batched only on a GPU) plus the spec's capacity rule (16 on
GPU, 32 on CPU) and the staged exact rows (CPU only, the XLA-CPU irfft trap).
``run_meta()`` is the ``run_meta`` block of a validation report: the device kind, the JAX
build, x64 and ``XLA_FLAGS`` (validation on a GPU sets ``--xla_gpu_deterministic_ops=true``,
spec section 3.4).  Nothing here imports JAX at module import; S9 adds
``require_accelerator()`` for the runners.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

__all__ = ["DETERMINISTIC_FLAG", "DeviceKind", "device_kind", "run_meta"]

DeviceKind = Literal["cpu", "gpu"]
#: The XLA flag validation runs set on a GPU (spec section 3.4).
DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops=true"
_GPU_PLATFORMS = ("gpu", "cuda", "rocm")


def device_kind(backend: str = "jax") -> DeviceKind:
    """``"cpu"`` for the NumPy backend and for XLA-CPU, ``"gpu"`` for a CUDA/ROCm default
    device; any other JAX platform (TPU, METAL) raises ``RuntimeError`` -- no strategy
    column was validated there."""
    if backend == "numpy":
        return "cpu"
    if backend != "jax":
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {backend!r}")
    import jax

    platform = jax.default_backend()
    if platform == "cpu":
        return "cpu"
    if platform in _GPU_PLATFORMS:
        return "gpu"
    raise RuntimeError(f"no sinusoid strategy column for the JAX platform {platform!r}")


def run_meta(backend: str) -> dict[str, Any]:
    """The report's ``run_meta``: python, numpy, scipy, ``XLA_FLAGS``, and on JAX the
    jax/jaxlib versions, the platform, the devices, ``jax_enable_x64`` and whether the
    deterministic-ops flag is set."""
    import numpy as np
    import scipy

    out: dict[str, Any] = {"backend": backend, "device_kind": device_kind(backend),
                           "python": sys.version.split()[0], "numpy": np.__version__,
                           "scipy": scipy.__version__, "XLA_FLAGS": os.environ.get("XLA_FLAGS")}
    if backend == "jax":
        import jax
        import jaxlib

        out.update(jax=jax.__version__, jaxlib=jaxlib.__version__, platform=jax.default_backend(),
                   devices=[str(d) for d in jax.devices()], x64=bool(jax.config.read("jax_enable_x64")),
                   deterministic_ops=DETERMINISTIC_FLAG in (os.environ.get("XLA_FLAGS") or ""))
    return out
