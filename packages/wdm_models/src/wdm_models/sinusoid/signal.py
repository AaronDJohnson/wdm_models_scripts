r"""The sinusoid signal :math:`h(t) = A_h\cos(2\pi f_0 t + \phi_0)` (spec §5.1 signal.py).

Operation order: ``phase = 2.0 * pi * f_0 * t + phi_0``, then ``A_h * cos(phase)`` -- WT
``likelihood.signal`` (``wdm_paper_2/toy_models/src/wdm_toys/likelihood.py:110-117``, fdot = 0)
and WG ``MatrixGPU._template_cells`` (``gpu_sinusoid/wdmtoysgpu/matrix.py:629``) associate it
the same way; on NumPy it reproduces the shipped ``h`` bit for bit.  The release sinusoid is
fixed-frequency: a fourth source entry (the chirp's ``fdot``) is refused.

:class:`Sinusoid` carries the sample times as its one pytree leaf, so on JAX they travel as a
``jit`` argument (spec §3.3), never as a closed-over constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wdm.backend import Backend, get_backend, pytree_dataclass

from .config import SinusoidConfig

__all__ = ["Sinusoid", "sinusoid"]


def sinusoid(t: Any, A_h: Any, f_0: Any, phi_0: Any, *, be: Backend | None = None) -> Any:
    """``A_h * cos(2.0 * pi * f_0 * t + phi_0)`` in the oracles' order."""
    be = be or get_backend(t, A_h, f_0, phi_0)
    xp = be.xp
    phase = 2.0 * xp.pi * f_0 * t + phi_0
    return A_h * xp.cos(phase)


@pytree_dataclass(data=("t",), meta=())
@dataclass(frozen=True, eq=False)
class Sinusoid:
    """``source = (A_h, f_0, phi_0) -> h(t)`` on the sample times ``t`` (``t_j = j dt``)."""

    t: Any

    @classmethod
    def for_config(cls, cfg: SinusoidConfig) -> Sinusoid:
        return cls(cfg.times())

    def __call__(self, source: Any, be: Backend | None = None) -> Any:
        if source.shape[-1] != 3:          # static shape: a Python check under jit
            raise ValueError("the release sinusoid is fixed-frequency: source = (A_h, f_0, phi_0); "
                             f"got {source.shape[-1]} entries (fit_fdot is refused)")
        return sinusoid(self.t, source[0], source[1], source[2], be=be)
