"""Special functions shared by the WDM II kernels (dual backend).

* :func:`bessel_i` -- modified Bessel functions ``I_p(a)``, ``p = 0..P-1``:
  ``scipy.special.ive(p, a) * exp(|a|)`` on NumPy (the WT expression), the
  40-term power series of ``wdmtoysgpu.jaxlike.bessel_i`` on JAX (relative
  error <= 1e-13 for ``|a| <= 8``; the sinusoid priors need ``|a| <= 3.2``).
* :func:`chebyshev_basis` -- ``T_p(u_i)`` on a record's time grid
  (``wdm_toys.fastcov.cheb_basis``), host NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import Backend, get_backend

__all__ = ["bessel_i", "chebyshev_basis"]


def bessel_i(P: int, a: Any, be: Backend | None = None) -> Any:
    """``(P, len(a))``: ``I_p(a)`` for ``p = 0..P-1`` (delegates to ``be.bessel_i``)."""
    be = be or get_backend(a)
    return be.bessel_i(P, be.asarray(a))


def chebyshev_basis(N: int, P: int) -> NDArray[np.floating]:
    """Chebyshev polynomials ``T_p(u_i)``, ``(P, N)``, ``u_i = 2 (i/N - 1/2)``.

    ``wdm_toys.fastcov.cheb_basis`` verbatim (host constant).

    Examples
    --------
    >>> import wdm.special
    >>> wdm.special.chebyshev_basis(4, 3)
    array([[ 1. ,  1. ,  1. ,  1. ],
           [-1. , -0.5,  0. ,  0.5],
           [ 1. , -0.5, -1. , -0.5]])
    """
    u = 2.0 * (np.arange(N) / N - 0.5)
    T = np.empty((P, N))
    T[0] = 1.0
    if P > 1:
        T[1] = u
    for p in range(2, P):
        T[p] = 2 * u * T[p - 1] - T[p - 2]
    return T
