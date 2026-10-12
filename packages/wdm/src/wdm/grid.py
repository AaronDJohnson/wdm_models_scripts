"""Time-frequency grid: pixel layout, indexing, and active-pixel masks.

The WDM transform partitions the time-frequency plane into a grid of
$N_f+1$ frequency bands (indexed by $m = 0,\\ldots,N_f$) and $N_t$
time bins (indexed by $n = 0,\\ldots,N_t-1$).  The total time-domain
sample count is $N = N_f N_t$.

The pixel widths are
$$\\Delta T = N_f\\,\\delta t,\\qquad \\Delta F = 1/(2\\,\\delta t\\,N_f).$$

Edge bands ($m=0$ DC, $m=N_f$ Nyquist) carry only $N_t/2$ active
coefficients each (parity-selected); interior bands carry all $N_t$.
The total active count is therefore
$(N_f - 1)\\,N_t + N_t/2 + N_t/2 = N$.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.integer]

__all__ = [
    "Grid",
    "active_mask",
    "band_frequencies",
    "band_indices",
    "carrier_phase",
    "count_active",
    "delta_F",
    "delta_T",
    "delta_omega",
    "n_samples",
    "sanitize_psd",
    "time_pixel_centers",
    "validate_dt",
    "validate_finite",
    "validate_grid",
]


@dataclass(frozen=True)
class Grid:
    """A validated ``(Nf, Nt, dt)`` triple, with the derived quantities.

    Every function in the package takes ``Nf, Nt, dt`` positionally, and the
    one misuse that cannot be detected downstream is swapping ``Nf`` and
    ``Nt`` -- both are positive even ints, so ``Nf*Nt`` is unchanged and no
    validation can see it. Building a ``Grid`` once and unpacking it at each
    call site removes the opportunity.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> g = wdm.Grid(Nf=8, Nt=16, dt=1.0)
    >>> g.N, g.dF, g.dT
    (128, 0.0625, 8.0)
    >>> x = np.random.default_rng(0).standard_normal(g.N)
    >>> wdm.forward_time(x, *g).shape          # unpacks as (Nf, Nt, dt)
    (9, 16)
    >>> wdm.Grid(Nf=9, Nt=16, dt=1.0)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    """

    Nf: int
    Nt: int
    dt: float

    def __post_init__(self) -> None:
        validate_grid(self.Nf, self.Nt)
        validate_dt(self.dt)

    def __iter__(self) -> Iterator[int | float]:
        """Unpack as ``(Nf, Nt, dt)`` so ``f(x, *grid)`` just works."""
        yield self.Nf
        yield self.Nt
        yield self.dt

    @property
    def N(self) -> int:
        """Total time-domain sample count $N = N_f N_t$."""
        return n_samples(self.Nf, self.Nt)

    @property
    def dF(self) -> float:
        """Frequency-pixel width $\\Delta F$."""
        return delta_F(self.Nf, self.dt)

    @property
    def dT(self) -> float:
        """Time-pixel width $\\Delta T$."""
        return delta_T(self.Nf, self.dt)


def _require_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, raising ``ValueError`` if it is not one
    (``bool`` is rejected too).  Verbatim from lisa/wdm b5973b7."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _require_even_positive_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, requiring it to be positive and even.
    Verbatim from lisa/wdm b5973b7 (same messages as :func:`validate_grid`)."""
    val = _require_int(name, value)
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")
    if val % 2 != 0:
        raise ValueError(f"{name} must be even, got {val}")
    return val


def validate_grid(Nf: int, Nt: int) -> None:
    """Raise ``ValueError`` if ``Nf`` or ``Nt`` are not positive even integers.

    Parameters
    ----------
    Nf : int
        Number of frequency bands.
    Nt : int
        Number of time bins.

    Raises
    ------
    ValueError
        If either argument is not a positive even integer.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.validate_grid(8, 16)          # even and positive: fine
    >>> wdm.validate_grid(9, 16)
    Traceback (most recent call last):
    ValueError: Nf must be even, got 9
    """
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
        if val % 2 != 0:
            raise ValueError(f"{name} must be even, got {val}")


def validate_dt(dt: float) -> None:
    """Raise ``ValueError`` unless ``dt`` is a positive finite number.

    A negative or zero sampling interval produces a nonsensical frequency grid
    (``delta_F`` flips sign or diverges) without raising anywhere, so the
    transform happily returns numbers that mean nothing.

    Parameters
    ----------
    dt : float
        Sampling interval.

    Raises
    ------
    ValueError
        If ``dt`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.validate_dt(0.5)
    >>> wdm.validate_dt(0.0)
    Traceback (most recent call last):
    ValueError: dt must be a positive finite number, got 0.0
    """
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt!r}")


def validate_finite(values: object, context: str) -> None:
    """Raise ``ValueError`` if ``values`` holds a ``nan`` or an ``inf``.

    The WDM transform is *global* within a band: every coefficient is an
    inner product against a window spanning many samples, and the
    implementation routes each band through an FFT.  A single non-finite
    entry therefore does not stay local.  Measured on an $N_f=8$, $N_t=16$
    grid, one ``nan`` anywhere in the input series makes **all 144** output
    coefficients ``nan`` -- and numpy emits no warning on the way, because
    ``nan`` arithmetic inside an FFT is not an invalid operation.  The result
    is a coefficient grid that is entirely meaningless yet raises nothing,
    which is exactly the failure this rejects.

    Detector data with gaps routinely arrives with ``nan`` fills, so this is
    a live path rather than a theoretical one.  The check is one $O(N)$ pass
    against an $O(N\\log N_t)$ transform, so it is not measurable in the
    package's own benchmarks.

    Deliberately stricter than :func:`sanitize_psd`, which maps ``inf`` to
    ``0.0``: an infinite *PSD* sample has a physical reading ("no measurable
    power in this bin"), an infinite *data* sample has none.

    Parameters
    ----------
    values : array-like
        Data, coefficient, or spectrum array.  Complex input is checked on
        both components.
    context : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If any entry is ``nan`` or infinite.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.validate_finite([1.0, 2.0], 'x')      # all finite: fine
    >>> wdm.validate_finite([1.0, np.nan], 'x')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: x: 1 of 2 entries are not finite (1 nan, 0 inf); first at index 1...
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    first = int(np.flatnonzero(bad.ravel())[0])
    raise ValueError(
        f"{context}: {int(np.count_nonzero(bad))} of {arr.size} entries are "
        f"not finite ({n_nan} nan, {int(np.count_nonzero(bad)) - n_nan} inf); "
        f"first at index {first}. A non-finite sample does not stay local -- "
        "it spreads through the transform's FFT to every coefficient, "
        "silently. Fill or excise the gaps before transforming."
    )


def sanitize_psd(values: object, context: str) -> Array:
    """Apply the package-wide policy for non-finite PSD samples.

    Sensitivity curves routinely return ``inf`` at ``f = 0`` (a LISA
    acceleration term goes as $f^{-4}$), so an infinite sample is a normal
    input meaning "this bin carries no measurable power". Every module maps it
    to ``0.0``, which is what keeps :func:`wdm.matrix.build_C_time` finite and
    positive semi-definite.

    ``nan`` is *not* treated that way: it has no physical reading, so it raises.
    Complex input raises too rather than being silently truncated to its real
    part.

    The point of routing every module through here is **uniformity**. These
    paths previously disagreed: ``build_C_time`` and ``stationary`` mapped
    non-finite to ``0.0``, ``noise.band_averaged_variance`` dropped the bin
    from a weighted average (a different answer), and
    ``inner_product.loglike_gaussian`` propagated it to ``+inf``. The same PSD
    gave three answers depending on which function you called.

    Negative samples are rejected outright: a negative PSD is not a convention,
    it is a bug in the caller's model.

    Parameters
    ----------
    values : array-like
        PSD samples.
    context : str
        Name used in the error message.

    Returns
    -------
    ndarray, float
        Finite, non-negative PSD values with non-finite entries set to ``0.0``.

    Raises
    ------
    ValueError
        If any entry is negative, ``nan``, or complex.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.grid.sanitize_psd([1e-3, np.inf], 'demo')   # inf -> no power
    array([0.001, 0.   ])
    >>> wdm.grid.sanitize_psd([-1.0], 'demo')
    Traceback (most recent call last):
    ValueError: demo: PSD has negative value(s); a power spectral density must be non-negative
    """
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        raise ValueError(f"{context}: PSD must be real-valued, got a complex array")
    # Preserve single precision rather than forcing float64: a float32 PSD
    # paired with a float32 coefficient grid keeps the *whole* working set
    # single-precision. Forcing double here meant the variance grid stayed
    # float64 no matter what, so a caller in "float32 mode" got a working set
    # that was only ~33% smaller, not 50%. Everything else promotes.
    arr = arr.astype(np.float32 if arr.dtype == np.float32 else float)
    if np.any(np.isnan(arr)):
        # inf has a physical reading (no measurable power in that bin); nan does
        # not -- it means the caller's model produced something undefined.
        raise ValueError(
            f"{context}: PSD has {int(np.count_nonzero(np.isnan(arr)))} NaN "
            "value(s). Unlike inf, a NaN has no physical reading here; fix the "
            "PSD model rather than letting it become zero power."
        )
    if np.any(arr < 0.0):
        raise ValueError(
            f"{context}: PSD has negative value(s); a power spectral density "
            "must be non-negative"
        )
    return np.where(np.isfinite(arr), arr, 0.0)


def n_samples(Nf: int, Nt: int) -> int:
    """Total time-domain sample count $N = N_f N_t$.

    Parameters
    ----------
    Nf, Nt : int
        Frequency and time bin counts.

    Returns
    -------
    int
        $N = N_f \\cdot N_t$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.n_samples(8, 16)
    128
    """
    for name, val in (("Nf", Nf), ("Nt", Nt)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            # int() would silently truncate: n_samples(8.7, 16.2) == 128, and
            # bool is an int subclass, so n_samples(True, 16) would give 16.
            raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
    return int(Nf) * int(Nt)


def delta_F(Nf: int, dt: float) -> float:
    """Frequency-pixel width $\\Delta F = 1/(2\\,\\delta t\\,N_f)$.

    Parameters
    ----------
    Nf : int
        Number of frequency bands.
    dt : float
        Sampling interval.

    Returns
    -------
    float
        Pixel width in Hz.

    Raises
    ------
    ValueError
        If ``Nf`` is not a positive even integer, if ``dt`` is not positive
        and finite, or if $2\\,\\delta t N_f$ is so small or so large that the
        width is not a positive finite double (lisa/wdm b5973b7: an ``inf`` or
        ``0.0`` width made :func:`wdm.forward_time` return exact zeros).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.delta_F(8, 1.0)
    0.0625
    >>> wdm.delta_F(0, 1.0)
    Traceback (most recent call last):
    ValueError: Nf must be positive, got 0
    >>> wdm.delta_F(8, 5e-324)          # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: delta_F is not representable for Nf=8, dt=5e-324...
    """
    _require_even_positive_int("Nf", Nf)
    validate_dt(dt)
    width = 1.0 / (2.0 * dt * Nf)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(
            f"delta_F is not representable for Nf={Nf}, dt={dt!r}: "
            f"1/(2*dt*Nf) is {width!r}. Both arguments are individually legal, "
            "but their product has left the normal double range, and the "
            "transform would return a grid of exact zeros without raising."
        )
    return width


def delta_T(Nf: int, dt: float) -> float:
    """Time-pixel width $\\Delta T = N_f\\,\\delta t$.

    Parameters
    ----------
    Nf : int
    dt : float

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.delta_T(8, 1.0)
    8.0
    """
    validate_dt(dt)
    return float(Nf * dt)


def delta_omega(Nf: int, dt: float) -> float:
    """Angular-frequency-pixel width $\\Delta\\Omega = 2\\pi\\Delta F$.

    Parameters
    ----------
    Nf : int
    dt : float

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np, wdm
    >>> float(np.round(wdm.delta_omega(8, 1.0), 6))
    0.392699
    """
    return 2.0 * np.pi * delta_F(Nf, dt)


def band_frequencies(Nf: int, dt: float) -> Array:
    """Band-center frequencies $f_m = m\\,\\Delta F$ for $m = 0,\\ldots,N_f$.

    Parameters
    ----------
    Nf : int
    dt : float

    Returns
    -------
    ndarray, shape (Nf+1,)
        Endpoints satisfy ``freqs[0] == 0`` (DC) and
        ``freqs[Nf] == 1/(2*dt)`` (Nyquist).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> f = wdm.band_frequencies(8, 1.0)
    >>> f.shape, float(f[0]), float(f[-1])   # DC .. Nyquist
    ((9,), 0.0, 0.5)
    """
    validate_grid(Nf, 2)  # only Nf is constrained here; Nt is a placeholder
    return np.arange(Nf + 1) * delta_F(Nf, dt)


def time_pixel_centers(Nf: int, Nt: int, dt: float) -> Array:
    """Pixel-center times $t_n = n\\,\\Delta T$ for $n = 0,\\ldots,N_t-1$.

    These are the time coordinates of the WDM lattice: the energy centroid of
    the basis function $g_{mn}$ sits at exactly $n\\,\\Delta T$.

    Parameters
    ----------
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, shape (Nt,)

    Notes
    -----
    .. note::
       Before 2026-07-27 this returned $(n + 1/2)\\,\\Delta T$, a half-pixel
       too late. Measured against the energy centroid of a single-pixel
       inverse transform, the offset was exactly $-0.5\\,\\Delta T$ at every
       grid tested. The consumer inside the package is
       :func:`wdm.noise.evolutionary_variance`, which sampled a time-varying
       PSD half a pixel off. See ``src/wdm/summary.md``.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.time_pixel_centers(8, 4, 1.0)
    array([ 0.,  8., 16., 24.])
    """
    validate_grid(Nf, Nt)
    return np.arange(Nt) * delta_T(Nf, dt)


def band_indices(m: int, Nf: int, Nt: int) -> IntArray:
    """Circular DFT bin indices for band $m$ in FFT-natural ordering.

    Returns the $N_t$ indices $j$ that the forward transform extracts
    when computing band $m$ via ``np.fft.ifft``.  The offsets within
    the band run as $[0, 1, \\ldots, N_t/2-1, -N_t/2, \\ldots, -1]$
    relative to the band center $j_0 = m N_t/2$, with circular wrap
    by $N = N_f N_t$.

    Parameters
    ----------
    m : int
        Band index, $0 \\le m \\le N_f$.
    Nf, Nt : int

    Returns
    -------
    ndarray of int, shape (Nt,)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.grid.band_indices(1, 8, 4)
    array([2, 3, 0, 1])
    """
    N = n_samples(Nf, Nt)
    center = m * (Nt // 2)
    offsets = np.fft.ifftshift(np.arange(Nt) - Nt // 2)
    return (center + offsets) % N


def carrier_phase(m: int, Nt: int) -> Array:
    """Carrier phase $(-1)^{mn}$ as a length-$N_t$ array.

    Equals an array of ones when $m$ is even, and the alternating
    sequence $[1, -1, 1, -1, \\ldots]$ when $m$ is odd.

    Parameters
    ----------
    m : int
    Nt : int

    Returns
    -------
    ndarray, shape (Nt,)
        Float array of $\\pm 1$ values.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.grid.carrier_phase(1, 4)
    array([ 1., -1.,  1., -1.])
    """
    if m % 2 == 0:
        return np.ones(Nt)
    return 1.0 - 2.0 * (np.arange(Nt) % 2)


def active_mask(Nf: int, Nt: int) -> BoolArray:
    """Boolean mask of shape ``(Nf+1, Nt)`` marking active (nonzero) pixels.

    - Interior bands ($1 \\le m \\le N_f - 1$): all $N_t$ time bins active.
    - DC band ($m=0$): only even $n$ are active ($N_t/2$ pixels).
    - Nyquist band ($m=N_f$): only $n$ with $(N_f + n)$ even are active
      ($N_t/2$ pixels).

    Total active count equals $N = N_f N_t$.

    Parameters
    ----------
    Nf, Nt : int

    Returns
    -------
    ndarray of bool, shape (Nf+1, Nt)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> mask = wdm.active_mask(8, 16)
    >>> mask.shape, int(mask.sum())   # total active == N
    ((9, 16), 128)
    """
    validate_grid(Nf, Nt)
    mask = np.ones((Nf + 1, Nt), dtype=bool)
    n = np.arange(Nt)
    mask[0] = n % 2 == 0
    mask[Nf] = (Nf + n) % 2 == 0
    return mask


def count_active(Nf: int, Nt: int) -> int:
    """Total active WDM pixels.

    Always equals $N = N_f N_t$ for valid grids.

    Parameters
    ----------
    Nf, Nt : int

    Returns
    -------
    int

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.count_active(8, 16)
    128
    """
    return int(np.sum(active_mask(Nf, Nt)))
