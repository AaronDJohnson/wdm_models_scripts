"""Fast stationary-noise WDM covariance via the slab-IFFT algorithm.

For stationary Gaussian noise, the frequency-domain covariance is
diagonal: $\\langle X_j X_{j'}^*\\rangle = \\sigma_j^2\\,\\delta_{jj'}$
with $\\sigma_j^2 = N\\,S_n(|f_j|)/(2\\,\\delta t)$.  In the WDM basis,
the resulting covariance $\\Lambda = W\\,\\Sigma_{\\rm time}\\,W^T$ is
block-tridiagonal in the band index $m$ — the Meyer filter
$\\hat\\phi$ has compact frequency support so only $|m - m'|\\le 1$
band pairs overlap.

This module implements the slab-IFFT algorithm from
``notes/Time_Frequency_Analysis/wdm_Lambda_stationary_fast.tex`` (and
the within-block-symmetry refinement in
``wdm_Lambda_stationary_implementation.tex``).  For each band pair
$(m, m')$ with $|m - m'|\\le 1$, the block is filled by one length-$N_t$
IFFT plus a vectorised Toeplitz expansion with the carrier
$(-1)^{(m+1)(n+n')}$ and Wilson parity factor $C^*_{mn}\\,C_{m'n'}$.

Total cost: $\\mathcal O((N_f+1)\\,N_t\\,\\log N_t + N_f\\,N_t^2)$,
versus $\\mathcal O(N^3)$ for the dense $W\\,C_{\\rm time}\\,W^T$.

Two API tiers
-------------
:func:`build_lambda_blocks` (and its packed variant) cover the
**interior** bands $m \\in [1, N_f - 1]$.  Useful when one is content
to drop the edge contributions (typically a 10--100 ppm bias on the
matched-filter SNR$^2$, depending on how much signal sits near DC or
Nyquist).

:func:`build_lambda_blocks_full_active` covers **every** band pair,
including the parity-restricted edge bands $m \\in \\{0, N_f\\}$, via
half-slab Hermitian-extended IFFTs.  The full-active result matches
the dense $W_a\\,C_{\\rm time}\\,W_a^T$ at machine precision and is
preferred when bias-free inner products matter (matched filtering of
broadband signals, parameter estimation, anything that consumes
$\\Lambda^{-1}$).

Edge-band convention
--------------------
There is one coefficient convention (unit-norm basis functions; see
:mod:`wdm.transform`), so nothing here is parameterised by it.  The edge
$\\sqrt 2$ factors are baked into the builders that touch the edge bands;
:func:`build_lambda_blocks` and its packed variant cover interior bands only
and never see them.

Inputs
------
Every input form expresses the **one-sided PSD** $S_n(f)$ — the
internal scaling $\\sigma_k^2 = N\\,S_n(|f_k|)/(2\\,\\delta t)$ is
applied uniformly. Three accepted forms (all stationary):

* a callable ``Sn_func(f) -> S_n(f)``;
* a 1D ndarray of $S_n$ values, length ``N`` (two-sided ``fftfreq``
  grid) or ``N//2+1`` (one-sided ``rfftfreq`` grid);
* a 2D diagonal ndarray of shape ``(N, N)`` or ``(N//2+1, N//2+1)``
  with $S_n$ values on the diagonal. Off-diagonal entries are checked
  against ``rtol`` and a non-diagonal input raises ``ValueError`` (the
  fast algorithm requires stationarity).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from vendored_2f7136c.filter import phi_centered
from vendored_2f7136c.grid import n_samples, sanitize_psd, validate_dt, validate_grid

type Array = NDArray[np.floating]
type CArray = NDArray[np.complexfloating]
type PsdFunc = Callable[[Array], Array]
type NoiseInput = PsdFunc | Array | CArray

__all__ = [
    "apply_lambda_from_psd",
    "assemble_lambda",
    "assemble_lambda_full_active",
    "build_K_full_active",
    "build_lambda_blocks",
    "build_lambda_blocks_full_active",
    "build_lambda_blocks_packed",
    "is_diagonal",
    "lambda_to_psd",
    "lambda_to_psd_slab",
    "unpack_diag_block",
]


def is_diagonal(matrix: NDArray[np.number], *, rtol: float = 1e-10) -> bool:
    """Check whether ``matrix`` is diagonal up to a relative tolerance.

    Parameters
    ----------
    matrix : ndarray, shape (M, M)
    rtol : float, optional
        Off-diagonal entries are tolerated if their absolute value is at
        most ``rtol`` times the maximum absolute diagonal entry.
        Default ``1e-10``.

    Returns
    -------
    bool
        True if every off-diagonal entry satisfies the tolerance.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> wdm.stationary.is_diagonal(np.diag([1.0, 2.0]))
    True
    >>> wdm.stationary.is_diagonal(np.array([[1.0, 0.5], [0.5, 2.0]]))
    False
    """
    M = np.asarray(matrix)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"matrix must be square 2D; got shape {M.shape}")
    diag_vals = np.abs(np.diag(M))
    finite = diag_vals[np.isfinite(diag_vals)]
    diag_max = float(finite.max()) if finite.size else 0.0
    # Zero out diagonal in place (on a copy) to avoid inf-inf = nan.
    off = M.astype(np.result_type(M.dtype, float)).copy()
    np.fill_diagonal(off, 0.0)
    if diag_max == 0.0:
        return bool(np.max(np.abs(off)) == 0.0)
    return bool(np.max(np.abs(off)) <= rtol * diag_max)


def _mirror_to_two_sided(one_sided: Array, N: int) -> Array:
    """Expand a one-sided ``rfft``-grid PSD onto the full two-sided grid.

    Hermitian symmetry of a real process: $S(f_{N-k}) = S(f_{-k}) = S(|f_k|)$,
    so bins $1 \\ldots N/2-1$ mirror in reverse and DC and Nyquist -- which
    have no partner -- are left alone.

    Written out inline twice before (once for the 1D-array branch, once for
    the 2D-diagonal branch).  The two copies happened to agree, but the
    off-by-one at the Nyquist end is the kind of thing that gets fixed in one
    place and missed in the other.
    """
    n_rfft = N // 2 + 1
    out = np.empty(N, dtype=float)
    out[:n_rfft] = one_sided
    out[n_rfft:] = one_sided[1 : N // 2][::-1]
    return out


def _normalize_noise_to_sigma2(
    noise: NoiseInput,
    N: int,
    dt: float,
    *,
    rtol: float = 1e-10,
) -> Array:
    """Normalize a stationary-noise specification to a length-$N$ $\\sigma^2$ array.

    Every input form expresses the **one-sided PSD** $S_n(f)$ — never
    the already-scaled covariance. This function applies the single
    physical scaling $\\sigma_k^2 = N\\,S_n(|f_k|)/(2\\,\\delta t)$ to
    every form, so callers don't have to remember to do it themselves.

    Returns the per-bin variance $\\sigma_k^2$ on the full two-sided
    ``fftfreq(N, dt)`` grid.

    Parameters
    ----------
    noise : callable, 1D ndarray, or 2D ndarray
        See :func:`build_lambda_blocks` for accepted forms — all of
        which are interpreted as $S_n$ values (no internal scaling).
    N : int
        Total time-domain sample count.
    dt : float
        Sampling interval.
    rtol : float, optional
        Relative tolerance for the diagonality check on 2D inputs.

    Returns
    -------
    sigma2_full : ndarray, float, shape (N,)

    Raises
    ------
    ValueError
        If a 2D matrix input is not diagonal (i.e., the noise is not
        stationary), or if the input shape is not recognized.
    """
    if callable(noise):
        freqs_full = np.fft.fftfreq(N, d=dt)
        Sn_full = np.asarray(noise(np.abs(freqs_full)), dtype=float)
        if Sn_full.shape != freqs_full.shape:
            raise ValueError(
                f"noise callable must be elementwise: expected shape "
                f"{freqs_full.shape}, got {Sn_full.shape}"
            )
    else:
        arr = np.asarray(noise)
        n_rfft = N // 2 + 1

        if arr.ndim == 1:
            if arr.size == N:
                Sn_full = np.asarray(arr, dtype=float)
            elif arr.size == n_rfft:
                Sn_full = _mirror_to_two_sided(np.asarray(arr, dtype=float), N)
            else:
                raise ValueError(
                    f"1D PSD array must have length N={N} or N//2+1={n_rfft}; "
                    f"got {arr.size}"
                )
        elif arr.ndim == 2:
            if arr.shape not in ((N, N), (n_rfft, n_rfft)):
                raise ValueError(
                    f"2D PSD matrix must have shape ({N}, {N}) or "
                    f"({n_rfft}, {n_rfft}); got {arr.shape}"
                )
            if not is_diagonal(arr, rtol=rtol):
                raise ValueError(
                    "2D PSD matrix is not diagonal (input is non-stationary). "
                    "The fast stationary algorithm requires off-diagonal "
                    f"entries < {rtol:g} times the max diagonal entry. "
                    "For non-stationary noise build the dense pipeline "
                    "explicitly: W = wdm.matrix.build_W(...) (or active_W), "
                    "then W @ C @ W.T with your own (N, N) covariance. Note "
                    "wdm.matrix.build_C_wdm does NOT accept one -- it takes a "
                    "stationary PSD callable and builds a Toeplitz C_time."
                )
            diag = np.real(np.diag(arr)).astype(float)
            Sn_full = diag if arr.shape[0] == N else _mirror_to_two_sided(diag, N)
        else:
            raise ValueError(
                f"noise must be a callable or a 1D / 2D ndarray; got ndim={arr.ndim}"
            )

    Sn_full = sanitize_psd(Sn_full, "stationary noise input")
    return N * Sn_full / (2.0 * dt)


def _carriers(Nt: int) -> dict[tuple[int, int], CArray]:
    """The four Wilson parity/carrier outer products, keyed by ``(m_l%2, m_r%2)``.

    ``carrier[n, n'] = a_n(m_l) * C*_{m_l,n} * a_{n'}(m_l) * C_{m_r,n'}`` where
    ``a_n(m) = (-1)^{(m+1)n}`` and ``C_{m,n} = i^{(m+n) mod 2}``.  It depends on
    the band indices only through their parities, so all ``2 N_f + 1`` blocks
    are covered by four matrices.

    This lives at module level on purpose: it is the single most error-prone
    convention in the file, and it used to be duplicated verbatim inside three
    different builders, where fixing a sign in one left the other two wrong.
    """
    n_arr = np.arange(Nt)

    def build(m_l: int, m_r: int) -> CArray:
        a_n = np.where(n_arr % 2 == 0, 1.0, (-1.0) ** (m_l + 1))
        c_left = np.where((m_l + n_arr) % 2 == 0, 1.0 + 0j, -1.0j)
        c_right = np.where((m_r + n_arr) % 2 == 0, 1.0 + 0j, 1.0j)
        return (a_n * c_left)[:, None] * (a_n * c_right)[None, :]

    return {(a, b): build(a, b) for a in (0, 1) for b in (0, 1)}


def _interior_slab_K(
    sigma2: Array, phi_c: Array, Nf: int, Nt: int
) -> tuple[CArray, CArray]:
    """Batched interior-band slab IFFTs: ``(K_diag[1:Nf], K_off[1:Nf-1])``.

    The interior slab for band ``m`` starts at ``(m-1)*Nt/2`` and is a pure
    gather, so all ``Nf-1`` bands go through one length-``Nt`` IFFT.  This is
    the single definition of that recipe; it used to appear verbatim in three
    builders.

    ``sigma2`` is the one-sided ``sigma2_rfft = sigma2_full / N`` grid, the
    module's single convention (see :func:`_interior_setup`).  Interior slabs
    index at most ``N//2 - 1``, so the one-sided array is always long enough.
    """
    half = Nt // 2
    phi_shift = np.roll(phi_c, half)
    n_arr = np.arange(Nt)
    ell_starts = (np.arange(1, Nf) - 1) * half
    slab = sigma2[ell_starts[:, None] + n_arr[None, :]]

    K_diag = Nt * np.fft.ifft(phi_c[None, :] ** 2 * slab, axis=-1)
    V_off = slab[:-1] * phi_c[None, :] * phi_shift[None, :]
    V_off[:, :half] = 0
    K_off = Nt * np.fft.ifft(V_off, axis=-1)
    return K_diag, K_off


def _block_from_K_with_carrier(
    K: CArray, carrier: CArray, diff: NDArray[np.integer], pref: float
) -> Array:
    """Block expansion given a precomputed outer-product carrier matrix.

    The carrier $\\mathrm{carrier}[n, n'] = v_l[n]\\,v_r[n']$ encodes the
    band-dependent sign + Wilson parity factors. Building it once per
    distinct $(m_l\\%2, m_r\\%2)$ pattern (only 4 patterns total across
    all band pairs) and reusing it for every block of that pattern
    halves the wall-clock cost: one fused element-wise complex multiply
    plus one $\\Re$ instead of two broadcasts.
    """
    return pref * np.real(carrier * K[diff])


def _block_upper_from_K(
    K: CArray,
    carrier: CArray,
    diff_iu: NDArray[np.integer],
    pref: float,
    iu_n: NDArray[np.integer],
    iu_np: NDArray[np.integer],
) -> Array:
    """Upper-triangular entries of the symmetric diagonal block $\\Lambda^{(m,m)}$.

    The triangular counterpart of :func:`_block_from_K_with_carrier`, gathering
    only the ``np.triu_indices`` entries.

    Takes ``carrier`` and ``diff_iu`` precomputed by the caller.  Both depend
    only on ``Nt`` and the parity of $m$, so deriving them here -- which meant
    building all four $N_t \\times N_t$ carrier matrices and re-deriving the
    index difference once per band -- repeated $O(N_t^2)$ work $N_f - 1$ times
    to get four distinct answers.
    """
    return pref * np.real(carrier[iu_n, iu_np] * K[diff_iu])


def _interior_setup(
    Nf: int, Nt: int, dt: float, noise: NoiseInput, rtol: float
) -> tuple[CArray, CArray, float, NDArray[np.integer], dict[tuple[int, int], CArray]]:
    """Shared preamble of the two interior-band builders.

    Returns ``(K_diag, K_off, pref, diff, carriers)``.

    Both tiers now use the **same** prefactor, ``2*pi/N`` against
    ``sigma2_rfft = sigma2_full / N``.  The interior tier used to carry a
    second, algebraically identical form -- ``alpha**2 / (2 Nt**2)`` with
    ``alpha = 2*sqrt(pi)/Nf``, applied to the unnormalised ``sigma2_full`` --
    which differed only in where the ``1/N`` lived:

        alpha**2 / 2 / Nt**2  ==  (4 pi / Nf**2) / 2 / Nt**2  ==  2 pi / N**2
        (2 pi / N) * (1 / N)  ==  2 pi / N**2

    Two spellings of one number is a standing invitation to "fix" one of them.
    They are not bit-identical in floating point (they group the divisions
    differently, ~1e-16 apart), which is precisely why having both was worse
    than useless: a reader comparing the tiers could not tell whether a
    discrepancy was a bug or rounding.
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    sigma2_full = _normalize_noise_to_sigma2(noise, N, dt, rtol=rtol)
    sigma2_rfft = sigma2_full[: N // 2 + 1] / N
    phi_c = phi_centered(Nf, Nt, dt)

    pref = 2.0 * np.pi / N
    n_arr = np.arange(Nt)
    diff = (n_arr[:, None] - n_arr[None, :]) % Nt

    K_diag, K_off = _interior_slab_K(sigma2_rfft, phi_c, Nf, Nt)
    return K_diag, K_off, pref, diff, _carriers(Nt)


def build_lambda_blocks(
    Nf: int,
    Nt: int,
    dt: float,
    noise: NoiseInput,
    *,
    rtol: float = 1e-10,
) -> tuple[list[Array], list[Array]]:
    """Block-tridiagonal stationary $\\Lambda = W\\,\\Sigma\\,W^T$ via slab-IFFT.

    For each interior band pair $(m, m')$ with $1\\le m,m'\\le N_f - 1$
    and $|m - m'|\\le 1$, builds the dense $N_t\\times N_t$ block
    using one length-$N_t$ IFFT plus a vectorised Toeplitz expansion.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    noise : callable, 1D ndarray, or 2D ndarray
        Stationary noise specification — every form is the **one-sided
        PSD** $S_n(f)$. The package applies the $N/(2\\,\\delta t)$
        scaling internally; do not pre-scale.

        * ``Sn_func(f) -> S_n(f)``: one-sided PSD function.
        * 1D ndarray of length ``N`` or ``N//2+1``: $S_n$ values on the
          two-sided ``fftfreq`` grid or one-sided ``rfftfreq`` grid.
        * 2D diagonal ndarray of shape ``(N, N)`` or ``(N//2+1, N//2+1)``
          with $S_n$ values on the diagonal. Diagonality is enforced
          (see ``rtol``); non-diagonal input raises ``ValueError``.
    rtol : float, optional
        Relative tolerance for the diagonality check on 2D inputs.
        Off-diagonal entries are tolerated if their absolute value is
        at most ``rtol`` times the maximum absolute diagonal entry.
        Default ``1e-10``.

    Returns
    -------
    diag : list of ($N_f - 1$) ndarrays, shape ($N_t$, $N_t$)
        ``diag[i]`` is $\\Lambda^{(m,m)}$ for $m = i + 1$.
    offd : list of ($N_f - 2$) ndarrays, shape ($N_t$, $N_t$)
        ``offd[i]`` is $\\Lambda^{(m,m+1)}$ for $m = i + 1$.

    Raises
    ------
    ValueError
        If a 2D input matrix is not diagonal, or if a 1D input has
        unrecognized length.

    Notes
    -----
    Edge bands ($m \\in \\{0, N_f\\}$) are not covered by the fast
    algorithm; their complementary-covariance contribution is small
    and typically dropped (see module docstring).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> diag, offd = wdm.stationary.build_lambda_blocks(8, 16, 1.0, flat)
    >>> len(diag), len(offd), diag[0].shape
    (7, 6, (16, 16))
    """
    K_diag_all, K_off_all, pref, diff, carrier = _interior_setup(
        Nf, Nt, dt, noise, rtol
    )
    carrier_diag = (carrier[0, 0], carrier[1, 1])  # by m parity
    carrier_off = (carrier[0, 1], carrier[1, 0])  # by m_left parity
    m_arr = np.arange(1, Nf)

    # Block expansion stays in a Python loop: one (Nt, Nt) buffer per
    # iteration is cache-friendly, while a fully batched (Nf, Nt, Nt)
    # intermediate would blow the L2/L3 cache and be slower.
    diag: list[Array] = []
    offd: list[Array] = []
    for i, m in enumerate(m_arr):
        diag.append(
            _block_from_K_with_carrier(K_diag_all[i], carrier_diag[m % 2], diff, pref)
        )
        if m < Nf - 1:
            offd.append(
                _block_from_K_with_carrier(K_off_all[i], carrier_off[m % 2], diff, pref)
            )

    return diag, offd


def build_lambda_blocks_packed(
    Nf: int,
    Nt: int,
    dt: float,
    noise: NoiseInput,
    *,
    rtol: float = 1e-10,
) -> tuple[list[Array], list[Array]]:
    """Same as :func:`build_lambda_blocks`, but with packed-upper diagonal blocks.

    Each diagonal block $\\Lambda^{(m,m)}$ is symmetric in $(n, n')$,
    so only the $N_t(N_t + 1)/2$ upper-triangular entries are stored.
    Off-diagonal blocks are returned full.  Roughly $25\\%$ less
    **storage** than :func:`build_lambda_blocks`; it is not faster --
    the triangular fancy-index gather costs more than the full
    circulant expansion, so this trades runtime for the memory saving.
    Currently ~1.2-1.3x slower (it was several times slower before the
    per-band carrier rebuild was hoisted out of :func:`_block_upper_from_K`).
    Run ``python tools/benchmark.py`` for the figure on your machine; choose
    this variant for the storage, not for speed.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    noise : callable, 1D ndarray, or 2D ndarray
        See :func:`build_lambda_blocks`.
    rtol : float, optional

    Returns
    -------
    diag_packed : list of ($N_f - 1$) ndarrays, shape ($N_t(N_t+1)/2$,)
        ``np.triu_indices``-order packing of $\\Lambda^{(m,m)}$.
        Use :func:`unpack_diag_block` to materialize a full block.
    offd : list of ($N_f - 2$) ndarrays, shape ($N_t$, $N_t$)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> packed, offd = wdm.stationary.build_lambda_blocks_packed(
    ...     8, 16, 1.0, flat
    ... )
    >>> packed[0].shape   # Nt*(Nt+1)/2 upper-triangular entries
    (136,)
    """
    K_diag_all, K_off_all, pref, diff_off, carrier = _interior_setup(
        Nf, Nt, dt, noise, rtol
    )
    carrier_diag = (carrier[0, 0], carrier[1, 1])  # by m parity
    carrier_off = (carrier[0, 1], carrier[1, 0])
    iu_n, iu_np = np.triu_indices(Nt)
    diff_iu = (iu_n - iu_np) % Nt
    m_arr = np.arange(1, Nf)

    diag_packed: list[Array] = []
    offd: list[Array] = []
    for i, m in enumerate(m_arr):
        diag_packed.append(
            _block_upper_from_K(
                K_diag_all[i], carrier_diag[m % 2], diff_iu, pref, iu_n, iu_np
            )
        )
        if m < Nf - 1:
            offd.append(
                _block_from_K_with_carrier(
                    K_off_all[i], carrier_off[m % 2], diff_off, pref
                )
            )

    return diag_packed, offd


def unpack_diag_block(packed: Array, Nt: int) -> Array:
    """Reconstruct a full $N_t\\times N_t$ symmetric matrix from a packed-upper vector.

    Parameters
    ----------
    packed : ndarray, shape ($N_t(N_t+1)/2$,)
        Upper-triangular entries in ``np.triu_indices`` order.
    Nt : int

    Returns
    -------
    ndarray, shape ($N_t$, $N_t$)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> packed, _ = wdm.stationary.build_lambda_blocks_packed(8, 16, 1.0, flat)
    >>> block = wdm.stationary.unpack_diag_block(packed[0], 16)
    >>> block.shape, bool(np.allclose(block, block.T))
    ((16, 16), True)
    """
    block = np.zeros((Nt, Nt), dtype=packed.dtype)
    iu = np.triu_indices(Nt)
    block[iu] = packed
    diag = np.diag(np.diag(block))
    return block + block.T - diag


def assemble_lambda(
    diag: list[Array],
    offd: list[Array],
    Nf: int,
    Nt: int,
) -> Array:
    """Assemble the interior block-tridiagonal $\\Lambda$ into a dense $(N_f+1)N_t$ matrix.

    Edge-band rows/columns ($m = 0, N_f$) are zero; only the interior
    bands $m \\in [1, N_f - 1]$ are populated.  The lower off-diagonal
    is filled by transposition: $\\Lambda^{(m+1,m)} = (\\Lambda^{(m,m+1)})^T$.

    Parameters
    ----------
    diag : list of ($N_f - 1$) ndarrays, shape ($N_t$, $N_t$)
    offd : list of ($N_f - 2$) ndarrays, shape ($N_t$, $N_t$)
    Nf, Nt : int

    Returns
    -------
    ndarray, shape ($(N_f+1) N_t$, $(N_f+1) N_t$)

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> diag, offd = wdm.stationary.build_lambda_blocks(8, 16, 1.0, flat)
    >>> wdm.stationary.assemble_lambda(diag, offd, 8, 16).shape
    (144, 144)
    """
    validate_grid(Nf, Nt)
    if len(diag) != Nf - 1:
        raise ValueError(
            f"diag must have Nf-1 = {Nf - 1} interior blocks; got {len(diag)}"
        )
    if len(offd) != Nf - 2:
        raise ValueError(f"offd must have Nf-2 = {Nf - 2} blocks; got {len(offd)}")
    n_total = (Nf + 1) * Nt
    out = np.zeros((n_total, n_total))
    for k, m in enumerate(range(1, Nf)):
        s = m * Nt
        out[s : s + Nt, s : s + Nt] = diag[k]
    for k, m in enumerate(range(1, Nf - 1)):
        s = m * Nt
        out[s : s + Nt, s + Nt : s + 2 * Nt] = offd[k]
        out[s + Nt : s + 2 * Nt, s : s + Nt] = offd[k].T
    return out


# ---------------------------------------------------------------------------
# Full-active slab-IFFT (interior + edges) — unified prefactor 2π/N
# ---------------------------------------------------------------------------
#
# For edge bands m ∈ {0, N_f}, the basis function $\hat g_{m,n}$ has
# rfft support on the half-slab adjacent to DC (m=0) or Nyquist (m=N_f),
# of width ~N_t/2 vs N_t for interior. The slab-IFFT formula below
# reproduces $\Lambda^{(m,m')}$ for any band pair |m-m'| ≤ 1 in a
# unified form, with edge √2 factors absorbed into the slab vector V
# (cross blocks) or via Hermitian-extending a half-slab of width N_t/2
# (diagonal blocks). After the block expansion, edge rows/columns are
# parity-restricted to (m+n) even.


def _slab_K(
    Nf: int, Nt: int, dt: float, sigma2_rfft: Array
) -> tuple[list[CArray], list[CArray]]:
    """Toeplitz-coefficient vectors $K^{(m,m)}$ and $K^{(m,m+1)}$ for every band.

    This is the single definition of the slab-IFFT recipe.  Both
    :func:`build_lambda_blocks_full_active` (which expands these into dense
    blocks) and :func:`build_K_full_active` (which returns them as-is) go
    through here; the two used to carry byte-identical copies of the five
    stanzas below, so a slab or index fix had to land twice.

    Interior bands read a full length-$N_t$ slab; the two edge bands read a
    half-slab and Hermitian-extend it, and the edge cross terms pick up the
    $\\sqrt 2$ from the unit-norm edge rescaling.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    sigma2_rfft : ndarray
        Per-bin variance on the one-sided grid, already divided by $N$.

    Returns
    -------
    K_diag : list of $N_f+1$ complex arrays, each shape $(N_t,)$
    K_off : list of $N_f$ complex arrays, each shape $(N_t,)$
    """
    N = n_samples(Nf, Nt)
    half = Nt // 2
    phi_c = phi_centered(Nf, Nt, dt)

    K_diag: list[CArray] = [np.zeros(Nt, dtype=complex) for _ in range(Nf + 1)]
    K_off: list[CArray] = [np.zeros(Nt, dtype=complex) for _ in range(Nf)]

    # Edge diag at m = 0: half-slab below DC, Hermitian-extended about 0.
    V = np.zeros(Nt)
    V[:half] = phi_c[half:] ** 2 * sigma2_rfft[:half]
    V[half + 1 :] = V[1:half][::-1]
    K_diag[0] = Nt * np.fft.ifft(V)

    # Edge cross (0, 1).
    V = np.zeros(Nt)
    V[half:] = np.sqrt(2.0) * phi_c[half:] * phi_c[:half] * sigma2_rfft[:half]
    K_off[0] = Nt * np.fft.ifft(V)

    # Interior bands m = 1 .. Nf-1, in one batched IFFT.
    m_arr = np.arange(1, Nf)
    K_diag_int, K_off_int = _interior_slab_K(sigma2_rfft, phi_c, Nf, Nt)
    for i, m in enumerate(m_arr):
        K_diag[m] = K_diag_int[i]
        if m < Nf - 1:
            K_off[m] = K_off_int[i]

    # Edge cross (Nf-1, Nf).
    V = np.zeros(Nt)
    V[half:] = (
        np.sqrt(2.0) * phi_c[half:] * phi_c[:half] * sigma2_rfft[N // 2 - half : N // 2]
    )
    K_off[Nf - 1] = Nt * np.fft.ifft(V)

    # Edge diag at m = Nf: half-slab up to Nyquist inclusive, Hermitian-extended.
    V = np.zeros(Nt)
    V[: half + 1] = phi_c[: half + 1] ** 2 * sigma2_rfft[N // 2 - half : N // 2 + 1]
    V[half + 1 :] = V[1:half][::-1]
    K_diag[Nf] = Nt * np.fft.ifft(V)

    return K_diag, K_off


def _active_indices(m: int, Nf: int, Nt: int) -> NDArray[np.integer]:
    """Active $n$ indices in band $m$. Edges keep only $(m+n)$ even."""
    if m in (0, Nf):
        return np.array([n for n in range(Nt) if (m + n) % 2 == 0])
    return np.arange(Nt)


def build_lambda_blocks_full_active(
    Nf: int,
    Nt: int,
    dt: float,
    noise: NoiseInput,
    *,
    rtol: float = 1e-10,
) -> tuple[list[Array], list[Array], list[NDArray[np.integer]]]:
    r"""Stationary $\Lambda$ for the **full** active subspace, including edge bands.

    Returns variable-size diagonal and off-diagonal blocks, plus the
    per-band active-$n$ index lists.  Edge bands $m \in \{0, N_f\}$ are
    parity-restricted to half size $N_t/2$; interior bands are full
    $N_t$.  Off-diagonals between an edge and its interior neighbour
    have the edge dimension restricted accordingly.

    The slab-IFFT recipe per band pair $(m, m')$:

    - **Interior diag** ($1 \le m \le N_f-1$, $m'=m$):
      $V[k] = \hat\phi[k]^2\,\sigma^2_{\rm rfft}[k + (m-1)N_t/2]$,
      $K = N_t \cdot \mathrm{IFFT}(V)$.

    - **Interior offd** ($1 \le m \le N_f-2$, $m'=m+1$):
      $V[k] = \hat\phi[k]\,\hat\phi[k - N_t/2]\,\sigma^2_{\rm rfft}[k + (m-1)N_t/2]$
      for $k \in [N_t/2, N_t-1]$, else 0.

    - **Edge diag at m=0**: $V[k] = \hat\phi[k+N_t/2]^2\,\sigma^2_{\rm rfft}[k]$
      for $k \in [0, N_t/2)$, then Hermitian-extended around 0 with $V[N_t/2]=0$.

    - **Edge diag at m=N_f**: $V[k] = \hat\phi[k]^2\,\sigma^2_{\rm rfft}[k + N/2 - N_t/2]$
      for $k \in [0, N_t/2]$ (Nyquist included), then Hermitian-extended.

    - **Edge cross (0,1) and (N_f-1, N_f)**: same product
      $\sqrt 2\,\hat\phi[k]\,\hat\phi[k - N_t/2]\,\sigma^2$ on the upper half
      $k \in [N_t/2, N_t-1]$, with the appropriate slab indexing.

    All blocks share the same prefactor $2\pi/N$ and the same
    block-from-K expansion (carrier $(-1)^{(m+1)(n+n')}$, Wilson parity
    $C^*_{m,n} C_{m',n'}$ with $C_{m,n} = i^{(m+n)\bmod 2}$).

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    noise : callable, 1D ndarray, or 2D ndarray
        See :func:`build_lambda_blocks`.
    rtol : float, optional

    Returns
    -------
    diag_blocks : list of ($N_f + 1$) ndarrays
        Diagonal blocks. Edges have shape ($N_t/2$, $N_t/2$); interior
        have shape ($N_t$, $N_t$).
    offd_blocks : list of $N_f$ ndarrays
        Upper off-diagonal blocks $\Lambda^{(m, m+1)}$. The first and
        last have an edge-restricted dimension; the middle $N_f - 2$
        are full ($N_t$, $N_t$).
    n_active : list of ($N_f + 1$) integer arrays
        Active-$n$ index arrays per band; edge entries have length
        $N_t/2$, interior entries are $\arange(N_t)$.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> psd = lambda f: 1e-3 * np.ones_like(f)
    >>> diag, offd, n_active = wdm.stationary.build_lambda_blocks_full_active(
    ...     Nf, Nt, dt, psd
    ... )
    >>> len(diag), len(offd)
    (9, 8)
    >>> diag[0].shape, diag[1].shape          # edge band, then interior
    ((8, 8), (16, 16))
    >>> Lam = wdm.stationary.assemble_lambda_full_active(
    ...     diag, offd, n_active, Nf, Nt
    ... )
    >>> Lam.shape
    (144, 144)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    sigma2_full = _normalize_noise_to_sigma2(noise, N, dt, rtol=rtol)
    sigma2_rfft = sigma2_full[: N // 2 + 1] / N

    pref = 2.0 * np.pi / N
    n_arr = np.arange(Nt)
    diff = (n_arr[:, None] - n_arr[None, :]) % Nt

    carrier = _carriers(Nt)

    def _diag_carrier(m: int | np.integer) -> CArray:
        return carrier[int(m) % 2, int(m) % 2]

    def _off_carrier(m_l: int | np.integer) -> CArray:
        return carrier[int(m_l) % 2, (int(m_l) + 1) % 2]

    n_active = [_active_indices(m, Nf, Nt) for m in range(Nf + 1)]
    K_diag, K_off = _slab_K(Nf, Nt, dt, sigma2_rfft)

    diag_blocks: list[Array] = []
    for m in range(Nf + 1):
        block = _block_from_K_with_carrier(K_diag[m], _diag_carrier(m), diff, pref)
        if m in (0, Nf):  # parity-restricted on both sides
            block = block[np.ix_(n_active[m], n_active[m])]
        diag_blocks.append(block)

    offd_blocks: list[Array] = []
    for m in range(Nf):
        block = _block_from_K_with_carrier(K_off[m], _off_carrier(m), diff, pref)
        if m == 0:  # restricted on the row (edge) side only
            block = block[n_active[0], :]
        elif m == Nf - 1:  # restricted on the column (edge) side only
            block = block[:, n_active[Nf]]
        offd_blocks.append(block)

    return diag_blocks, offd_blocks, n_active


def _check_block_layout(
    diag_blocks: list[Array],
    offd_blocks: list[Array],
    n_active: list[NDArray[np.integer]],
    Nf: int,
) -> None:
    """Validate a block-tridiagonal layout against the grid it claims to be on.

    Without this a list that is short by one silently produces a plausible but
    badly wrong answer (measured: a PSD off by 110%) rather than raising, which
    is the one place in the package where that was still possible.
    """
    if len(diag_blocks) != Nf + 1:
        raise ValueError(
            f"diag_blocks must have Nf+1 = {Nf + 1} entries; got {len(diag_blocks)}"
        )
    if len(offd_blocks) != Nf:
        raise ValueError(
            f"offd_blocks must have Nf = {Nf} entries; got {len(offd_blocks)}"
        )
    if len(n_active) != Nf + 1:
        raise ValueError(
            f"n_active must have Nf+1 = {Nf + 1} entries; got {len(n_active)}"
        )
    for m, block in enumerate(diag_blocks):
        want = (n_active[m].size, n_active[m].size)
        if block.shape != want:
            raise ValueError(
                f"diag_blocks[{m}] must have shape {want} to match "
                f"n_active[{m}]; got {block.shape}"
            )
    for m, block in enumerate(offd_blocks):
        rows = n_active[m].size
        cols = n_active[m + 1].size
        if block.shape != (rows, cols):
            raise ValueError(
                f"offd_blocks[{m}] must have shape {(rows, cols)} to match "
                f"n_active[{m}] x n_active[{m + 1}]; got {block.shape}"
            )


def _block_tridiag_matvec(
    diag_blocks: list[Array],
    offd_blocks: list[Array],
    n_active: list[NDArray[np.integer]],
    v_2d: Array,
) -> Array:
    r"""Compute $u = \Lambda v$ for block-tridiagonal $\Lambda$ stored as
    variable-size diagonal and off-diagonal blocks (the ``..._full_active``
    layout).  Inputs and outputs are dense $(N_f+1) \times N_t$ arrays; the
    inactive edge-band entries of $v$ are ignored on read and stay zero on write.
    """
    Nbp1 = len(diag_blocks)
    u_2d = np.zeros_like(v_2d)

    for m in range(Nbp1):
        idx = n_active[m]
        u_2d[m, idx] += diag_blocks[m] @ v_2d[m, idx]

    for m in range(Nbp1 - 1):
        offd = offd_blocks[m]
        v_m = v_2d[m, n_active[m]]
        v_mp1 = v_2d[m + 1, n_active[m + 1]]
        u_2d[m, n_active[m]] += offd @ v_mp1
        u_2d[m + 1, n_active[m + 1]] += offd.T @ v_m

    return u_2d


def build_K_full_active(
    Nf: int,
    Nt: int,
    dt: float,
    noise: NoiseInput,
    *,
    rtol: float = 1e-10,
) -> tuple[list[CArray], list[CArray]]:
    r"""Build the $K$ vectors that parameterise the stationary $\Lambda$.

    Each band pair $(m, m')$ with $|m - m'| \le 1$ carries a single
    length-$N_t$ complex Toeplitz-coefficient vector $K^{(m,m')}$ from
    which the dense block is reconstructible via the slab-IFFT recipe.
    Storage is $\sim 2 N_f N_t$ complex numbers, an $N_t$-fold reduction
    over storing the full $N_t \times N_t$ blocks.

    Mathematically (interior diagonal):
    $K^{(m,m)} = N_t \cdot \mathrm{ifft}(|\hat\phi_c|^2 \odot \sigma^2_{\rm rfft}[\mathrm{slab}_m])$,
    where slab$_m = [(m-1)N_t/2, (m-1)N_t/2 + N_t)$. Edge bands use the
    Hermitian-extended half-slab. Off-diagonals use the cross-band
    window product $\hat\phi_c \cdot \hat\phi_{c, \rm shift}$.

    Parameters
    ----------
    Nf, Nt : int
    dt : float
    noise : callable, 1D ndarray, or 2D ndarray
        Same accepted forms as :func:`build_lambda_blocks_full_active`.
    rtol : float, optional

    Returns
    -------
    K_diag : list of $N_f + 1$ ndarrays, each shape $(N_t,)$, complex
    K_off  : list of $N_f$ ndarrays, each shape $(N_t,)$, complex

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> K_diag, K_off = wdm.stationary.build_K_full_active(8, 16, 1.0, flat)
    >>> len(K_diag), len(K_off), K_diag[0].shape
    (9, 8, (16,))
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    N = n_samples(Nf, Nt)
    sigma2_full = _normalize_noise_to_sigma2(noise, N, dt, rtol=rtol)
    sigma2_rfft = sigma2_full[: N // 2 + 1] / N
    return _slab_K(Nf, Nt, dt, sigma2_rfft)


def lambda_to_psd_slab(
    K_diag: list[CArray],
    Nf: int,
    Nt: int,
    dt: float,
) -> Array:
    r"""Slab-extraction backward: recover $\mathcal{S}_n$ from the
    diagonal $K$ vectors.

    For each band $m$, undo the slab-IFFT:
    $V_m = \mathrm{fft}(K^{(m,m)})/N_t = |\hat\phi_c|^2 \odot \sigma^2_{\rm rfft}[\mathrm{slab}_m]$,
    then divide by the window to extract $\sigma^2_{\rm rfft}[\mathrm{slab}_m]$.
    Stitch the slabs into a single rfft-length vector by assigning each
    bin to the band whose window value is largest there (which is
    numerically the most stable estimate).

    Edge bands $m \in \{0, N_f\}$ use the half-slab windows
    $\hat\phi_c[k + N_t/2]^2$ and $\hat\phi_c[k]^2$ respectively, matching
    the forward of :func:`build_K_full_active`.

    Parameters
    ----------
    K_diag : list of $N_f + 1$ ndarrays, each shape $(N_t,)$, complex
        As produced by :func:`build_K_full_active`.
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape $(N/2 + 1,)$
        One-sided PSD evaluated on the ``np.fft.rfftfreq(N, dt)`` grid.

    Notes
    -----
    Cost: one length-$N_t$ FFT per band + bin assignment scan. Total
    $O(N \log N_t)$ — same as the forward :func:`build_K_full_active`
    and as :func:`apply_lambda_from_psd`. Compared with
    :func:`lambda_to_psd` (operator form on stored blocks), this avoids
    the $O(N_f N_t^2)$ block-tridiag matvec.  A few times faster than
    :func:`lambda_to_psd` in practice -- short of the $N_t/\log N_t$ the
    asymptotics alone suggest, because both routines are dominated by their
    FFTs at these sizes rather than by the matvec.  Run
    ``python tools/benchmark.py`` for the figure on your machine.

    **Why the three band groups are not one loop.** The body handles DC, the
    interior bands and Nyquist separately, which looks like avoidable
    duplication and is not: :func:`_slab_K` packs each band's slab into $V$
    differently, so the map from a $V$ index to its window index differs.

    ==========  =====================================  ====================
    band        built by :func:`_slab_K` as             $V$ index $j$ pairs
    ==========  =====================================  ====================
    $m=0$       ``V[:half]  = phi[half:]**2 * s2[...]``  ``phi2[half + j]``
    $1..N_f-1$  ``V[k]      = phi[k]**2 * s2[start+k]``  ``phi2[j]``
    $m=N_f$     ``V[:half+1]= phi[:half+1]**2 * s2[.]``  ``phi2[j]``
    ==========  =====================================  ====================

    Only DC carries the ``+half`` offset, because its half-slab lies *below*
    DC and is Hermitian-extended about 0 -- it is packed from ``V[0]`` upward
    while its window comes from the upper half of $\hat\phi$.  A unified loop
    using one index for both silently uses the wrong window on band 0; that
    was tried, and
    ``test_lambda_to_psd_slab_dc_band_uses_the_offset_window`` now fails when
    it is tried again.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> K_diag, _ = wdm.stationary.build_K_full_active(8, 16, 1.0, flat)
    >>> wdm.stationary.lambda_to_psd_slab(K_diag, 8, 16, 1.0).shape
    (65,)
    """
    validate_grid(Nf, Nt)
    validate_dt(dt)
    if len(K_diag) != Nf + 1:
        raise ValueError(f"K_diag must have Nf+1 = {Nf + 1} entries; got {len(K_diag)}")
    N = n_samples(Nf, Nt)
    n_rfft = N // 2 + 1
    phi_c = phi_centered(Nf, Nt, dt)
    eps = 1e-30

    # One batched FFT for every band, then a single vectorised
    # "largest window value wins" reduction over all (bin, band) candidates.
    V_all = (np.fft.fft(np.asarray(K_diag), axis=-1) / Nt).real  # (Nf+1, Nt)
    phi2 = phi_c * phi_c

    k_loc_half = np.arange(Nt // 2)
    k_loc_full = np.arange(Nt)

    # Each band contributes candidates (rfft bin, estimate, window weight).
    # DC's window index is offset by `half` where the other two are not -- see
    # the table in this function's Notes for why, before trying to merge them.
    bins_list = [k_loc_half]
    est_list = [
        V_all[0, : Nt // 2] / np.where(phi2[Nt // 2 :] > eps, phi2[Nt // 2 :], 1.0)
    ]
    wgt_list = [phi2[Nt // 2 :]]

    for m in range(1, Nf):
        slab_start = (m - 1) * (Nt // 2)
        keep = slab_start + k_loc_full < n_rfft
        bins_list.append(slab_start + k_loc_full[keep])
        est_list.append(V_all[m, keep] / np.where(phi2[keep] > eps, phi2[keep], 1.0))
        wgt_list.append(phi2[keep])

    tail = np.arange(Nt // 2 + 1)
    bins_list.append(N // 2 - Nt // 2 + tail)
    est_list.append(
        V_all[Nf, : Nt // 2 + 1]
        / np.where(phi2[: Nt // 2 + 1] > eps, phi2[: Nt // 2 + 1], 1.0)
    )
    wgt_list.append(phi2[: Nt // 2 + 1])

    bins = np.concatenate(bins_list)
    est = np.concatenate(est_list)
    wgt = np.concatenate(wgt_list)

    usable = wgt > eps
    bins, est, wgt = bins[usable], est[usable], wgt[usable]

    # Reverse first, then stable-sort by ascending weight, so that the final
    # write per bin is the *earliest* candidate among equal weights -- matching
    # the strict ``w > best_w`` comparison of the original scan.
    order = np.argsort(wgt[::-1], kind="stable")
    sigma2 = np.zeros(n_rfft)
    sigma2[bins[::-1][order]] = est[::-1][order]

    return 2.0 * dt * sigma2


def apply_lambda_from_psd(
    x_wmn: Array,
    Nf: int,
    Nt: int,
    dt: float,
    Sn_rfft: Array,
) -> Array:
    r"""Apply $\Lambda$ to a WDM vector via the operator-form forward.

    Computes $\Lambda \bm x$ from the one-sided PSD via the three-step
    pipeline

    .. math::
        \Lambda \bm x \;=\;
            \mathtt{forward}\!\bigl((\mathcal{S}_n / 2)\odot
                          \mathtt{inverse}(\bm x)\bigr),

    i.e., inverse-WDM, pointwise scale each rfft bin by half the PSD,
    forward-WDM. No assembly of $\Lambda$ as a matrix is needed.

    This is the operator dual of :func:`lambda_to_psd`: where the latter
    extracts a PSD from a stored $\Lambda$ in one inverse-WDM call, this
    one applies a $\Lambda$ implied by a stored PSD to any vector in two
    WDM-transform calls.

    Parameters
    ----------
    x_wmn : ndarray, float, shape (Nf+1, Nt)
        Input WDM coefficient array.  Inactive (parity-killed) edge entries
        are ignored on read -- :func:`wdm.transform.inverse` masks them -- so
        junk there does not affect the result (measured deviation exactly
        0.0).  This was not always so; see ``src/wdm/summary.md``.
    Sn_rfft : ndarray, float, shape (N//2 + 1,)
        One-sided PSD evaluated on the ``rfftfreq(N, dt)`` grid.  Last, like
        every other PSD argument in the package.
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (Nf+1, Nt)
        $\Lambda \bm x$.

    Notes
    -----
    Cost is $O(N \log N_t)$ per call — dominated by the two fast WDM
    transforms.  ``assemble_lambda_full_active`` builds the matrix in
    $O(N_f N_t^2)$ time and $O(N_f N_t^2)$ storage; for downstream uses
    that only ever need matvecs (MCMC, conjugate-gradient solves) the
    operator form is preferable.

    The result is bit-for-bit equal to ``Lambda @ x.ravel()`` from
    :func:`build_lambda_blocks_full_active` plus
    :func:`assemble_lambda_full_active`.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> Nf, Nt, dt = 8, 16, 1.0
    >>> N = Nf * Nt
    >>> x = wdm.forward_time(
    ...     np.random.default_rng(0).standard_normal(N), Nf, Nt, dt
    ... )
    >>> Sn = 1e-3 * np.ones(N // 2 + 1)
    >>> wdm.stationary.apply_lambda_from_psd(x, Nf, Nt, dt, Sn).shape
    (9, 16)
    """
    from vendored_2f7136c.transform import forward as _forward
    from vendored_2f7136c.transform import inverse as _inverse

    validate_grid(Nf, Nt)
    validate_dt(dt)
    Sn_rfft = sanitize_psd(Sn_rfft, "apply_lambda_from_psd")
    n_rfft = n_samples(Nf, Nt) // 2 + 1
    if Sn_rfft.shape != (n_rfft,):
        raise ValueError(
            f"apply_lambda_from_psd: Sn_rfft must have shape ({n_rfft},) -- the "
            f"np.fft.rfftfreq(N, dt) grid; got {Sn_rfft.shape}"
        )
    y = _inverse(x_wmn, Nf, Nt, dt)
    z = (Sn_rfft / 2.0) * y
    return _forward(z, Nf, Nt, dt)


def lambda_to_psd(
    diag_blocks: list[Array],
    offd_blocks: list[Array],
    n_active: list[NDArray[np.integer]],
    Nf: int,
    Nt: int,
    dt: float,
) -> Array:
    r"""Fast inverse of :func:`build_lambda_blocks_full_active`:
    recover the one-sided PSD $\mathcal{S}_n(f)$ from $\Lambda$.

    Implements the operator identity
    $\Lambda = \widetilde W\,\mathrm{diag}(\mathcal{S}_n/2)\,\widetilde W^{-1}$
    read backwards as

    .. math::
        \tfrac{1}{2}\mathcal{S}_n
          = \Re\!\bigl[\widetilde W^{-1}\bigl(\Lambda\,\widetilde W\mathbf{1}\bigr)\bigr]
          = \Re\!\bigl[\mathtt{inverse}\bigl(\Lambda\cdot\mathtt{forward}(\mathbf{1})\bigr)\bigr],

    where $\mathbf{1}$ is the all-ones rfft vector (= rfft of a time-domain
    impulse at $t=0$).  The pipeline is therefore three steps:

    1. Build the basis-only test wmn $\widetilde W\,\mathbf{1}$ via one
       :func:`wdm.transform.forward` call on the all-ones rfft.
    2. Apply $\Lambda$ to that real wmn vector (block-tridiag matvec).
    3. Apply :func:`wdm.transform.inverse` and take the real part.

    The round-trip with :func:`build_lambda_blocks_full_active` is exact
    to machine precision: feeding the returned $\mathcal{S}_n$ back to
    the forward reproduces $\Lambda$ bit-for-bit.

    Parameters
    ----------
    diag_blocks, offd_blocks, n_active
        Block representation of $\Lambda$, as produced by
        :func:`build_lambda_blocks_full_active`.
    Nf, Nt : int
    dt : float

    Returns
    -------
    ndarray, float, shape (N//2 + 1,)
        One-sided PSD evaluated on the ``np.fft.rfftfreq(N, dt)`` grid.

    Notes
    -----
    Cost: one forward-WDM call (test vector) + one block-tridiag matvec
    ($O(N_f N_t^2)$) + one inverse-WDM call ($O(N \log N_t)$).
    This is the operator dual of :func:`apply_lambda_from_psd`.

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> d, o, na = wdm.stationary.build_lambda_blocks_full_active(
    ...     8, 16, 1.0, flat
    ... )
    >>> Sn = wdm.stationary.lambda_to_psd(d, o, na, 8, 16, 1.0)
    >>> Sn.shape
    (65,)
    """
    from vendored_2f7136c.transform import forward as _forward_wdm
    from vendored_2f7136c.transform import inverse as _inverse_wdm

    validate_grid(Nf, Nt)
    validate_dt(dt)
    _check_block_layout(diag_blocks, offd_blocks, n_active, Nf)
    N = n_samples(Nf, Nt)

    ones_rfft = np.ones(N // 2 + 1, dtype=complex)
    bar_w = _forward_wdm(ones_rfft, Nf, Nt, dt)

    u = _block_tridiag_matvec(diag_blocks, offd_blocks, n_active, bar_w)

    X = _inverse_wdm(u, Nf, Nt, dt)

    return 2.0 * X.real


def assemble_lambda_full_active(
    diag_blocks: list[Array],
    offd_blocks: list[Array],
    n_active: list[NDArray[np.integer]],
    Nf: int,
    Nt: int,
) -> Array:
    """Assemble the full-active blocks into a dense $(N_f+1)N_t \\times (N_f+1)N_t$ matrix.

    Parity-killed edge entries stay exactly zero. The lower
    off-diagonal is filled by transposition.

    Parameters
    ----------
    diag_blocks : list of ($N_f+1$) ndarrays
    offd_blocks : list of $N_f$ ndarrays
    n_active : list of ($N_f+1$) integer arrays (active-$n$ per band)
    Nf, Nt : int

    Returns
    -------
    ndarray, shape ($(N_f+1)N_t$, $(N_f+1)N_t$)
        Symmetric. The result is identical to dense $W\\,C_{\\rm time}\\,W^T$
        at machine precision (verified in :mod:`wdm.tests.test_stationary`).

    Examples
    --------
    >>> import numpy as np, wdm
    >>> flat = lambda f: 1e-3 * np.ones_like(f)
    >>> d, o, na = wdm.stationary.build_lambda_blocks_full_active(
    ...     8, 16, 1.0, flat
    ... )
    >>> wdm.stationary.assemble_lambda_full_active(d, o, na, 8, 16).shape
    (144, 144)
    """
    validate_grid(Nf, Nt)
    _check_block_layout(diag_blocks, offd_blocks, n_active, Nf)
    M = (Nf + 1) * Nt
    Lambda = np.zeros((M, M))
    band_offset = [m * Nt for m in range(Nf + 2)]

    # Diagonal blocks: place in active rows/cols of band-m's slot.  Interior
    # bands are contiguous, so a plain slice beats np.ix_'s fancy indexing.
    for m, D in enumerate(diag_blocks):
        if m in (0, Nf):
            idx = band_offset[m] + n_active[m]
            Lambda[np.ix_(idx, idx)] = D
        else:
            s0 = band_offset[m]
            Lambda[s0 : s0 + Nt, s0 : s0 + Nt] = D

    # Off-diagonal (m, m+1) blocks: edge dimensions on the edge side only.
    for m, L in enumerate(offd_blocks):
        row_idx: NDArray[np.integer]
        col_idx: NDArray[np.integer]
        if m == 0:
            row_idx = band_offset[0] + n_active[0]  # edge restriction
            col_idx = band_offset[1] + np.arange(Nt)  # interior full
        elif m == Nf - 1:
            row_idx = band_offset[Nf - 1] + np.arange(Nt)
            col_idx = band_offset[Nf] + n_active[Nf]  # edge restriction
        else:
            row_idx = band_offset[m] + np.arange(Nt)
            col_idx = band_offset[m + 1] + np.arange(Nt)
        Lambda[np.ix_(row_idx, col_idx)] = L
        Lambda[np.ix_(col_idx, row_idx)] = L.T

    return Lambda
