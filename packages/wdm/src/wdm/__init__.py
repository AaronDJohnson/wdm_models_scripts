"""Wilson--Daubechies--Meyer time-frequency transform.

A Wilson-basis time-frequency tiling with a Meyer scaling-function
window.  Despite the historical name, this is **not** a wavelet
transform: the frequency grid is uniform (not dyadic) and the basis
functions are Wilson-basis trigonometric atoms on a half-shifted
lattice.  The acronym stands for *Wilson--Daubechies--Meyer*, the
mathematicians whose work underlies the construction.

Public API
----------
Forward / inverse data transforms:

* :func:`forward`, :func:`inverse`
* :func:`forward_time`, :func:`inverse_time` (time-domain wrappers)
* :func:`forward_many`, :func:`inverse_many` and their ``_time`` wrappers
  (batched over a stack of series sharing one grid -- one FFT for the whole
  stack, bit-identical to looping)

A validated ``(Nf, Nt, dt)`` container, which removes the one misuse the
package cannot otherwise detect (swapping the two same-typed ints):

* :class:`Grid` -- unpacks as ``(Nf, Nt, dt)``, so ``f(x, *grid)`` works

Input validation (all applied automatically by the entry points):

* :func:`validate_grid`, :func:`validate_dt`, :func:`validate_finite`,
  :func:`grid.sanitize_psd`

Grid helpers:

* :func:`n_samples`, :func:`band_frequencies`, :func:`time_pixel_centers`
* :func:`delta_F`, :func:`delta_T`, :func:`delta_omega`
* :func:`active_mask`, :func:`count_active`

Noise variance grids:

* :func:`whittle_variance`, :func:`whittle_variance_from_array`
* :func:`band_averaged_variance`
* :func:`evolutionary_variance`
* :func:`empirical_variance`
* :func:`evaluate_psd_on_band_grid`

Inner products and log-likelihoods (in :mod:`wdm.inner_product`):

* :func:`inner_product.whittle`
* :func:`inner_product.band_averaged`
* :func:`inner_product.evolutionary_whittle`
* :func:`inner_product.gaussian`, :func:`inner_product.gaussian_from_func`
* :func:`inner_product.loglike_whittle`
* :func:`inner_product.loglike_band_averaged`
* :func:`inner_product.loglike_evolutionary_whittle`
* :func:`inner_product.loglike_gaussian`

Batched over a stack of coefficient grids (one call, ``(B,)`` out):

* :func:`inner_product.whittle_many`,
  :func:`inner_product.loglike_whittle_many` -- these cover the
  ``band_averaged`` and ``evolutionary_whittle`` flavours too, which differ
  only in the variance grid

Explicit matrix construction (in :mod:`wdm.matrix`):

* :func:`matrix.build_W`, :func:`matrix.active_W`
* :func:`matrix.build_C_time`, :func:`matrix.build_C_wdm`,
  :func:`matrix.build_C_wdm_active`
* :func:`matrix.rfft_to_real`, :func:`matrix.real_to_rfft`

Fast stationary noise covariance (in :mod:`wdm.stationary`):

* :func:`stationary.build_lambda_blocks_full_active` (interior + edges,
  matches dense $W_a\\,C\\,W_a^T$ to machine precision)
* :func:`stationary.assemble_lambda_full_active`
* :func:`stationary.is_diagonal`
* :func:`stationary.psd_from_lambda_blocks` (fast inverse: $\\Lambda \\to S_n(f)$)
* :func:`stationary.apply_lambda_from_psd` ($\\Lambda x$ without assembling
  $\\Lambda$; $O(N \\log N_t)$ per call)
* :func:`stationary.build_K_full_active`, :func:`stationary.psd_from_K`
  (compact $\\Lambda$: one length-$N_t$ vector per band pair, and the inverse
  taken straight from it)

Filter (in :mod:`wdm.filter`):

* :func:`filter.phi_centered`, :func:`filter.phi_fft_order`
"""

from __future__ import annotations

from . import (
    backend,
    evolutionary,
    filter,
    inner_product,
    matrix,
    noise,
    paper,
    special,
    stationary,
)
from .filter import phi_centered, phi_fft_order
from .grid import (
    Grid,
    active_mask,
    band_frequencies,
    count_active,
    delta_F,
    delta_omega,
    delta_T,
    n_samples,
    time_pixel_centers,
    validate_dt,
    validate_finite,
    validate_grid,
)
from .noise import (
    band_averaged_variance,
    empirical_variance,
    evaluate_psd_on_band_grid,
    evolutionary_variance,
    whittle_variance,
    whittle_variance_from_array,
)
from .transform import (
    forward,
    forward_many,
    forward_time,
    forward_time_many,
    inverse,
    inverse_many,
    inverse_time,
    inverse_time_many,
)

# Release sentinels: literals, never read from importlib.metadata (which reports
# whichever distribution named "wdm" happens to be installed).  domainia and
# wdm_models refuse any wdm whose __release__ differs.
__version__ = "1.0.0"
__wdm_api__ = "functions/1"
__release__ = "wdm-paper-1"

__all__ = [
    "Grid",
    "__release__",
    "__version__",
    "__wdm_api__",
    "active_mask",
    "backend",
    "band_averaged_variance",
    "band_frequencies",
    "count_active",
    "delta_F",
    "delta_T",
    "delta_omega",
    "empirical_variance",
    "evaluate_psd_on_band_grid",
    "evolutionary_variance",
    # Submodules
    "evolutionary",
    "filter",
    # Transform
    "forward",
    "forward_many",
    "forward_time",
    "forward_time_many",
    "inner_product",
    "inverse",
    "inverse_many",
    "inverse_time",
    "inverse_time_many",
    "matrix",
    # Grid
    "n_samples",
    "noise",
    "paper",
    # Filter
    "phi_centered",
    "phi_fft_order",
    "special",
    "stationary",
    "time_pixel_centers",
    "validate_dt",
    "validate_finite",
    "validate_grid",
    # Noise
    "whittle_variance",
    "whittle_variance_from_array",
]
