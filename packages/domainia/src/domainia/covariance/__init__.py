"""Covariance operators (spec §4.4-§4.6): one protocol, one factor per noise state.

Modules: ``base`` (protocol, factors, ``Covariance``), ``diagonal``, ``dense``,
``blocktri``, ``eq45`` (Eq. 45, five builders), ``exact_fourier`` (M'), ``hermitian``,
``exact_time`` (Eq. 47 in time), ``pushed`` (exact rows on cells), ``edges`` (``cut``,
``SchurEdge``, ``ZeroPadded``), ``stack`` (``ChannelStack``).  The flat marginals live in
``domainia.nuisance``.
"""

from .base import CholFactor, Covariance, DiagFactor, Evaluator, Factor, QuadForm, as_evaluator, quad_form
from .blocktri import BlockTriCovFactor, BlockTridiagonalCovariance
from .dense import DenseCovariance, dense_factor
from .diagonal import DiagonalCovariance
from .eq45 import Eq45Covariance, keep_mask, window_bins
from .exact_fourier import ExactFourierCovariance, ghat
from .exact_time import CovOperator, ExactTimeCovariance, OperatorFactor, bessel_coefficients
from .hermitian import HermitianCovariance, HermitianFactor

__all__ = [
    "BlockTriCovFactor",
    "BlockTridiagonalCovariance",
    "CholFactor",
    "CovOperator",
    "Covariance",
    "DenseCovariance",
    "DiagFactor",
    "DiagonalCovariance",
    "Eq45Covariance",
    "Evaluator",
    "ExactFourierCovariance",
    "ExactTimeCovariance",
    "Factor",
    "HermitianCovariance",
    "HermitianFactor",
    "OperatorFactor",
    "QuadForm",
    "as_evaluator",
    "bessel_coefficients",
    "dense_factor",
    "ghat",
    "keep_mask",
    "quad_form",
    "window_bins",
]
