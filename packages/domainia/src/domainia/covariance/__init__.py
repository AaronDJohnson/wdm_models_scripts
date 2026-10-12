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
from .hermitian import HermitianCovariance, HermitianFactor

__all__ = [
    "BlockTriCovFactor",
    "BlockTridiagonalCovariance",
    "CholFactor",
    "Covariance",
    "DenseCovariance",
    "DiagFactor",
    "DiagonalCovariance",
    "Eq45Covariance",
    "Evaluator",
    "ExactFourierCovariance",
    "Factor",
    "HermitianCovariance",
    "HermitianFactor",
    "QuadForm",
    "as_evaluator",
    "dense_factor",
    "ghat",
    "keep_mask",
    "quad_form",
    "window_bins",
]
