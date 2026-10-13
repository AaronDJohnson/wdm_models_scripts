r"""The sinusoid ``exact`` model's fitted assets and its composition (spec §5.1 exact.py).

The exact model is the time-domain covariance :math:`A_n\tilde\Sigma(\alpha_c, \alpha_d)` (WDM II
Eq. 47 at :math:`\theta = 0`) restricted to the retained Fourier coordinates
:math:`4\ \mathrm{Hz} \le f < f_{\rm Nyq}` (a flat marginal over the 128 discarded ones):

* quadratic form -- projected PCG (tolerance ``EXACT_TOL = 1e-7``, true-residual check) on the
  Chebyshev operator (``domainia ExactTimeCovariance(form="operator")`` inside
  ``ProjectedOperatorMarginal``);
* log-determinant -- :class:`ExactLogDet` (the closed-form Weyl term plus the fitted
  :math:`c_N\alpha_d^2`, WT ``FastCovariance.logdet``) plus :class:`BandCorrection` (the
  validated 2-D Chebyshev interpolant of the discarded block's log-det correction
  :math:`\log|B C^{-1} B^T|`, WT ``BandCovariance.logdet``).

Both assets are shipped (``data/sinusoid/cache``) and loaded **fail-closed** through
``domainia.assets.Artifact``: sha256 pinned here and in ``data/MANIFEST.json``; the bandcov file
is named by its identity (``bandcov_v1_<sha256(json.dumps(identity, sort_keys=True))[:20]>``),
and that identity dict is kept verbatim from WT ``BandCovariance._setup``
(``wdm_paper_2/toy_models/src/wdm_toys/bandcov.py:131-145``) so
``bandcov_v1_4bfb466c191c79f9879a.npz`` loads unchanged.  Nothing here ever fits or writes an
asset: refitting is ``scripts/sinusoid/prepare_bandcov.py`` (offline, hours).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.polynomial import chebyshev as cheb

from domainia.assets import Artifact, ArtifactError
from domainia.covariance import ExactTimeCovariance
from domainia.nuisance import FourierBandNuisance, ProjectedOperatorMarginal
from wdm.backend import Backend, get_backend

from ..common.validation import resolve_data_root, shipped
from .config import ANALYSIS_FMIN, EXACT_P, EXACT_TOL, SinusoidConfig
from .priors import noise_bounds
from .psd import as_loglinear

__all__ = ["BANDCOV_KEY", "BANDCOV_SHA256", "BandCorrection", "CN_SHA256", "ExactLogDet",
           "ProjectedExact", "bandcov_filename", "bandcov_identity", "check_bandcov", "cn_filename",
           "scaled"]

CACHE_REL = "sinusoid/cache"
CN_SHA256 = "d263bf25960ab6f4e371fd597da0e28123cba569ca085e2af5e6c3599257213f"
BANDCOV_SHA256 = "752b3ea46cf14feaf5b8845ad39999a0b596a956d6af9e58072f691a8cd6c2c2"
BANDCOV_KEY = "4bfb466c191c79f9879a"
BANDCOV_VERSION = 1
#: WT bandcov.MAX_CACHE_ERROR / MAX_CACHE_SOLVER_TOL; the c_N agreement WT required (atol).
MAX_CACHE_ERROR = 2e-5
MAX_CACHE_SOLVER_TOL = 1e-9
CN_ATOL = 1e-12
_MANUSCRIPT = SinusoidConfig()


def cn_filename(cfg: SinusoidConfig) -> str:
    """WT ``FastCovariance._load_or_compute_cN``'s key (``fastcov.py:184``)."""
    return f"cN_N{cfg.N}_dt{cfg.dt:.12g}_fmin{cfg.f_min:g}_fref{cfg.f_ref:g}.npy"


def bandcov_identity(cfg: SinusoidConfig = _MANUSCRIPT, analysis_fmin: float = ANALYSIS_FMIN,
                     P: int = EXACT_P) -> dict[str, Any]:
    """WT ``BandCovariance._setup``'s identity dict, verbatim (keys, values, order)."""
    band = FourierBandNuisance(cfg.N, cfg.dt, analysis_fmin)
    return dict(version=BANDCOV_VERSION, N=cfg.N, dt=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
                analysis_fmin=band.fmin, n_keep=band.n_keep, n_drop=band.n_drop,
                support="real-orthonormal;f>=fmin;f<Nyquist;noDC", P=int(P),
                bounds=noise_bounds().tolist(), drift_transform="asinh",
                drift_scale=min(.03, .384 * cfg.f_min / cfg.f_nyq),
                algorithm="latent-C0-PCG-discarded-precision-chebyshev")


def bandcov_filename(cfg: SinusoidConfig = _MANUSCRIPT, analysis_fmin: float = ANALYSIS_FMIN,
                     P: int = EXACT_P) -> str:
    """``bandcov_v1_<key>.npz`` (WT ``BandCovariance.cache_path``)."""
    return Artifact.keyed(".", f"bandcov_v{BANDCOV_VERSION}_",
                          bandcov_identity(cfg, analysis_fmin, P)).path.name


def scaled(ac: Any, ad: Any, bounds: np.ndarray, drift_scale: float) -> tuple[Any, Any]:
    """WT ``bandcov._scaled``: ``alpha_c`` affinely and ``asinh(alpha_d / drift_scale)`` onto
    ``[-1, 1]`` over the prior box."""
    x = 2*(ac-bounds[0, 0])/(bounds[0, 1]-bounds[0, 0])-1
    lo, hi = np.arcsinh(bounds[1]/drift_scale)
    y = 2*(np.arcsinh(ad/drift_scale)-lo)/(hi-lo)-1
    return x, y


def check_bandcov(arrays: dict[str, np.ndarray], identity: dict[str, Any]) -> dict[str, Any]:
    """WT ``BandCovariance._load``'s checks on a loaded artifact; returns its metadata.
    Raises ``ArtifactError`` on any failure."""
    missing = {"metadata", "coefficients", "validation_points", "validation_exact"} - set(arrays)
    if missing:
        raise ArtifactError(f"the bandcov artifact lacks {sorted(missing)}")
    try:
        meta = json.loads(str(arrays["metadata"]))
        coeff = np.array(arrays["coefficients"], dtype=float)
        points = np.array(arrays["validation_points"], dtype=float)
        exact = np.array(arrays["validation_exact"], dtype=float)
        if meta["identity"] != json.loads(json.dumps(identity, sort_keys=True)):
            raise ArtifactError("band covariance cache identity mismatch")
        if not meta.get("validated", False):
            raise ArtifactError("band covariance cache is not validated")
        if not 0 < meta["solver_tol"] <= MAX_CACHE_SOLVER_TOL:
            raise ArtifactError("band covariance cache solver accuracy is insufficient")
        if not np.isfinite(meta["c_N"]):
            raise ArtifactError("invalid c_N in the band covariance cache")
        tol = meta["validation_tol"]
        if not 0 < tol <= MAX_CACHE_ERROR:
            raise ArtifactError("band covariance cache validation tolerance is insufficient")
        if coeff.ndim != 2 or not np.all(np.isfinite(coeff)):
            raise ArtifactError("invalid band covariance cache coefficients")
        if list(coeff.shape) != [d + 1 for d in meta["degree"]]:
            raise ArtifactError("band covariance cache polynomial degree mismatch")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 9 or exact.shape != (len(points),):
            raise ArtifactError("band covariance cache lacks independent validation points")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(exact)):
            raise ArtifactError("non-finite band covariance cache validation")
        xy = scaled(points[:, 0], points[:, 1], np.asarray(identity["bounds"], dtype=float),
                    float(identity["drift_scale"]))
        error = np.max(np.abs(cheb.chebval2d(*xy, coeff) - exact))
        if not np.isfinite(error) or error > tol or meta["max_error"] > tol:
            raise ArtifactError("band covariance cache failed validation")
    except (KeyError, TypeError, ValueError) as err:
        if isinstance(err, ArtifactError):
            raise
        raise ArtifactError("invalid band covariance cache") from err
    return meta


@dataclass(frozen=True)
class ExactLogDet:
    r""":math:`\log|\tilde\Sigma| = \alpha_c S_L - \alpha_d S_L/(2N) - N\ln(2\delta t) + c_N\alpha_d^2`
    (WT ``FastCovariance.logdet``); ``c_N`` is the shipped fit, never refitted here."""

    c_N: float
    cfg: SinusoidConfig = _MANUSCRIPT

    def __post_init__(self) -> None:
        if not math.isfinite(self.c_N):
            raise ArtifactError(f"c_N must be finite; got {self.c_N!r}")

    @classmethod
    def load(cls, cfg: SinusoidConfig = _MANUSCRIPT, *,
             data_root: str | os.PathLike | None = None) -> ExactLogDet:
        art = shipped(resolve_data_root(data_root), f"{CACHE_REL}/{cn_filename(cfg)}", CN_SHA256)
        value = art.load_npy()
        if value.shape != () or value.dtype != np.float64:
            raise ArtifactError(f"{art.path}: c_N must be a float64 scalar; got {value.dtype} {value.shape}")
        return cls(float(value), cfg)

    def covariance(self, *, P: int = EXACT_P, tol: float = EXACT_TOL, maxit: int = 200,
                   on_fail: Literal["raise", "nan"] = "raise") -> ExactTimeCovariance:
        """The exact operator covariance (``A_n`` outside) carrying this ``c_N``."""
        return ExactTimeCovariance(as_loglinear(self.cfg), self.cfg.N, self.cfg.dt, form="operator",
                                   P=P, c_N=self.c_N, tol=tol, maxit=maxit, on_fail=on_fail)

    def weyl(self, nu: Any) -> float:
        return self.covariance().weyl_logdet(nu)

    def __call__(self, nu: Any) -> float:
        return self.covariance().logdet_model(nu)


@dataclass(frozen=True, eq=False)
class BandCorrection:
    """The bandcov interpolant ``chebval2d(scaled(alpha_c, alpha_d), coefficients)``; refuses
    noise outside the validated box (WT ``_check_domain``)."""

    coefficients: np.ndarray
    bounds: np.ndarray
    drift_scale: float
    metadata: dict = field(repr=False)
    path: Path

    @property
    def c_N(self) -> float:
        return float(self.metadata["c_N"])

    @property
    def identity(self) -> dict:
        """The artifact's identity dict (checked equal to :func:`bandcov_identity` on load)."""
        return self.metadata["identity"]

    @classmethod
    def load(cls, cfg: SinusoidConfig = _MANUSCRIPT, *, data_root: str | os.PathLike | None = None,
             analysis_fmin: float = ANALYSIS_FMIN, P: int = EXACT_P) -> BandCorrection:
        identity = bandcov_identity(cfg, analysis_fmin, P)
        rel = f"{CACHE_REL}/{bandcov_filename(cfg, analysis_fmin, P)}"
        art = shipped(resolve_data_root(data_root), rel, BANDCOV_SHA256)
        arrays = Artifact(art.path, identity=identity, sha256=BANDCOV_SHA256).load_npz()
        meta = check_bandcov(arrays, identity)
        coeff = np.array(arrays["coefficients"], dtype=float)
        coeff.setflags(write=False)
        bounds = np.asarray(identity["bounds"], dtype=float)
        bounds.setflags(write=False)
        return cls(coeff, bounds, float(identity["drift_scale"]), meta, art.path)

    def check_domain(self, alpha_c: float, alpha_d: float) -> None:
        values = np.asarray([alpha_c, alpha_d], dtype=float)
        if (not np.all(np.isfinite(values)) or np.any(values < self.bounds[:, 0])
                or np.any(values > self.bounds[:, 1])):
            raise ValueError(f"noise indices {values.tolist()} are outside the validated cache/prior "
                             f"domain {self.bounds.tolist()}")

    def __call__(self, nu: Any, be: Backend | None = None) -> Any:
        """The correction at ``nu = (alpha_c, alpha_d)`` (NumPy; the traced evaluation is S4's)."""
        be = be or get_backend(nu)
        if be.name != "numpy":
            raise NotImplementedError("BandCorrection evaluates on NumPy only; its traced (JAX) "
                                      "Chebyshev evaluation is sub-project S4's")
        ac, ad = float(nu[0]), float(nu[1])
        self.check_domain(ac, ad)
        return cheb.chebval2d(*scaled(ac, ad, self.bounds, self.drift_scale), self.coefficients)


@dataclass(frozen=True, eq=False)
class ProjectedExact:
    """The exact model's covariance: ``ExactLogDet`` + ``BandCorrection`` over the retained
    Fourier coordinates (the two assets must agree on ``c_N`` to ``1e-12``, as WT required)."""

    logdet: ExactLogDet
    correction: BandCorrection
    nuisance: FourierBandNuisance

    def __post_init__(self) -> None:
        if abs(self.logdet.c_N - self.correction.c_N) > CN_ATOL:
            raise ArtifactError(f"c_N {self.logdet.c_N!r} does not match the bandcov artifact's "
                                f"{self.correction.c_N!r} (atol {CN_ATOL})")
        cfg = self.logdet.cfg
        if (self.nuisance.N, self.nuisance.dt) != (cfg.N, cfg.dt):
            raise ValueError("the nuisance and the log-determinant describe different records")

    @classmethod
    def load(cls, cfg: SinusoidConfig = _MANUSCRIPT, *, data_root: str | os.PathLike | None = None,
             analysis_fmin: float = ANALYSIS_FMIN, P: int = EXACT_P) -> ProjectedExact:
        return cls(ExactLogDet.load(cfg, data_root=data_root),
                   BandCorrection.load(cfg, data_root=data_root, analysis_fmin=analysis_fmin, P=P),
                   FourierBandNuisance(cfg.N, cfg.dt, analysis_fmin))

    def covariance(self, *, P: int = EXACT_P, tol: float = EXACT_TOL, maxit: int = 200,
                   on_fail: Literal["raise", "nan"] = "raise") -> ProjectedOperatorMarginal:
        return ProjectedOperatorMarginal(self.logdet.covariance(P=P, tol=tol, maxit=maxit, on_fail=on_fail),
                                         self.nuisance, self.correction)
