"""S4 preview: the sinusoid manuscript_mid models on JAX with the historical (WG) strategies,
assembled from domainia alone -- the inputs of the S4 state-memory, keyed-jit timing and
HLO-budget tests (spec section 7.6 S4).  The JAX registry (``wdm_models``) arrives in a later
task of S4 and must reproduce these compositions; until then they are test-local and named
as such (``scripts.golden.s2_cases`` is the NumPy counterpart).

JAX historical strategies (spec section 3.2; part-2 plan D-C note 4; S4 deferral (b)):

=============  =============================================================================
piece          JAX default
=============  =============================================================================
Eq. 45 (tri)   ``builder="gram"``, ``layout="padded"``; ``FlatMarginal(form="whiten")`` (raw/cut)
diagonal       ``Band``/``MidpointVariance``; ``FlatMarginal(form="solve")``
exact-edge     ``PushedRows(order="adaptive", phi_cache=True, staged=True)``,
               ``SchurEdge(form="scaled")`` over a diagonal interior (``"solve"`` over tri)
pad            ``ZeroPadded(form="reduce")`` + ``FlatMarginal(form="reduce")`` (WG cQ, Gu)
noise          ``noise_clamp={"alpha_d": 4.0}`` (WG ``ALPHA_D_CLAMP``)
=============  =============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import wdm
from domainia.covariance import (DiagonalCovariance, Eq45Covariance, ExactTimeCovariance, PushedRows, SchurEdge,
                                 ZeroPadded)
from domainia.data import TimeSeries
from domainia.density import RealGaussian
from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.likelihood import GaussianLikelihood
from domainia.maps import Identity, TimeToWDM
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance
from wdm.backend import get_backend, pytree_dataclass

from scripts.golden.s2_cases import CUT, DRIFT, DT, EDGE, FMIN, NF, NT, N, PAD, sinusoid_inputs

#: WG's per-state bytes at the manuscript grid (MiB; review measurement 2026-10-12, lineage B,
#: ``wdmtoysgpu.MatrixGPU._factor`` outputs: band_pad cQ + Gu, band_exact Bm + cS, tri_raw
#: Ls + Xs + Zu + cH).  S4 gate: the release's JAX noise state within 1.5x.
WG_STATE_MIB = {"band_pad": 9.3, "band_exact": 120.0, "tri_raw": 81.0}


@pytree_dataclass(data=("t",), meta=())
@dataclass(frozen=True, eq=False)
class SinusoidSignal:
    """WT ``likelihood.signal`` (fdot = 0) with the sample times as a pytree leaf (a jit
    argument, not a literal): ``A_h cos(2 pi f0 t + phi0)``, ``t_j = j dt``."""

    t: Any

    def __call__(self, src: Any) -> Any:
        xp = get_backend(src, self.t).xp
        return src[0] * xp.cos(2.0 * np.pi * src[1] * self.t + src[2])


def _base(base: str, psd: LogLinearPSD, cells: CellSet, t_max: float | None = None):
    if base == "band":
        return DiagonalCovariance(BandAveragedVariance(psd, cells), len(cells.m))
    if base == "mid":
        return DiagonalCovariance(MidpointVariance(psd, cells), len(cells.m))
    return Eq45Covariance(psd, cells, builder="gram", layout="padded", t_max=t_max)


def sinusoid_jax_likelihood(model: str, *, forms: str = "wg") -> GaussianLikelihood:
    """One sinusoid WDM model (``{band,mid,tri}_{raw,cut,pad,exact}``) at manuscript_mid on
    JAX with the historical strategies above.  ``forms="wt"`` keeps WT's state forms instead
    (``ZeroPadded`` / ``FlatMarginal`` ``"solve"``, ``SchurEdge`` ``"solve"``): the S2 layout, the
    memory test's negative control."""
    if forms not in ("wg", "wt"):
        raise ValueError(f"forms must be 'wg' or 'wt'; got {forms!r}")
    wg = forms == "wg"
    s = sinusoid_inputs()
    time = TimeDomain(N, DT)
    data = TimeSeries(s.d, time)
    psd = LogLinearPSD(f_floor=FMIN, f_ref=1.0, T=N * DT)
    signal = SinusoidSignal(np.arange(N) * DT)
    nuis = FourierBandNuisance(N, DT, FMIN)
    base, edge = model.split("_")
    clamp = {"alpha_d": 4.0}
    if edge == "pad":
        ext = wdm.Grid(NF, NT + PAD, DT)
        zp = ZeroPadded(_base(base, psd, CellSet.all_active(ext), t_max=ext.Nt * wdm.delta_T(NF, DT)),
                        strategy="time_marginal", grid=ext, pad_columns=PAD, form="reduce" if wg else "solve")
        cov = FlatMarginal(zp, nuis.discarded_basis().T, form="reduce" if wg else "solve")
        return GaussianLikelihood(data, Identity(time), cov, RealGaussian(0.0, "log10_A_n"), signal=signal,
                                  params=DRIFT, noise_clamp=clamp, backend="jax")
    cells = CellSet.all_active(wdm.Grid(NF, NT, DT))
    if edge == "cut":
        cells = cells.drop_columns(*CUT)
    tmap = TimeToWDM(time, WDMDomain(cells))
    U = tmap.apply(nuis.discarded_basis()).T
    if edge == "exact":
        e = cells.columns(*EDGE)
        rows = PushedRows(ExactTimeCovariance(psd, N, DT, form="operator", P=24, on_fail="nan"), tmap, e,
                          order="adaptive", phi_cache=True, staged=True)
        cov = FlatMarginal(SchurEdge(_base(base, psd, cells.complement(e)), rows, e,
                                     form="scaled" if wg and base != "tri" else "solve"), U)
    else:
        cov = FlatMarginal(_base(base, psd, cells), U, form="whiten" if base == "tri" else "solve")
    return GaussianLikelihood(data, tmap, cov, RealGaussian(0.5 * N * float(np.log(DT)), "log10_A_n"),
                              signal=signal, params=DRIFT, noise_clamp=clamp, backend="jax")


def state_bytes(state: Any) -> int:
    """The per-state memory: the sum of the leaves' ``nbytes`` (what an LRU slot holds)."""
    import jax

    return int(sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(state)))
