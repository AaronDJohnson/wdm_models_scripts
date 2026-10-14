r"""The 15 galactic-binary likelihoods (spec sections 4.9, 5.3), composed from ``domainia``.

One function, :func:`likelihood`, builds a model of one prepared case
(:class:`~wdm_models.galactic_binary.prepared.PreparedCase`) from ``domainia`` pieces, as
``scripts/golden/s2_cases.py`` previewed (S2) and the lineage-C oracles define
(``toy_models/galactic_binary/matrix.py`` ``make_lnlike``, ``MatrixCache``,
``EdgeSchurCache``; ``fullcache.make_lnlike``, ``FullCache``; ``likelihoods.make_lnlike``):

* data: the store bins of A and E (``FourierBins`` on ``FourierDomain(N, DT, store_offset,
  store_n, "sigma0")``); the signal is the template on the store (``waveform.GBTemplate``,
  ``window="store"``) -- on the window bins it equals the window template bit for bit;
* the data vector differs by model (spec section 4.9): WDM raw / cut / exact-edge -- the cells
  of the analysis bands from the band windows (``FourierToWDM``, ``impl="dense"``); pad -- the
  padded record's kept cells (``FourierToWDM(padded window).after(FourierZeroPad(store))``);
  whittle_raw and exact -- the nominal bins (``BinSelect``); whittle_cut -- the cropped
  record's bins (``FourierCrop``);
* noise: ``SeparablePSD`` (additive: ``s_I + g(t) s_G``) on the store bins, one
  ``ChannelStack`` channel per TDI channel (``params_per_channel`` = 5 envelope
  coefficients); the Whittle forms carry one level per channel;
* density: ``RealGaussian()`` (``P ln 2 pi``, no amplitude) for the 12 WDM models,
  ``ComplexGaussian()`` (no ``n ln pi``) for ``exact``; the Whittle forms their own;
* ``row_ok``: ``priors.source_ok`` on the prepared GLASS box (``prior_lo``/``prior_hi``):
  rows outside it are ``-inf`` with a positive-definite noise state, as the oracles; the
  template is evaluated at the unwrapped source, as the oracles did.

The arithmetic strategies come from :func:`wdm_models.options.resolve` (``GBResolved``):
NumPy ``historical`` is lineage C's NumPy reference (``make_lnlike``: residual strategy,
dense assembly of the edge-exact matrix, dense Cholesky of the pad's ``A Lambda_R A^T``,
the 12-month exact factor in ``NT/2``-bin blocks); ``evaluator="cache"`` is lineage C's
NumPy cache classes (expanded strategy; band/mid_exact by the Schur complement of the
diagonal interior, ``EdgeSchurCache``).  JAX ``historical`` (S7) is lineage C's
``gpu/jax_models.build``: the cached models in a ``NoiseStateCache`` (rows batched except
``exact``), the others in a ``BatchedLikelihood``; the signal there is the ``GBTemplate``
pytree itself (:func:`signal_for`) and the pad's ``Lambda_R`` uses the jittable
:data:`PAD_LAYOUT`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

import wdm
from domainia.covariance import (ChannelStack, DiagonalCovariance, Eq45Covariance, ExactFourierCovariance,
                                 HermitianCovariance, PadMask, PushedRows, SchurEdge, ZeroPadded)
from domainia.data import FourierBins
from domainia.density import ComplexGaussian, RealGaussian
from domainia.domains import CellSet, ChannelAxis, FourierDomain, WDMDomain
from domainia.likelihood import FourierWhittle, GaussianLikelihood, StaticSignal
from domainia.maps import BinSelect, FourierCrop, FourierToWDM, FourierZeroPad
from domainia.sampling import BatchedLikelihood, NoiseStateCache
from domainia.spectra import SeparablePSD, TabulatedSpectrum
from domainia.variance import BandAveragedVariance, MidpointVariance

from ..common.params import check_model, split_model
from ..options import GB_CACHED as CACHED
from ..options import ModelOptions, resolve
from . import priors
from .config import CUT, ENV_NAMES, ENVELOPE_PERIOD, N_HARM, PAD_COLUMNS, PAD_EDGE_TOL, TEMPLATE_BW
from .envelope import AnnualEnvelope
from .prepared import PreparedCase

__all__ = ["CACHED", "CHANNELS", "Geometry", "NumpyTemplate", "PAD_LAYOUT", "SCHUR", "TARGET_MODELS", "build",
           "envelope", "geometry", "likelihood", "pad_mask", "pad_psd", "signal_for", "store_psd", "wrap"]

CHANNELS = ("A", "E")
#: The edge-exact models lineage C's cache evaluated by the Schur complement (``matrix.SCHUR``).
SCHUR = ("band_exact", "mid_exact")
#: The models of each case (spec section 5.3): at 12 months the periodic envelope makes the
#: cut and the exact edges identical to raw (``CUT["gb_12mo"] == (0, 0)``), so they are not targets.
TARGET_MODELS = {
    "gb_3mo": ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact", "mid_raw",
               "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad", "tri_exact", "exact"),
    "gb_6mo": ("whittle_raw", "whittle_cut", "band_raw", "band_cut", "band_pad", "band_exact", "mid_raw",
               "mid_cut", "mid_pad", "mid_exact", "tri_raw", "tri_cut", "tri_pad", "tri_exact", "exact"),
    "gb_12mo": ("whittle_raw", "band_raw", "mid_raw", "tri_raw", "band_pad", "mid_pad", "tri_pad", "exact"),
}


def _int(a: Any) -> int:
    return int(np.asarray(a))


@dataclass(frozen=True, eq=False)
class Geometry:
    """Grids, cells and bin domains of one prepared case (host constants, built once)."""

    case: str
    N: int
    DT: float
    T: float
    NT: int
    bands: tuple[int, ...]
    cut: tuple[int, int]
    periodic: bool
    store: FourierDomain
    window: FourierDomain
    nominal: FourierDomain
    crop: FourierDomain
    crop_start: int

    @property
    def Nf(self) -> int:
        return self.N // self.NT

    @cached_property
    def grid(self) -> wdm.Grid:
        return wdm.Grid(self.Nf, self.NT, self.DT)

    @cached_property
    def cells(self) -> CellSet:
        """The analysis bands x NT columns (720 / 1314 / 2700 cells per channel)."""
        return CellSet.bands(self.grid, self.bands)

    @cached_property
    def cut_cells(self) -> CellSet:
        return self.cells.drop_columns(*self.cut)

    @cached_property
    def edge(self) -> CellSet:
        """The exact-edge cells: the cut's columns (27 at 3 and 6 months, none at 12)."""
        return self.cells.columns(*self.cut)

    @cached_property
    def pad_grid(self) -> wdm.Grid:
        return wdm.Grid(self.Nf, self.NT + PAD_COLUMNS, self.DT)

    @cached_property
    def pad_window(self) -> FourierDomain:
        """The padded record's band windows (GB ``padded_grid``: ``g0``, ``G``)."""
        h, b0, b1 = self.pad_grid.Nt // 2, self.bands[0], self.bands[-1]
        return FourierDomain(self.pad_grid.N, self.DT, h * b0 - h, h * (b1 - b0) + self.pad_grid.Nt, "sigma0")

    @cached_property
    def pad_kept(self) -> CellSet:
        """The padded grid's analysis-band cells with ``n < NT`` (the data's columns)."""
        return CellSet.bands(self.pad_grid, self.bands).drop_columns(0, PAD_COLUMNS)

    @cached_property
    def store_bands(self) -> tuple[int, ...]:
        """GB ``store_bands``: every band whose record window overlaps the store (11)."""
        h, k0, n = self.NT // 2, self.store.k0, self.store.n
        return tuple(int(b) for b in np.arange(k0 // h, -(-(k0 + n) // h) + 1))


def geometry(prep: Any) -> Geometry:
    """The :class:`Geometry` of ``prep`` (a ``PreparedCase``)."""
    a = prep.arrays
    N, DT, NT = _int(a["N"]), float(a["DT"]), _int(a["NT"])
    first, nb = _int(a["first_band"]), _int(a["n_bands"])
    store = FourierDomain(N, DT, _int(a["store_offset"]), _int(a["store_n"]), "sigma0")
    window = FourierDomain(N, DT, _int(a["bin_offset"]), _int(a["n_bins"]), "sigma0")
    nominal = FourierDomain(N, DT, _int(a["bin_offset"]) + _int(a["freq_lo"]), _int(a["freq_n"]), "sigma0")
    lead, trail = CUT[prep.case]
    Nf = N // NT
    Nc = (NT - lead - trail) * Nf
    crop = FourierDomain(Nc, DT, _int(a["crop_k0"]), _int(a["crop_n"]), "sigma0")
    T, period = float(a["T"]), float(a["envelope_period"])
    return Geometry(prep.case, N, DT, T, NT, tuple(first + b for b in range(nb)), (lead, trail),
                    bool(np.isclose(T, period, rtol=0, atol=1e-6)), store, window, nominal, crop, lead * Nf)


def envelope(prep: Any) -> AnnualEnvelope:
    """The annual envelope of ``prep`` (``n_harm = 2``, period 365 d, ``basis_L`` shipped)."""
    a = prep.arrays
    period = float(a["envelope_period"])
    if period != ENVELOPE_PERIOD:
        raise ValueError(f"{prep.case}: envelope period {period} is not the release's {ENVELOPE_PERIOD}")
    return AnnualEnvelope(float(a["T"]), period=period, n_harm=N_HARM, basis_L=np.asarray(a["basis_L"], float))


def store_psd(prep: Any, geo: Geometry, env: Any) -> SeparablePSD:
    """``s_I + g(t) s_G`` tabulated on the store bins (both channels)."""
    a = prep.arrays
    return SeparablePSD(((TabulatedSpectrum(geo.store, np.asarray(a["store_sI"], float)), None),
                         (TabulatedSpectrum(geo.store, np.asarray(a["store_sG"], float)), env)), CHANNELS)


def pad_psd(prep: Any, geo: Geometry, env: Any) -> SeparablePSD:
    """The same on the padded record's store-band window (``Lambda_R``; ``pad_store_*``)."""
    a = prep.arrays
    dom = FourierDomain(geo.pad_grid.N, geo.DT, _int(a["pad_store_g0"]), _int(a["pad_store_G"]), "sigma0")
    return SeparablePSD(((TabulatedSpectrum(dom, np.asarray(a["pad_store_sI"], float)), None),
                         (TabulatedSpectrum(dom, np.asarray(a["pad_store_sG"], float)), env)), CHANNELS)


def pad_mask(prep: Any, geo: Geometry, data_map: FourierToWDM) -> PadMask:
    """GB ``pad_mask``: the rows of ``A`` with ``1 - rho > PAD_EDGE_TOL`` (99 at every duration)."""
    cells = geo.pad_kept
    m, n = np.asarray(cells.m), np.asarray(cells.n)
    b = m - geo.bands[0]
    R_bands = np.asarray(geo.store_bands)
    return PadMask.build(map_rows=np.asarray(data_map.C), rho=np.asarray(prep.arrays["rho_pad"], float)[b, n],
                         kept_band=b, kept_n=n, R_grid=geo.pad_grid, R_bands=R_bands, record_N=geo.N,
                         store_k0=geo.store.k0, store_n=geo.store.n,
                         k0=int(np.flatnonzero(R_bands == geo.bands[0])[0]), edge_tol=PAD_EDGE_TOL)


class NumpyTemplate:
    """A template callable whose output is a host NumPy array of the store's shape
    ``(2, store_n)`` (the GB "NumPy backend" computes templates on CPU-JAX and converts them,
    spec section 5.2).  Any other shape raises: a window-sized template would be read at the
    wrong bins by the bin selections, silently."""

    def __init__(self, template: Any, n_store: int) -> None:
        self.template, self.shape = template, (2, int(n_store))

    def __call__(self, p8: Any) -> np.ndarray:
        h = np.asarray(self.template(p8))
        if h.shape != self.shape:
            raise ValueError(f"the template has shape {h.shape}; the store needs {self.shape}")
        return h


def _diag_base(base: str, psd: SeparablePSD, cells: CellSet, c: int) -> DiagonalCovariance:
    var = BandAveragedVariance(psd, cells) if base == "band" else MidpointVariance(psd, cells)
    return DiagonalCovariance(var, cells.size, channel=c)


def likelihood(model: str, prep: Any, *, template: Any, resolved: Any) -> GaussianLikelihood | FourierWhittle:
    """``model`` of ``prep`` on ``resolved.backend`` (``resolved``: ``options.resolve("galactic_binary",
    model, backend, ...)``; on JAX ``template`` must be the ``GBTemplate`` pytree on the store, see
    :func:`signal_for`).  ``template``: ``p8 -> (2, store_n)`` complex in prepared units (the
    store window; ``waveform.GBTemplate(prep, backend=..., window="store")``, or a test's
    streamed table).  Rows are ``priors.names(model)``."""
    check_model(model)
    if model not in TARGET_MODELS[prep.case]:
        raise ValueError(f"{model!r} is not a target of galactic_binary/{prep.case}")
    a = prep.arrays
    geo = geometry(prep)
    env = envelope(prep)
    psd = store_psd(prep, geo, env)
    store = FourierBins(np.asarray(a["store_z"]), geo.store, ChannelAxis(CHANNELS))
    signal = signal_for(template, geo, resolved.backend)
    be = resolved.backend
    params = priors.params(model)
    lo, hi = np.asarray(a["prior_lo"], float), np.asarray(a["prior_hi"], float)

    def row_ok(theta: Any) -> Any:
        return priors.source_ok(theta[:8], lo, hi)

    base, edge = split_model(model)
    if base == "whittle":
        if edge == "raw":
            k = slice(geo.nominal.k0 - geo.store.k0, geo.nominal.k0 - geo.store.k0 + geo.nominal.n)
            return FourierWhittle("complex", store, map=BinSelect(geo.store, geo.nominal),
                                  sI=np.asarray(a["store_sI"], float)[:, k], sG=np.asarray(a["store_sG"], float)[:, k],
                                  signal=signal, params=params, row_ok=row_ok, backend=be)
        stored = tuple(int(v) for v in np.atleast_1d(a["matrix_cut"]))
        if stored != geo.cut:
            raise ValueError(f"{prep.case}: the prepared whittle_cut crop is for cut {stored}, not {geo.cut}")
        factor = geo.crop.N / geo.N
        return FourierWhittle("complex", store, map=FourierCrop(geo.store, geo.crop, geo.crop_start),
                              sI=factor * np.asarray(a["crop_sI"], float), sG=factor * np.asarray(a["crop_sG"], float),
                              signal=signal, params=params, row_ok=row_ok, backend=be)
    expanded = resolved.strategy == "expanded"
    if model == "exact":
        covs = tuple(HermitianCovariance(ExactFourierCovariance(psd, geo.nominal, channel=c),
                                         block=geo.NT // 2 if geo.periodic else None) for c in range(2))
        return GaussianLikelihood(store, BinSelect(geo.store, geo.nominal), ChannelStack(covs, len(ENV_NAMES)),
                                  ComplexGaussian(), signal=signal, params=params, strategy="expanded",
                                  row_ok=row_ok, backend=be)
    if edge == "pad":
        W = FourierToWDM(geo.pad_window, WDMDomain(geo.pad_kept))
        data_map = W.after(FourierZeroPad(geo.store, geo.pad_window))
        mask = pad_mask(prep, geo, data_map)
        R = CellSet.bands(geo.pad_grid, geo.store_bands)
        ppsd = pad_psd(prep, geo, env)
        covs = []
        for c in range(2):
            lam_R = (Eq45Covariance(ppsd, R, builder="separable", assembly="blocktri", layout=PAD_LAYOUT, channel=c)
                     if base == "tri" else _diag_base(base, ppsd, R, c))
            covs.append(ZeroPadded(lam_R, strategy="cell_congruence", mask=mask, n_bands=len(geo.bands),
                                   n_columns=geo.NT, factor=resolved.pad_factor))
        quad = "whiten"
    else:
        cells = geo.cut_cells if edge == "cut" else geo.cells
        data_map = FourierToWDM(geo.store, WDMDomain(cells), impl=resolved.cells)
        covs = []
        for c in range(2):
            if edge == "exact":
                e = geo.edge
                inner = cells.complement(e)
                interior = (Eq45Covariance(psd, inner, builder="separable", assembly="dense", channel=c)
                            if base == "tri" else _diag_base(base, psd, inner, c))
                rows = PushedRows(ExactFourierCovariance(psd, geo.window, channel=c),
                                  FourierToWDM(geo.window, WDMDomain(cells), restrict=True), e)
                schur = model in SCHUR and resolved.schur == "schur"
                covs.append(SchurEdge(interior, rows, e, assembly="schur" if schur else "dense", chol_impl="numpy",
                                      form="scaled" if schur else "solve"))
            elif base == "tri":
                covs.append(Eq45Covariance(psd, cells, builder="separable", assembly="dense", channel=c))
            else:
                covs.append(_diag_base(base, psd, cells, c))
        quad = "elementwise" if base in ("band", "mid") and edge in ("raw", "cut") else "whiten"
    return GaussianLikelihood(store, data_map, ChannelStack(tuple(covs), len(ENV_NAMES)), RealGaussian(),
                              signal=signal, params=params, strategy="expanded" if expanded else "residual",
                              quad_form=None if expanded else quad, row_ok=row_ok, backend=be)


#: The block layout of the pad's ``Lambda_R`` (store bands x ``NT + 16`` columns).  Only its band
#: blocks are read (``ZeroPadded`` factor ``"dense"``), identical in both layouts (bitwise on NumPy,
#: F-D3); ``"padded"`` is jittable, ``"ragged"`` (a host loop) is not.
PAD_LAYOUT = "padded"


def signal_for(template: Any, geo: Geometry, backend: str) -> Any:
    """The likelihood's signal.  NumPy: the template's host array (:class:`NumpyTemplate` in a
    ``StaticSignal``).  JAX: the ``waveform.GBTemplate`` pytree itself on the store window -- its
    orbit cache and ``DT sigma0`` are leaves, so they reach ``jit`` as arguments (spec section 3.3)
    and the template traces under ``jit``/``vmap``; any other callable is refused (it would be
    closed over, or fail on a tracer)."""
    if backend == "numpy":
        return StaticSignal(NumpyTemplate(template, geo.store.n))
    from .waveform import GBTemplate

    if not isinstance(template, GBTemplate):
        raise TypeError(f"the GB JAX build needs the waveform.GBTemplate pytree; got {type(template).__name__}")
    if (template.bin_lo, template.n_bins) != (geo.store.k0, geo.store.n):
        raise ValueError(f"the JAX template must be on the store window [{geo.store.k0}, {geo.store.k0 + geo.store.n}); "
                         f"got [{template.bin_lo}, {template.bin_lo + template.n_bins})")
    return template


def wrap(lik: Any, model: str, resolved: Any) -> Any:
    """``lik`` in its evaluator: ``"reference"`` -- the bare likelihood (each row factorises
    its own state, lineage C ``make_lnlike``); ``"cache"`` -- a ``NoiseStateCache`` of
    ``resolved.cache_size`` states (lineage C ``MatrixCache``/``EdgeSchurCache``/``FullCache``,
    keyed by the flat noise slice); ``"batched"`` (JAX) -- a ``BatchedLikelihood``, one
    ``jit(vmap)`` per call (lineage C ``gpu/jax_models.BatchedLikelihood``) counting no noise
    state: the legacy class had no counters and a diagonal or Whittle state has nothing to factor
    (``evaluations`` and ``pd_failed_rows`` still count)."""
    if resolved.evaluator == "reference":
        return lik
    if resolved.evaluator == "cache":
        if model not in CACHED:
            raise ValueError(f"{model!r} has no cached evaluator")
        return NoiseStateCache(lik, resolved.cache_size, batch_rows=resolved.batch_rows)
    if resolved.evaluator == "batched":
        if model in CACHED:
            raise ValueError(f"{model!r} is a cached model: its JAX evaluator is the cache")
        return BatchedLikelihood(lik, count_states=False)
    raise ValueError(f"unknown GB evaluator {resolved.evaluator!r}")


def build(case: str, model: str, *, backend: str = "numpy", options: ModelOptions = ModelOptions(),
          data_root: Any = None, template: Any = None) -> Any:
    """``registry.build("galactic_binary", case, model, ...)``: the shipped prepared case, the
    template (``waveform.GBTemplate(prep, backend=options.waveform_backend, window="store")``
    unless ``template`` is given -- a test's streamed gbonfly table), the model and its
    evaluator, on NumPy or JAX (S7: on JAX the template must be the ``GBTemplate`` pytree)."""
    r = resolve("galactic_binary", model, backend, options)
    prep = PreparedCase.load(case, data_root=data_root)
    if model not in TARGET_MODELS[prep.case]:
        raise ValueError(f"{model!r} is not a target of galactic_binary/{case}")
    if template is None:
        from .waveform import GBTemplate                   # binary_galactica (JAX) only when needed

        template = GBTemplate(prep, BW=TEMPLATE_BW, backend=r.waveform_backend, window="store")
    return wrap(likelihood(model, prep, template=template, resolved=r), model, r)
