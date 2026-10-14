r"""Model options and the per-backend strategy table (spec §3.2, §5.3).

**The one knob** is ``ModelOptions.strategy``:

* ``"historical"`` (default) -- each backend's golden algorithm: the NumPy column wrote the
  references (WT ``wdm_toys``), the JAX column ran the chains (WG ``wdmtoysgpu``);
* ``"reference"`` -- the NumPy column on either backend (separates backend differences from
  algorithm differences);
* ``"fast"`` -- every opt-in that has passed its gate (S8).  None has yet, so ``"fast"``
  resolves to ``"historical"`` (``FAST_OPT_INS`` is empty).

The other fields are per-job overrides (tests and benchmarks); ``None`` keeps the strategy's
choice.  ``cells``, ``gb_pad_factor``, ``evaluator`` and ``waveform_backend`` are the
galactic binary's (S6, :class:`GBResolved`); they are validated here and unused by the
sinusoid.  ``waveform_backend`` defaults to ``"onfly"`` (TDI on the fly, the release
default everywhere: user ruling 2026-10-11); every golden check passes ``"fourier"``
explicitly (the stored chains' templates).

Sinusoid table (:func:`resolve`; ``device`` is ``jax.default_backend()`` unless given):

=====================  ==============  ======================================================
setting                NumPy           JAX
=====================  ==============  ======================================================
``tri_build``          ``bessel``      ``gram``
``layout``             ``ragged``      ``padded``
``rows_order``         ``fixed``       ``adaptive`` (P from ``cheb_order``)
``phi_cache``          off             on
``staged``             off             on CPU only (XLA-CPU irfft-in-jit trap, spec §3.4)
``exact_chunk``        --              64 on CPU, 192 on GPU (WG)
``exact_on_fail``      ``raise``       ``nan`` (-> -inf)
``marginal_form``      ``solve``       ``whiten`` for ``tri_raw``/``tri_cut``, else ``solve``
``noise_clamp``        none            ``alpha_d`` clipped to +-4 (WG ``ALPHA_D_CLAMP``)
``cache_size``         32              16 on GPU, 32 on CPU
``batch_rows``         off             on GPU except edge-exact (WG ``batch_rungs``)
``density_order``      ``folded`` for ``exact`` (WT ``FastFullCovariance``), else ``default``
``correction``         ``chebval2d``   ``table`` (the exact model's bandcov correction: WT
                                       ``chebval2d`` vs WG ``_cheb_values`` tables, traceable)
``pad_form``           ``solve``       ``reduce`` for the pads (WG ``cQ, Gu`` state, E-C3)
``schur_form``         ``solve``       ``scaled`` for band/mid_exact (diagonal interior: WG
                                       ``B/v`` on the fly, E-C4), ``solve`` for tri_exact
=====================  ==============  ======================================================

Galactic-binary table (:class:`GBResolved`; spec section 3.2, lineage C):

=====================  ==================  ==================================================
setting                NumPy               JAX
=====================  ==================  ==================================================
``evaluator``          ``reference``       ``cache`` for :data:`GB_CACHED`, else ``batched``
``strategy``           ``residual``        ``expanded`` for ``cache``; ``exact`` always
                                           ``expanded`` (``fullcache``'s reference form)
``cells``              ``dense``           ``dense``
``pad_factor``         ``dense``           ``dense``
``schur``              ``dense``           ``schur`` for band/mid_exact (``EdgeSchurCache``)
``cache_size``         24                  24
``batch_rows``         off                 on for ``cache`` (GB ``batch_signal``)
=====================  ==================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .common.params import check_model, split_model

__all__ = ["FAST_OPT_INS", "GBResolved", "GB_CACHED", "ModelOptions", "Resolved", "STRATEGIES", "resolve"]

STRATEGIES = ("historical", "reference", "fast")
TRI_BUILDS = ("bessel", "gram", "interp")
LAYOUTS = ("ragged", "padded")
ROWS_ORDERS = ("fixed", "adaptive")
MARGINAL_FORMS = ("solve", "whiten")
GB_CELLS = ("dense", "fft")
GB_PAD_FACTORS = ("dense", "schur")
WAVEFORM_BACKENDS = ("fourier", "onfly")
GB_EVALUATORS = ("reference", "cache", "batched")
#: GB models whose noise state lineage C cached (``matrix.CACHED`` without ``*_win``, and
#: ``exact``: ``fullcache.FullCache``) -- the ``evaluator="cache"`` models.
GB_CACHED = ("tri_raw", "tri_cut", "tri_pad", "tri_exact", "band_exact", "mid_exact", "band_pad", "mid_pad",
             "exact")
DEVICES = ("cpu", "gpu")
#: Opt-in strategies that passed their gate, per toy (spec §3.2 "fast"; S8 fills it).
FAST_OPT_INS: dict[str, frozenset[str]] = {"sinusoid": frozenset(), "galactic_binary": frozenset()}


def _one_of(name: str, value: object, allowed: tuple) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {allowed} or None; got {value!r}")


@dataclass(frozen=True)
class ModelOptions:
    """Strategy knob plus per-job overrides (``None``: the strategy decides)."""

    strategy: Literal["historical", "reference", "fast"] = "historical"
    tri_build: str | None = None
    rows_order: str | None = None
    cells: str | None = None
    gb_pad_factor: str | None = None
    waveform_backend: str = "onfly"
    cache_size: int | None = None
    exact_chunk: int | None = None
    whitened: bool = False
    layout: str | None = None
    phi_cache: bool | None = None
    staged: bool | None = None
    marginal_form: str | None = None
    batch_rows: bool | None = None
    evaluator: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}; got {self.strategy!r}")
        _one_of("tri_build", self.tri_build, TRI_BUILDS)
        _one_of("rows_order", self.rows_order, ROWS_ORDERS)
        _one_of("layout", self.layout, LAYOUTS)
        _one_of("marginal_form", self.marginal_form, MARGINAL_FORMS)
        _one_of("cells", self.cells, GB_CELLS)
        _one_of("gb_pad_factor", self.gb_pad_factor, GB_PAD_FACTORS)
        _one_of("evaluator", self.evaluator, GB_EVALUATORS)
        if self.waveform_backend not in WAVEFORM_BACKENDS:
            raise ValueError(f"waveform_backend must be one of {WAVEFORM_BACKENDS}; got "
                             f"{self.waveform_backend!r}")
        for name in ("cache_size", "exact_chunk"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} must be a positive int or None; got {v!r}")
        if self.staged and self.phi_cache is False:
            raise ValueError("staged=True runs the phi_cache stages: it needs phi_cache=True")


@dataclass(frozen=True)
class Resolved:
    """The concrete choices for one model on one backend (what ``models.py`` builds from)."""

    backend: str
    device: str
    tri_build: str
    layout: str
    rows_order: str
    phi_cache: bool
    staged: bool
    exact_chunk: int | None
    exact_on_fail: str
    marginal_form: str
    noise_clamp: tuple[tuple[str, float], ...] | None
    cache_size: int
    batch_rows: bool
    density_order: str
    #: S4 (E-D2): the ``exact`` model's bandcov correction order (``BandCorrection.algorithm``)
    #: and the domainia forms of the pad marginal (``ZeroPadded``/``FlatMarginal``) and of the
    #: exact-edge Schur complement (``SchurEdge``) -- NumPy keeps WT's ``solve`` everywhere.
    correction: str
    pad_form: str
    schur_form: str


def _device(backend: str, device: str | None) -> str:
    if backend == "numpy":
        return "cpu"
    if device is None:
        import jax                                    # only when a JAX build is asked for

        device = "cpu" if jax.default_backend() == "cpu" else "gpu"
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}; got {device!r}")
    return device


def _sinusoid(model: str, backend: str, device: str, strategy: str) -> Resolved:
    base, edge = split_model(model)
    numpy_column = backend == "numpy" or strategy == "reference"
    density = "folded" if model == "exact" else "default"
    cache = 32 if device == "cpu" else 16
    if numpy_column:
        return Resolved(backend, device, "bessel", "ragged", "fixed", False, False, None, "raise", "solve",
                        None, cache, False, density, "chebval2d", "solve", "solve")
    return Resolved(backend, device, "gram", "padded", "adaptive", True, device == "cpu",
                    64 if device == "cpu" else 192, "nan",
                    "whiten" if base == "tri" and edge in ("raw", "cut") else "solve",
                    (("alpha_d", 4.0),), cache, device != "cpu" and edge != "exact", density, "table",
                    "reduce" if edge == "pad" else "solve",
                    "scaled" if edge == "exact" and base in ("band", "mid") else "solve")


@dataclass(frozen=True)
class GBResolved:
    """The galactic binary's concrete choices for one model on one backend (spec section 3.2;
    ``galactic_binary.models`` builds from it).

    ``evaluator``: ``"reference"`` -- lineage C's NumPy reference ``make_lnlike`` (each row
    factorises its own state; the ``residual`` strategy, except ``exact``, whose reference
    ``fullcache.make_lnlike`` is the expanded form); ``"cache"`` -- lineage C's cache classes
    (``MatrixCache`` / ``EdgeSchurCache`` / ``FullCache``: ``expanded`` strategy behind a
    ``NoiseStateCache`` of ``cache_size`` states); ``"batched"`` -- the uncached models on JAX
    (``gpu/jax_models.BatchedLikelihood``).  ``schur``: the band/mid_exact assembly, ``"dense"``
    (``covariance_builder``'s assembled matrix) or ``"schur"`` (``EdgeSchurCache``: the Schur
    complement of the diagonal interior, ``B / v`` on the fly)."""

    backend: str
    device: str
    cells: str
    pad_factor: str
    schur: str
    strategy: str
    evaluator: str
    cache_size: int
    batch_rows: bool
    waveform_backend: str


def _galactic_binary(model: str, backend: str, device: str, strategy: str, options: ModelOptions) -> GBResolved:
    """GB table: NumPy (``historical``/``reference``) = lineage C's NumPy reference; JAX
    ``historical`` = ``gpu/jax_models.build`` (the cache classes, batched rows on every device:
    GB ``batch_signal``).  ``evaluator`` overrides; ``"cache"`` is refused outside
    :data:`GB_CACHED`."""
    numpy_column = backend == "numpy" or strategy == "reference"
    if options.evaluator is not None:
        evaluator = options.evaluator
    elif numpy_column:
        evaluator = "reference"
    else:
        evaluator = "cache" if model in GB_CACHED else "batched"
    if evaluator == "cache" and model not in GB_CACHED:
        raise ValueError(f"evaluator='cache' applies to the cached GB models {GB_CACHED}; not {model!r}")
    if evaluator == "batched" and backend != "jax":
        raise ValueError("evaluator='batched' is the JAX evaluator of the uncached GB models")
    cached = evaluator == "cache"
    return GBResolved(backend, device, options.cells or "dense", options.gb_pad_factor or "dense",
                      "schur" if cached and model in ("band_exact", "mid_exact") else "dense",
                      "expanded" if cached or model == "exact" else "residual", evaluator,
                      options.cache_size or 24, bool(options.batch_rows) if options.batch_rows is not None
                      else backend == "jax" and cached, options.waveform_backend)


def resolve(toy: str, model: str, backend: str, options: ModelOptions = ModelOptions(), *,
            device: str | None = None) -> Resolved | GBResolved:
    """``options`` resolved for ``model`` of ``toy`` on ``backend`` (``"numpy"``/``"jax"``):
    a :class:`Resolved` for the sinusoid, a :class:`GBResolved` for the galactic binary."""
    check_model(model)
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {backend!r}")
    if toy not in FAST_OPT_INS:
        raise KeyError(f"no strategy table for toy {toy!r}; the toys are {tuple(FAST_OPT_INS)}")
    dev = _device(backend, device)
    strategy = "historical" if options.strategy == "fast" and not FAST_OPT_INS[toy] else options.strategy
    if toy == "galactic_binary":
        return _galactic_binary(model, backend, dev, strategy, options)
    r = _sinusoid(model, backend, dev, strategy)
    over = {k: getattr(options, k) for k in ("tri_build", "layout", "rows_order", "phi_cache", "staged",
                                              "exact_chunk", "cache_size", "marginal_form", "batch_rows")
            if getattr(options, k) is not None}
    if over:
        r = replace(r, **over)
    if r.staged and not r.phi_cache:
        raise ValueError("staged=True needs phi_cache=True")
    return r
