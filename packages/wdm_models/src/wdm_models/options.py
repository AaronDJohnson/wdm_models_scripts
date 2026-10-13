r"""Model options and the per-backend strategy table (spec §3.2, §5.3).

**The one knob** is ``ModelOptions.strategy``:

* ``"historical"`` (default) -- each backend's golden algorithm: the NumPy column wrote the
  references (WT ``wdm_toys``), the JAX column ran the chains (WG ``wdmtoysgpu``);
* ``"reference"`` -- the NumPy column on either backend (separates backend differences from
  algorithm differences);
* ``"fast"`` -- every opt-in that has passed its gate (S8).  None has yet, so ``"fast"``
  resolves to ``"historical"`` (``FAST_OPT_INS`` is empty).

The other fields are per-job overrides (tests and benchmarks); ``None`` keeps the strategy's
choice.  ``cells``, ``gb_pad_factor`` and ``waveform_backend`` are the galactic binary's
(S5/S6); they are validated here and unused by the sinusoid.

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
=====================  ==============  ======================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .common.params import check_model, split_model

__all__ = ["FAST_OPT_INS", "ModelOptions", "Resolved", "STRATEGIES", "resolve"]

STRATEGIES = ("historical", "reference", "fast")
TRI_BUILDS = ("bessel", "gram", "interp")
LAYOUTS = ("ragged", "padded")
ROWS_ORDERS = ("fixed", "adaptive")
MARGINAL_FORMS = ("solve", "whiten")
GB_CELLS = ("dense", "fft")
GB_PAD_FACTORS = ("dense", "schur")
WAVEFORM_BACKENDS = ("fourier", "onfly")
DEVICES = ("cpu", "gpu")
#: Opt-in strategies that passed their gate, per toy (spec §3.2 "fast"; S8 fills it).
FAST_OPT_INS: dict[str, frozenset[str]] = {"sinusoid": frozenset()}


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
    waveform_backend: str = "fourier"
    cache_size: int | None = None
    exact_chunk: int | None = None
    whitened: bool = False
    layout: str | None = None
    phi_cache: bool | None = None
    staged: bool | None = None
    marginal_form: str | None = None
    batch_rows: bool | None = None

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}; got {self.strategy!r}")
        _one_of("tri_build", self.tri_build, TRI_BUILDS)
        _one_of("rows_order", self.rows_order, ROWS_ORDERS)
        _one_of("layout", self.layout, LAYOUTS)
        _one_of("marginal_form", self.marginal_form, MARGINAL_FORMS)
        _one_of("cells", self.cells, GB_CELLS)
        _one_of("gb_pad_factor", self.gb_pad_factor, GB_PAD_FACTORS)
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
                        None, cache, False, density)
    return Resolved(backend, device, "gram", "padded", "adaptive", True, device == "cpu",
                    64 if device == "cpu" else 192, "nan",
                    "whiten" if base == "tri" and edge in ("raw", "cut") else "solve",
                    (("alpha_d", 4.0),), cache, device != "cpu" and edge != "exact", density)


def resolve(toy: str, model: str, backend: str, options: ModelOptions = ModelOptions(), *,
            device: str | None = None) -> Resolved:
    """``options`` resolved for ``model`` of ``toy`` on ``backend`` (``"numpy"``/``"jax"``)."""
    check_model(model)
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {backend!r}")
    if toy != "sinusoid":
        raise KeyError(f"no strategy table for toy {toy!r} yet (galactic_binary arrives in S5/S6)")
    dev = _device(backend, device)
    strategy = "historical" if options.strategy == "fast" and not FAST_OPT_INS[toy] else options.strategy
    r = _sinusoid(model, backend, dev, strategy)
    over = {k: getattr(options, k) for k in ("tri_build", "layout", "rows_order", "phi_cache", "staged",
                                              "exact_chunk", "cache_size", "marginal_form", "batch_rows")
            if getattr(options, k) is not None}
    if over:
        r = replace(r, **over)
    if r.staged and not r.phi_cache:
        raise ValueError("staged=True needs phi_cache=True")
    return r
