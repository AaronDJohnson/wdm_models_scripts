"""Rebuild the shipped prepared galactic-binary cases from the shipped segments and stores, and
compare them with the shipped files (spec sections 5.2 ``prepare.py``, 6 and 7.3 "prepared
regeneration"); never writes the release tree.

From the release root, after ``source env/activate.sh``:

    $PY -P -m scripts.galactic_binary.prepare --from-shipped [--case gb_3mo ...] \\
        [--engine release|legacy] [--out DIR] [--report PATH] [--data-root DIR]

``--engine release`` (default) runs ``wdm_models.galactic_binary.prepare.prepare`` (``mu`` and
``calc_Nt`` refitted) and ``.add_matrix`` on ``data/galactic_binary/segments`` (segment + store
of each case) into ``--out`` (default: a fresh directory under ``$TMPDIR``).  ``--engine
legacy`` runs the frozen ``toy_models/galactic_binary/prepare.py`` on the same shipped inputs in
a lineage-C subprocess (``scripts/golden/workers/gbw_prepared.py``; this machine only; an
audit).  Either way each rebuilt ``<case>.npz`` is compared with
``data/galactic_binary/prepared/<case>.npz`` (sha256 checked against ``data/MANIFEST.json``)
under :data:`POLICY`:

* every paper key (``prepared.PAPER_KEYS``) must be in both files; the deterministic ones are
  ``array_equal`` with the same dtype and shape;
* the fits ``mu_fit`` and ``mu_freq`` (``prepared.FIT_KEYS``, L-BFGS): ``max |delta| <= 1e-6``;
* ``nt_star`` (the third fit key): reported, never gated -- Sec. VII A's ``NT*`` is
  informational (``NT`` is the configuration constant 240 / 438 / 900) and the shipped files
  carry it from an earlier, tiling-dependent estimate (253.63 / 447.78 / 928.49); the fixed-band
  ``calc_Nt`` gives 246.87 / 444.39 / 925.78;
* the eight legacy-only keys (:data:`LEGACY_ONLY`: the frequency-padded geometry, ``rho_win``
  and the absolute ``matrix_store`` path, read only by the dropped models) are compared when
  the rebuilt file has them (the legacy engine: ``array_equal``, ``matrix_store`` by file name)
  and reported as not regenerated when it does not (the release port);
* any other key, in either file, fails;

and each ``<case>_report.json`` under :data:`REPORT_POLICY` (``snr_full`` and the residual
chi^2 to 1e-9 relative, ``mu_fit``/``mu_freq`` to 1e-6, the window geometry and the
foreground's parameters exactly, the segment by file name; the ``calc_Nt`` fields and the
foreground's model label reported).

Measured 2026-10-13 (wdmrel, all three cases): the release engine passes in ~32 s -- all 50
deterministic paper keys ``array_equal``, ``mu_fit`` within 1.7e-16 / 1.1e-16 / 7.2e-16,
``mu_freq`` 0.0, ``snr_full`` within 2.5e-16 relative, ``nt_star`` 246.86598304856258 /
444.38939984813663 / 925.7765186111524; the legacy engine passes in ~34 s with every compared
key equal (``mu_fit`` and ``mu_freq`` 0.0, the eight legacy-only keys too) and ``nt_star``
246.86598221011158 / 444.3893998481366 / 925.7765186242784.

Exit status 0: every case passes.  1: a comparison failed, or a rebuild or a comparison raised
(the reason is the report's ``error``; ``--report`` is written either way).  2: usage, including
an ``--out``/``--report`` inside the release tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REL = Path(__file__).resolve().parents[2]
#: The release sources, in front of sys.path unless already there (``__file__``-relative, as
#: ``scripts/sinusoid/validate.py``: ``python -P`` stays cwd-free).
SOURCES = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
for _src in reversed(SOURCES):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from wdm_models.common.validation import manifest_sha256, resolve_data_root  # noqa: E402
from wdm_models.galactic_binary.prepared import FIT_KEYS, PAPER_KEYS  # noqa: E402

CASES = ("gb_3mo", "gb_6mo", "gb_12mo")
#: Keys of the shipped prepared files that only the dropped models read; the release port
#: (``prepare.prepare`` + ``add_matrix``) does not write them, the legacy engine does.
LEGACY_ONLY = ("bands", "g0", "G", "phi", "sI_grid", "sG_grid", "rho_win", "matrix_store")
#: The comparison policy of the prepared npz (module docstring); ``fit_atol`` and ``info``
#: together are ``prepared.FIT_KEYS``.
POLICY: dict[str, Any] = {
    "fit_atol": {"mu_fit": 1e-6, "mu_freq": 1e-6},
    "info": {"nt_star": "Sec. VII A's NT* (informational; the shipped value predates the fixed-band calc_Nt)"},
    "paths": ("matrix_store",),
    "legacy_only": LEGACY_ONLY,
}
assert set(POLICY["fit_atol"]) | set(POLICY["info"]) == set(FIT_KEYS)
#: The comparison policy of ``<case>_report.json``.
REPORT_POLICY: dict[str, Any] = {
    "exact": ("case", "N", "n_bins", "window_band", "bands", "window_bins", "frequency_bins",
              "source_band_bins", "NT", "T_days", "envelope_period_days"),
    "rtol": {"snr_full": 1e-9, "residual_chi2_per_bin": 1e-9},
    "atol": {"mu_fit": 1e-6, "mu_freq": 1e-6},
    "paths": ("segment",),
    "foreground": "params_lnA_lnf1_alpha_lnfk_lnf2",
}


def _max_abs(a: Any, b: Any) -> float:
    """``max |a - b|`` of two numeric arrays (complex included); ``inf`` for another shape, a
    non-numeric array or a non-finite difference."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
        return float("inf")
    if not a.size:
        return 0.0
    d = np.abs(a.astype(np.result_type(a, b, float)) - b)
    return float(np.max(d)) if np.all(np.isfinite(d)) else float("inf")


def _same(key: str, a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """``array_equal`` with the same dtype and shape (a path key: the same file name)."""
    if key in POLICY["paths"]:
        ok = Path(str(a)).name == Path(str(b)).name
        return ok, "same file name" if ok else f"{Path(str(a)).name} != {Path(str(b)).name}"
    ok = a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))
    if ok:
        return True, "equal"
    if (a.dtype, a.shape) != (b.dtype, b.shape):
        return False, f"{a.dtype}{a.shape} vs {b.dtype}{b.shape}"
    return False, f"max |delta| {_max_abs(a, b):.3e}"


def compare_prepared(new: Mapping[str, np.ndarray], shipped: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """``new`` (a rebuilt prepared case) against ``shipped`` under :data:`POLICY`: the gated
    paper keys (``"equal"`` or the fit's delta), the legacy-only keys, the reported ``info``,
    the problems and ``ok``."""
    rep: dict[str, Any] = {"keys": {}, "legacy_only": {}, "info": {}, "problems": []}
    for key in sorted(set(new) | set(shipped)):
        if key not in PAPER_KEYS and key not in LEGACY_ONLY:
            rep["problems"].append(f"{key} is not a prepared key ({'rebuilt' if key in new else 'shipped'} file)")
            continue
        if key in LEGACY_ONLY:
            if key not in new:
                rep["legacy_only"][key] = "not regenerated"
                continue
            if key not in shipped:
                rep["problems"].append(f"{key} is only in the rebuilt file")
                continue
            ok, what = _same(key, np.asarray(new[key]), np.asarray(shipped[key]))
            rep["legacy_only"][key] = what
            if not ok:
                rep["problems"].append(f"{key} (legacy-only) is not the shipped array ({what})")
            continue
        if key not in new or key not in shipped:
            rep["problems"].append(f"{key} is only in the {'rebuilt' if key in new else 'shipped'} file")
            continue
        a, b = np.asarray(new[key]), np.asarray(shipped[key])
        if key in POLICY["info"]:
            rep["info"][key] = {"rebuilt": a.tolist(), "shipped": b.tolist(), "why": POLICY["info"][key]}
            continue
        if key in POLICY["fit_atol"]:
            delta, gate = _max_abs(a, b), POLICY["fit_atol"][key]
            rep["keys"][key] = delta
            if not delta <= gate:
                rep["problems"].append(f"{key} differs by {delta:.3e} (gate {gate:.0e})")
            continue
        ok, what = _same(key, a, b)
        rep["keys"][key] = what
        if not ok:
            rep["problems"].append(f"{key} is not array_equal to the shipped array ({what})")
    rep["ok"] = not rep["problems"]
    return rep


def compare_report(new: Mapping[str, Any], shipped: Mapping[str, Any]) -> dict[str, Any]:
    """``<case>_report.json`` under :data:`REPORT_POLICY`; every other key is reported."""
    rep: dict[str, Any] = {"keys": {}, "info": {}, "problems": []}
    P = REPORT_POLICY
    for key in sorted(set(new) | set(shipped)):
        a, b = new.get(key), shipped.get(key)
        if key in P["exact"]:
            ok = key in new and key in shipped and a == b
            rep["keys"][key] = "equal" if ok else f"{a!r} vs {b!r}"
        elif key in P["rtol"]:
            delta = _max_abs(a, b) / max(float(np.max(np.abs(np.asarray(b, dtype=float)))), 1e-300)
            ok = delta <= P["rtol"][key]
            rep["keys"][key] = delta
        elif key in P["atol"]:
            delta = _max_abs(a, b)
            ok = delta <= P["atol"][key]
            rep["keys"][key] = delta
        elif key in P["paths"]:
            ok = a is not None and b is not None and Path(str(a)).name == Path(str(b)).name
            rep["keys"][key] = "same file name" if ok else f"{a!r} vs {b!r}"
        elif key == "foreground":
            fa, fb = (x if isinstance(x, Mapping) else {} for x in (a, b))
            pa, pb = fa.get(P["foreground"]), fb.get(P["foreground"])
            ok = pa is not None and pa == pb
            rep["keys"][key] = "equal parameters" if ok else f"{pa!r} vs {pb!r}"
            rep["info"]["foreground_model"] = {"rebuilt": fa.get("model"), "shipped": fb.get("model")}
        else:
            rep["info"][key] = {"rebuilt": a, "shipped": b}
            continue
        if not ok:
            rep["problems"].append(f"report {key}: {rep['keys'][key]}")
    rep["ok"] = not rep["problems"]
    return rep


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def shipped_case(case: str, data_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The shipped prepared arrays and report of ``case`` (sha256 = ``data/MANIFEST.json``)."""
    out = []
    for rel in (f"galactic_binary/prepared/{case}.npz", f"galactic_binary/prepared/{case}_report.json"):
        path = data_root / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest_sha256(data_root, rel):
            raise RuntimeError(f"{path} does not match data/MANIFEST.json")
        out.append(path)
    return _load_npz(out[0]), json.loads(out[1].read_text())


def rebuild_release(cases: Sequence[str], out_dir: Path, data_root: Path) -> dict[str, Path]:
    """The release ``prepare.prepare`` (fits refitted) + ``prepare.add_matrix`` on the shipped
    inputs; ``<case>.npz`` and ``<case>_report.json`` in ``out_dir``."""
    from wdm.backend import enable_x64
    from wdm_models.galactic_binary import prepare as P

    enable_x64()                                     # fit_mu (JAX) requires float64
    out = {}
    for case in cases:
        P.prepare(case, out_dir=out_dir, data_root=data_root)       # BASE_KEYS + <case>_report.json
        out[case] = Path(P.add_matrix(case, out_dir=out_dir, data_root=data_root))    # + MATRIX_KEYS
    return out


def rebuild_legacy(cases: Sequence[str], out_dir: Path, data_root: Path) -> dict[str, Path]:
    """The frozen ``prepare.py`` on the shipped inputs (lineage C), copied into ``out_dir``."""
    from scripts.golden.lineage import run_worker
    from scripts.golden.paths import ROOT, legacy_available

    if not legacy_available():
        raise RuntimeError("--engine legacy needs the frozen trees (this machine only)")
    work = run_worker("C", "gb_legacy", "prepared", "--workspace", str(ROOT), "--data-dir", str(data_root),
                      "--case", *cases, timeout=3600.0)
    paths = {}
    for case in cases:
        for name in (f"{case}.npz", f"{case}_report.json"):
            shutil.copyfile(work / name, out_dir / name)
        paths[case] = out_dir / f"{case}.npz"
    return paths


ENGINES = {"release": rebuild_release, "legacy": rebuild_legacy}


def run(cases: Sequence[str], engine: str, out_dir: Path, data_root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        paths = ENGINES[engine](cases, out_dir, data_root)
    except Exception as exc:                                  # fail closed, with the reason
        return {"engine": engine, "ok": False, "error": f"rebuild raised {type(exc).__name__}: {exc}",
                "cases": {}}
    rep: dict[str, Any] = {"engine": engine, "out": str(out_dir), "seconds": time.perf_counter() - t0,
                           "cases": {}}
    try:
        for case in cases:
            npz, report = shipped_case(case, data_root)
            new_report = json.loads((paths[case].parent / f"{case}_report.json").read_text())
            rep["cases"][case] = {"npz": compare_prepared(_load_npz(paths[case]), npz),
                                  "report": compare_report(new_report, report)}
    except Exception as exc:          # a rebuilt file missing, a shipped one off the manifest: fail closed
        return dict(rep, ok=False, error=f"comparison raised {type(exc).__name__}: {exc}")
    rep["ok"] = all(c["npz"]["ok"] and c["report"]["ok"] for c in rep["cases"].values())
    return rep


def summary_line(case: str, engine: str, c: Mapping[str, Any]) -> str:
    """One case's verdict: paper keys equal, the fits' deltas, ``nt_star``, the legacy-only keys."""
    keys, legacy = c["npz"]["keys"], c["npz"]["legacy_only"]
    exact = [k for k in keys if k not in POLICY["fit_atol"]]
    nt = c["npz"]["info"].get("nt_star", {})
    regenerated = sum(v != "not regenerated" for v in legacy.values())
    return (f"[prepare] {'PASS' if c['npz']['ok'] and c['report']['ok'] else 'FAIL'}: {case} ({engine}) "
            f"{sum(keys[k] == 'equal' for k in exact)}/{len(exact)} deterministic paper keys array_equal, "
            f"mu_fit delta {keys.get('mu_fit')}, mu_freq delta {keys.get('mu_freq')}, nt_star "
            f"{nt.get('rebuilt')} (shipped {nt.get('shipped')}), legacy-only keys "
            + (f"{sum(v in ('equal', 'same file name') for v in legacy.values())}/{len(legacy)} equal"
               if regenerated else f"{len(legacy)} not regenerated")
            + f", snr_full delta {c['report']['keys'].get('snr_full')}")


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:                      # spec section 1, guard 5
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(prog="python -P -m scripts.galactic_binary.prepare",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-shipped", action="store_true", required=True,
                    help="rebuild from the shipped segments and stores (the only mode)")
    ap.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--engine", choices=sorted(ENGINES), default="release")
    ap.add_argument("--out", type=Path, default=None, help="output directory (outside the release tree)")
    ap.add_argument("--report", type=Path, default=None, help="JSON report (outside the release tree)")
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    for name in ("out", "report"):
        p = getattr(a, name)
        if p is not None and p.resolve().is_relative_to(REL.resolve()):
            ap.error(f"--{name} must lie outside the release tree")
    out = (a.out.resolve() if a.out is not None
           else Path(tempfile.mkdtemp(prefix="gb_prepare_", dir=Path(tempfile.gettempdir()).resolve())))
    out.mkdir(parents=True, exist_ok=True)
    rep = run(a.case, a.engine, out, resolve_data_root(a.data_root).resolve())
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    if "error" in rep:
        print(f"[prepare] FAIL: {a.engine} {rep['error']}", file=sys.stderr)
        return 1
    for case, c in rep["cases"].items():
        print(summary_line(case, a.engine, c))
        for p in c["npz"]["problems"] + c["report"]["problems"]:
            print(f"[prepare]   {p}", file=sys.stderr)
    print(f"[prepare] {'PASS' if rep['ok'] else 'FAIL'}: {len(rep['cases'])} case(s), {rep['seconds']:.0f} s; "
          f"output {out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
