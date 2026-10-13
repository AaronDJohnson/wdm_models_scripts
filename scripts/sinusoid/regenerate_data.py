"""Regenerate one sinusoid case's shipped data and compare it with the shipped file (spec
sections 5.1 ``data.regenerate`` and 6); evaluates, never samples, never writes the release tree.

From the release root, after ``source env/activate.sh``:

    $PY -P -m scripts.sinusoid.regenerate_data --case manuscript_mid --out /outside/the/release/dir

**Heavy: ask before running** (spec section 6, "CPU, heavy, ask first").  ``data.regenerate``
builds the dense 16384 x 16384 voice factor of the exact time-domain covariance (2 GB, plus
temporaries) and draws ``n = G z`` in WT's voice order with ``default_rng(seed)`` (seed 1, the
draw the three cases share), then adds the release signal.  The draw is not bitwise at this
size; the comparison with the shipped ``toy_data.npz`` (``data.load``: sha256 and every
consistency check) requires ``h`` bit for bit, the shipped seed, and
``max|n_new - n| / max|n|`` and ``max|d_new - d| / max|d|`` at most :data:`N_RTOL` (spec
section 6: 1e-12; ~8.9e-14 was measured when the data were made).  Outputs, both under
``--out``: ``<case>_regenerated.npz`` (``d``, ``h``, ``n``, ``seed``) and
``<case>_regenerate_report.json``.

Exit status 0: inside the gate.  1: outside it (or a non-finite value).  2: usage, including
an ``--out`` inside the release tree.
"""

from __future__ import annotations

import argparse
import sys
import time
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

from wdm_models.sampling.validate import write_report  # noqa: E402  (after the path bootstrap)
from wdm_models.sinusoid import data  # noqa: E402
from wdm_models.sinusoid.config import CASE_NAMES  # noqa: E402

#: Spec section 6: the regenerated noise agrees with the shipped draw to <= 1e-12 relative.
N_RTOL = 1e-12


def compare(new: dict[str, Any], shipped: data.SinusoidData, *, rtol: float = N_RTOL) -> dict[str, Any]:
    """The regenerated ``{"d", "h", "n", "seed"}`` against the shipped case: per array the
    largest absolute and relative (to the shipped array's largest magnitude) difference,
    ``h_bitwise``, the problems and ``ok``.  Fail-closed: a shape mismatch, a non-finite
    value, another seed, an ``h`` that is not bit for bit, or ``n``/``d`` beyond ``rtol``."""
    rep: dict[str, Any] = {"rtol": rtol, "seed": int(new["seed"]), "problems": []}
    problems = rep["problems"]
    if rep["seed"] != shipped.seed:
        problems.append(f"seed {rep['seed']} is not the shipped draw's seed {shipped.seed}")
    for key in ("d", "h", "n"):
        a, b = np.asarray(new[key], dtype=np.float64), getattr(shipped, key)
        if a.shape != b.shape:
            problems.append(f"{key} has shape {a.shape}, the shipped {b.shape}")
            rep[f"{key}_max_abs"] = rep[f"{key}_rel"] = float("inf")
            continue
        if not np.all(np.isfinite(a)):
            problems.append(f"{key} has non-finite values")
        diff = float(np.max(np.abs(a - b))) if np.all(np.isfinite(a)) else float("inf")
        rep[f"{key}_max_abs"], rep[f"{key}_rel"] = diff, diff / float(np.max(np.abs(b)))
    rep["h_bitwise"] = bool(np.shape(new["h"]) == shipped.h.shape and np.array_equal(new["h"], shipped.h))
    if not rep["h_bitwise"]:
        problems.append("h is not the shipped signal bit for bit")
    for key in ("n", "d"):
        if not rep[f"{key}_rel"] <= rtol:
            problems.append(f"{key} differs by {rep[f'{key}_rel']:.3e} relative (gate {rtol:.0e})")
    rep["problems"] = sorted(set(problems), key=problems.index)
    rep["ok"] = not rep["problems"]
    return rep


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:                      # spec section 1, guard 5
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(prog="python -P -m scripts.sinusoid.regenerate_data",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=CASE_NAMES, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (outside the release tree)")
    ap.add_argument("--seed", type=int, default=data.NOISE_SEED)
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.is_relative_to(REL.resolve()):
        ap.error("--out must lie outside the release tree")
    shipped = data.load(a.case, data_root=a.data_root)
    t0 = time.perf_counter()
    new = data.regenerate(a.case, seed=a.seed)
    rep = {"case": a.case, **compare(new, shipped), "seconds": time.perf_counter() - t0}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.case}_regenerated.npz", d=new["d"], h=new["h"], n=new["n"], seed=new["seed"])
    write_report(out / f"{a.case}_regenerate_report.json", rep)
    verdict = "PASS" if rep["ok"] else "FAIL"
    print(f"[regenerate_data] {verdict}: {a.case} n {rep['n_rel']:.3e} / d {rep['d_rel']:.3e} relative "
          f"(gate {N_RTOL:.0e}), h bitwise {rep['h_bitwise']}, {rep['seconds']:.0f} s; output {out}")
    for p in rep["problems"]:
        print(f"[regenerate_data]   {p}", file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
