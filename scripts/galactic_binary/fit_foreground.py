"""Resolve the Sangria galactic binaries and fit the GLASS foreground for each duration (spec
sections 5.2 ``pipeline/foreground_fit.py`` and 6); never writes the release tree.

From the release root, after ``source env/activate.sh``:

    $PY -P -m scripts.galactic_binary.fit_foreground --sangria-h5 PATH --out DIR \\
        [--months 3 6 12] [--snr 7] [--backend onfly|fourier] [--compare-shipped] [--report PATH]

**Approval-gated** (spec section 6): reads the catalogues of all 29,000,036 Sangria galactic
binaries and their TDI from the private ``LDC2_sangria_training_v2.h5``, generates the
45,443 / 63,898 / 89,366 candidates of 3 / 6 / 12 months, iterates, and runs the GLASS MAP
(impulse PT, 100000 iterations x 8 temperatures, seed 0; ~30-40 s per duration).  Needs
``h5py`` (``[pipeline]``) and ``impulse`` (``[sampling]``, impulse-mcmc v2 964283c), checked
before the HDF5 is opened.  Writes ``foreground_<m>mo_snr<snr>.npz`` and ``.json`` into ``--out``.

The release default waveform backend is ``onfly``; ``--backend fourier`` (gbonfly's algorithm)
reproduces the shipped fits, and ``--compare-shipped`` then gates each written npz against
``data/galactic_binary/foreground/foreground_<m>mo_snr7.npz`` (``foreground_fit.compare_fit``:
the resolved set, catalogue rows, block geometry exactly; SNRs and the periodogram to 1e-9;
the MAP to 1e-6).  With ``--backend onfly`` the comparison is reported, not gated.

Expected with ``--backend fourier --compare-shipped`` (from the shipped fits; the end-to-end run
is not done here): 3,089 / 5,784 / 10,126 resolved binaries after 11 / 18 / 30 iterations, proxy
check below SNR 3.86 / 3.96 / 3.43; the MAP of ``map_fit_binned`` on the shipped binned
periodogram with these settings reproduces the shipped ``fg_params`` and ``fg_sd`` bit for bit
(measured with impulse 964283c, 2026-10-13).

Exit status 0: fitted (and, with ``--compare-shipped --backend fourier``, inside the gate).
1: a fit raised or a gated comparison failed.  2: usage, including a missing ``impulse``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[2]
SOURCES = tuple(REL / "packages" / p / "src" for p in ("wdm", "domainia", "wdm_models"))
for _src in reversed(SOURCES):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import numpy as np  # noqa: E402

from wdm_models.common.validation import manifest_sha256, resolve_data_root  # noqa: E402
from wdm_models.galactic_binary.pipeline import SHIPPED_BACKEND  # noqa: E402
from wdm_models.galactic_binary.pipeline import foreground_fit as F  # noqa: E402

#: The fit (module-level so a test can substitute a small run).
resolve = F.resolve


def _impulse_available() -> bool:
    try:
        import impulse  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:                      # spec section 1, guard 5
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(prog="python -P -m scripts.galactic_binary.fit_foreground",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--sangria-h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (outside the release tree)")
    ap.add_argument("--months", nargs="+", type=int, choices=[3, 6, 12], default=[3, 6, 12])
    ap.add_argument("--snr", type=float, nargs="+", default=[7.0])
    ap.add_argument("--backend", choices=["onfly", "fourier"], default="onfly")
    ap.add_argument("--compare-shipped", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    for name in ("out", "report"):
        p = getattr(a, name)
        if p is not None and p.resolve().is_relative_to(REL.resolve()):
            ap.error(f"--{name} must lie outside the release tree")
    if not a.sangria_h5.is_file():
        ap.error(f"--sangria-h5 {a.sangria_h5} is not a file")
    if not _impulse_available():
        ap.error("the GLASS MAP needs impulse (pip install the wdm-models[sampling] extra: impulse-mcmc v2 964283c)")
    if a.compare_shipped and a.snr != [7.0]:
        ap.error("--compare-shipped needs --snr 7 (the shipped threshold)")
    from wdm.backend import enable_x64

    enable_x64()                                     # before the first binary_galactica call
    data_root = resolve_data_root(a.data_root).resolve()
    rep: dict = {"backend": a.backend, "files": {}, "problems": []}
    try:
        for months in a.months:
            for snr in a.snr:
                path = F.write(resolve(a.sangria_h5, months, snr, backend=a.backend), a.out)
                entry = {"path": str(path)}
                if a.compare_shipped:
                    rel = f"galactic_binary/foreground/foreground_{months}mo_snr7.npz"
                    shipped = data_root / rel
                    import hashlib

                    if hashlib.sha256(shipped.read_bytes()).hexdigest() != manifest_sha256(data_root, rel):
                        raise RuntimeError(f"{shipped} does not match data/MANIFEST.json")
                    with np.load(path, allow_pickle=False) as new, np.load(shipped, allow_pickle=False) as old:
                        cmp = F.compare_fit(new, old)
                    entry["compare"] = cmp
                    gated = a.backend == SHIPPED_BACKEND
                    print(f"[fit_foreground] {months} mo: {'PASS' if cmp['ok'] else 'FAIL'} vs shipped "
                          f"({'gated' if gated else 'audit only: backend ' + a.backend}); "
                          + ", ".join(f"{k} {v:.2e}" for k, v in cmp["close"].items()))
                    if gated and not cmp["ok"]:
                        rep["problems"] += [f"{months} mo: {p}" for p in cmp["problems"]]
                rep["files"][f"{months}mo_snr{snr:g}"] = entry
    except Exception as exc:                          # fail closed, with the reason
        rep["problems"].append(f"{type(exc).__name__}: {exc}")
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for p in rep["problems"]:
        print(f"[fit_foreground]   {p}", file=sys.stderr)
    print(f"[fit_foreground] {'FAIL' if rep['problems'] else 'DONE'}: {len(rep['files'])} fit(s) in {a.out}")
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
