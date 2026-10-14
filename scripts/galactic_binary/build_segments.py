"""Build the galactic-binary analysis segments (and, with ``--store``, their +-3-band stores)
from the Sangria HDF5 (spec sections 5.2 ``pipeline/segments.py`` and 6); never writes the
release tree.

From the release root, after ``source env/activate.sh``:

    $PY -P -m scripts.galactic_binary.build_segments --sangria-h5 PATH --out DIR \\
        [--months 3 6 12] [--backend onfly|fourier] [--store] [--segments-dir DIR] \\
        [--foreground-dir DIR] [--catalog-npz PATH] [--compare-shipped] [--report PATH]

**Approval-gated** (spec section 6: "need --sangria-h5; hours"): reads the private 3 GB
``LDC2_sangria_training_v2.h5`` (``sky/{dgb,igb,vgb}/tdi`` over the duration) and generates
every resolved binary that reaches the window twice (``BW`` and ``2 BW``: 32768 / 32768 / 131072).
Each duration's resolved binaries and foreground come from
``<foreground-dir>/foreground_<m>mo_snr7.npz`` (default: the shipped fits, sha256-checked).
``--store`` builds the stores around the segments in ``--segments-dir`` (default: the shipped
segments).  ``--catalog-npz`` (``sangria_minimal.npz``) adds the legacy raw-convention check.

The release default waveform backend is ``onfly``; ``--backend fourier`` (gbonfly's algorithm)
reproduces the shipped files, and ``--compare-shipped`` then gates each rebuilt file against
``data/galactic_binary/segments`` (``segments.compare_npz``: the instrument draws, the bin
geometry and the selection ``array_equal``; data, residual, injection and SNRs within 1e-12 of
their largest magnitude).  With ``--backend onfly`` the comparison is reported, not gated.

Expected with ``--backend fourier --compare-shipped`` (the shipped provenance; the run itself is
the user's): 234 / 262 / 2015 resolved binaries subtracted at 3 / 6 / 12 months (1148 / 759 /
7880 reach the window at 2 BW), injection SNR 9.7306 / 12.2779 / 17.9340, bandwidth checks
0.0253 / 0.0294 / 0.0441 (subtraction) and 0.0021 / 0.0057 / 6.7e-6 (injection); the stores
reach 245 / 276 / 2113 resolved binaries.  Every piece that needs no Sangria file is already
pinned to these files by ``packages/wdm_models/tests/test_wm_gb_segments.py``.

Exit status 0: built (and, with ``--compare-shipped --backend fourier``, inside the gate).
1: a build raised or a gated comparison failed.  2: usage.
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
from wdm_models.galactic_binary.pipeline import segments as S  # noqa: E402


def _checked(data_root: Path, rel: str) -> Path:
    import hashlib

    path = data_root / rel
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest_sha256(data_root, rel):
        raise RuntimeError(f"{path} does not match data/MANIFEST.json")
    return path


def _input(directory: Path | None, data_root: Path, sub: str, name: str) -> Path:
    """``directory/name``, or the shipped ``data/galactic_binary/<sub>/name`` (sha256-checked)."""
    if directory is not None:
        return directory / name
    return _checked(data_root, f"galactic_binary/{sub}/{name}")


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:                      # spec section 1, guard 5
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(prog="python -P -m scripts.galactic_binary.build_segments",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--sangria-h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (outside the release tree)")
    ap.add_argument("--months", nargs="+", type=int, choices=[3, 6, 12], default=[3, 6, 12])
    ap.add_argument("--backend", choices=["onfly", "fourier"], default="onfly")
    ap.add_argument("--store", action="store_true", help="build the stores around --segments-dir")
    ap.add_argument("--segments-dir", type=Path, default=None)
    ap.add_argument("--foreground-dir", type=Path, default=None)
    ap.add_argument("--catalog-npz", type=Path, default=None)
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
    from wdm.backend import enable_x64

    enable_x64()                                     # before the first binary_galactica call
    data_root = resolve_data_root(a.data_root).resolve()
    rep: dict = {"backend": a.backend, "store": a.store, "files": {}, "problems": []}
    try:
        verification = (None if a.catalog_npz is None
                        else S.verify_raw_convention(a.sangria_h5, a.catalog_npz))
        for months in a.months:
            spec = S.SegmentSpec.paper(months)
            fg = S.load_foreground(_input(a.foreground_dir, data_root, "foreground",
                                          f"foreground_{months}mo_snr7.npz"), spec.T)
            if a.store:
                seg = _input(a.segments_dir, data_root, "segments", f"segment_snr7_{months}mo_3bands.npz")
                path = S.build_store(a.sangria_h5, spec, seg, a.out, foreground=fg, backend=a.backend)
                shipped_rel = f"galactic_binary/segments/store_snr7_{months}mo_3bands_pm3.npz"
            else:
                path = S.build_segment(a.sangria_h5, spec, a.out, foreground=fg, backend=a.backend,
                                       verification=verification)
                shipped_rel = f"galactic_binary/segments/segment_snr7_{months}mo_3bands.npz"
            entry = {"path": str(path)}
            if a.compare_shipped:
                with np.load(path, allow_pickle=False) as new, \
                        np.load(_checked(data_root, shipped_rel), allow_pickle=False) as old:
                    cmp = S.compare_npz(new, old)
                entry["compare"] = cmp
                gated = a.backend == SHIPPED_BACKEND
                print(f"[build_segments] {months} mo: {'PASS' if cmp['ok'] else 'FAIL'} vs shipped "
                      f"({'gated' if gated else 'audit only: backend ' + a.backend}); "
                      + ", ".join(f"{k} {v:.2e}" for k, v in cmp["close"].items()))
                if gated and not cmp["ok"]:
                    rep["problems"] += [f"{months} mo: {p}" for p in cmp["problems"]]
            rep["files"][str(months)] = entry
    except Exception as exc:                          # fail closed, with the reason
        rep["problems"].append(f"{type(exc).__name__}: {exc}")
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for p in rep["problems"]:
        print(f"[build_segments]   {p}", file=sys.stderr)
    print(f"[build_segments] {'FAIL' if rep['problems'] else 'DONE'}: {len(rep['files'])} file(s) in {a.out}")
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
