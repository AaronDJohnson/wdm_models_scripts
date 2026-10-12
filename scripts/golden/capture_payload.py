"""Run one section-B golden worker in its lineage subprocess and copy its payload into golden/.

    $PY -P scripts/golden/capture_payload.py [--timeout S] [--replace] LINEAGE WORKER -- [worker args]

The worker writes ``<out_dir>/payload/<relpath>`` and ``<out_dir>/payload_index.json``
(``gbw_common.Payload``). Every listed file is sha-checked, copied to ``golden/<relpath>``
and merged into ``golden/MANIFEST.json`` (``files`` + ``provenance``). An existing entry
with different bytes is refused before anything is copied, unless ``--replace``.

The worker runs inside Section A's frozen-tree guard (A10 ``capture.py``): the quick
``frozen_tree_hashes.verify`` against ``golden/frozen_trees.json`` must pass first, and the
trees' stat listing plus every lineage ``__pycache__`` (``capture.guard_state``) must be
unchanged afterwards. The worker's stdout (``out_dir/stdout.log``, A3) is echoed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if not sys.flags.safe_path:
    raise SystemExit("run with python -P (PYTHONSAFEPATH)")
HERE = Path(__file__).resolve().parent
REL = HERE.parents[1]
for _p in (str(HERE), str(REL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import manifest_merge  # noqa: E402
from scripts.golden import lineage  # noqa: E402  (Section A: LINEAGES, run_worker)
from scripts.golden.capture import FrozenTreeChanged, guard_state  # noqa: E402  (Section A, A10)
from scripts.golden.paths import GOLDEN, ROOT  # noqa: E402
from tools import frozen_tree_hashes as fth  # noqa: E402  (Section A, A2)


def run_guarded(lineage_name: str, worker: str, worker_args, *, timeout: float,
                golden: Path = GOLDEN) -> Path:
    """``lineage.run_worker`` between A10's frozen-tree checks; echoes the worker's stdout."""
    problems = fth.verify(ROOT, golden, full=False)
    if problems:
        raise FrozenTreeChanged("frozen trees differ from golden/frozen_trees.json:\n"
                                + "\n".join(problems[:20]))
    before = guard_state()
    out = Path(lineage.run_worker(lineage_name, worker, *worker_args, timeout=timeout))
    sys.stdout.write((out / "stdout.log").read_text())
    sys.stdout.flush()
    if guard_state() != before:
        raise FrozenTreeChanged(f"a frozen tree or a lineage __pycache__ changed while {worker} ran")
    return out


def capture(lineage_name: str, worker: str, worker_args, *, timeout: float = 1800.0,
            golden: Path = GOLDEN, replace: bool = False) -> dict[str, str]:
    out = run_guarded(lineage_name, worker, worker_args, timeout=timeout, golden=golden)
    index = json.loads((out / "payload_index.json").read_text())
    if (index.get("schema"), index.get("lineage"), index.get("worker")) != (1, lineage_name, worker):
        raise RuntimeError(f"{out}: payload index does not belong to {lineage_name}/{worker}")
    golden = golden.resolve()
    files = dict(index["files"])
    for rel, sha in files.items():
        if manifest_merge.sha256_file(out / "payload" / rel) != sha:
            raise RuntimeError(f"{rel}: payload bytes do not match the index")
        if golden not in (golden / rel).resolve().parents:
            raise RuntimeError(f"{rel}: escapes golden/")
    bad = manifest_merge.conflicts(manifest_merge.load(golden / "MANIFEST.json"), files)
    if bad and not replace:
        raise SystemExit(f"golden/MANIFEST.json holds other bytes for {bad[:5]}; rerun with --replace "
                         "only if the change is intended")
    provenance = {}
    for rel, sha in files.items():
        dst = golden / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "payload" / rel, dst)
        if manifest_merge.sha256_file(dst) != sha:
            raise RuntimeError(f"{rel}: copy differs from the payload")
        provenance[rel] = {k: index.get(k) for k in ("lineage", "worker", "subcommand", "args",
                                                      "inputs", "build")}
    manifest_merge.merge_entries(golden / "MANIFEST.json", files, provenance=provenance,
                                 replace=replace)
    return files


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    worker_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, worker_args = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lineage")
    ap.add_argument("worker")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    files = capture(a.lineage, a.worker, worker_args, timeout=a.timeout, replace=a.replace)
    print(f"captured {len(files)} file(s) from {a.lineage}/{a.worker} into {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
