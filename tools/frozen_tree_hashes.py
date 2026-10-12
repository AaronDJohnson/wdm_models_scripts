"""sha256 of every file of the frozen legacy trees (spec §0 rule 6, §7.1).

    $PY -m tools.frozen_tree_hashes record          # once, before any legacy subprocess runs
    $PY -m tools.frozen_tree_hashes verify          # quick: path set + size + mtime_ns (seconds)
    $PY -m tools.frozen_tree_hashes verify --full   # re-hash every file (~100 GB, 1-3 min)

Excluded, because the OS, Finder or tools rewrite them and they are never inputs:
the directories __pycache__, .pytest_cache, .mypy_cache, .ruff_cache and
.ipynb_checkpoints, and the file name .DS_Store. The list is written into
golden/frozen_trees.json. A symlink anywhere in a tree is an error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("WDM_LEGACY_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
GOLDEN = Path(__file__).resolve().parents[1] / "golden"
FROZEN_TREES = ("toy_models", "wdm_paper_2/toy_models", "wdm", "wdm_paper_2/gbonfly",
                "wdm_paper_2/sinusoid_final", "wdm_paper_2/final_chains", "wdm_paper_1")
EXCLUDED_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints")
EXCLUDED_FILES = (".DS_Store",)
SUMMARY = "frozen_trees.json"
LISTING = "frozen_trees_files.tsv.gz"
_CHUNK = 1 << 22


@dataclass(frozen=True)
class Entry:
    tree: str
    rel: str
    size: int
    mtime_ns: int
    sha256: str          # "" when the scan did not hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(tree_root: Path) -> Iterator[Path]:
    """Regular files below ``tree_root`` in a deterministic order, exclusions removed."""
    if not tree_root.is_dir():
        raise FileNotFoundError(f"frozen tree missing: {tree_root}")
    for dirpath, dirnames, filenames in os.walk(tree_root, followlinks=False):
        base = Path(dirpath)
        for name in dirnames:
            if (base / name).is_symlink():
                raise ValueError(f"symlink in a frozen tree: {base / name}")
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            if name in EXCLUDED_FILES:
                continue
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink in a frozen tree: {path}")
            yield path


def scan(root: Path = DEFAULT_ROOT, trees: Iterable[str] = FROZEN_TREES, *, hash_files: bool) -> list[Entry]:
    out = []
    for tree in trees:
        tree_root = Path(root) / tree
        for path in iter_files(tree_root):
            rel = path.relative_to(tree_root).as_posix()
            if "\t" in rel or "\n" in rel:
                raise ValueError(f"unsupported character in {path}")
            st = path.stat()
            out.append(Entry(tree, rel, st.st_size, st.st_mtime_ns,
                             sha256_file(path) if hash_files else ""))
    return out


def tree_summary(entries: list[Entry], tree: str) -> dict:
    sel = sorted((e for e in entries if e.tree == tree), key=lambda e: e.rel)
    digest = hashlib.sha256()
    for e in sel:
        digest.update(f"{e.rel}\0{e.size}\0{e.sha256}\n".encode())
    return {"n_files": len(sel), "n_bytes": sum(e.size for e in sel),
            "max_mtime_ns": max((e.mtime_ns for e in sel), default=0),
            "tree_sha256": digest.hexdigest()}


def write_listing(entries: list[Entry], path: Path) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        for e in sorted(entries, key=lambda e: (e.tree, e.rel)):
            gz.write(f"{e.tree}\t{e.rel}\t{e.size}\t{e.mtime_ns}\t{e.sha256}\n".encode())
    Path(path).write_bytes(buf.getvalue())


def read_listing(path: Path) -> list[Entry]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    return [Entry(t, r, int(s), int(m), h) for t, r, s, m, h in rows]


def read_summary(golden: Path = GOLDEN) -> dict:
    return json.loads((Path(golden) / SUMMARY).read_text())


def record(root: Path = DEFAULT_ROOT, golden: Path = GOLDEN, trees: Iterable[str] = FROZEN_TREES) -> dict:
    trees = tuple(trees)
    entries = scan(root, trees, hash_files=True)
    golden = Path(golden)
    golden.mkdir(parents=True, exist_ok=True)
    write_listing(entries, golden / LISTING)
    summary = {"schema": 1, "root": str(Path(root).resolve()),
               "trees": {t: tree_summary(entries, t) for t in trees},
               "excluded": {"dirs": list(EXCLUDED_DIRS), "files": list(EXCLUDED_FILES)},
               "listing": LISTING, "listing_sha256": sha256_file(golden / LISTING)}
    (golden / SUMMARY).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def verify(root: Path = DEFAULT_ROOT, golden: Path = GOLDEN, *, full: bool = False) -> list[str]:
    """Differences between the trees now and the recorded listing (empty list = unchanged)."""
    summary = read_summary(golden)
    trees = tuple(summary["trees"])
    recorded = {(e.tree, e.rel): e for e in read_listing(Path(golden) / summary["listing"])}
    now = scan(root, trees, hash_files=full)
    current = {(e.tree, e.rel): e for e in now}
    problems = [f"missing: {t}/{r}" for t, r in sorted(recorded.keys() - current.keys())]
    problems += [f"new file: {t}/{r}" for t, r in sorted(current.keys() - recorded.keys())]
    for key in sorted(recorded.keys() & current.keys()):
        a, b = recorded[key], current[key]
        name = f"{key[0]}/{key[1]}"
        if a.size != b.size:
            problems.append(f"size changed: {name}")
        elif full and a.sha256 != b.sha256:
            problems.append(f"content changed: {name}")
        elif not full and a.mtime_ns != b.mtime_ns:
            problems.append(f"mtime changed: {name}")
    if full and not problems:
        for t in trees:
            if tree_summary(now, t)["tree_sha256"] != summary["trees"][t]["tree_sha256"]:
                problems.append(f"tree digest changed: {t}")
    return problems


def pycache_fingerprint(dirs: Iterable[Path | str]) -> dict[str, list[int]]:
    """Every file under any ``__pycache__`` below ``dirs`` -> [size, mtime_ns]."""
    out: dict[str, list[int]] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
            dirnames.sort()
            if Path(dirpath).name == "__pycache__":
                for name in sorted(filenames):
                    st = (Path(dirpath) / name).stat()
                    out[str(Path(dirpath) / name)] = [st.st_size, st.st_mtime_ns]
    return out


def main(argv: list[str] | None = None) -> int:
    if not sys.flags.safe_path:          # spec §1 guard 5: no cwd injection
        raise SystemExit("run with python -P or PYTHONSAFEPATH=1 (source env/activate.sh)")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--golden", type=Path, default=GOLDEN)
        if name == "verify":
            p.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        summary = record(a.root, a.golden)
        for tree, info in summary["trees"].items():
            print(f"{tree}: {info['n_files']} files, {info['n_bytes'] / 1e9:.2f} GB, {info['tree_sha256'][:16]}")
        return 0
    problems = verify(a.root, a.golden, full=a.full)
    print("\n".join(problems) if problems else f"FROZEN TREES OK ({'full sha256' if a.full else 'quick'})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
