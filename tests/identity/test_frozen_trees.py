"""The frozen legacy trees are never written (spec §0 rule 6, §7.1; Task A2)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import frozen_tree_hashes as fth

GOLDEN = Path(__file__).resolve().parents[2] / "golden"
LEGACY = fth.DEFAULT_ROOT
requires_legacy = pytest.mark.skipif(
    not all((LEGACY / t).is_dir() for t in fth.FROZEN_TREES),
    reason="frozen legacy trees not present (CI/Colab)")


def _make_tree(root: Path) -> None:
    (root / "a" / "sub").mkdir(parents=True)
    (root / "a" / "x.py").write_text("x = 1\n")
    (root / "a" / "sub" / "y.bin").write_bytes(bytes(10))
    (root / "a" / "__pycache__").mkdir()
    (root / "a" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"pyc")
    (root / "a" / ".pytest_cache").mkdir()
    (root / "a" / ".pytest_cache" / "README.md").write_text("cache")
    (root / "a" / ".DS_Store").write_bytes(b"finder")
    (root / "b").mkdir()
    (root / "b" / "z.txt").write_text("z")


@pytest.fixture
def tree(tmp_path):
    root, golden = tmp_path / "legacy", tmp_path / "golden"
    _make_tree(root)
    summary = fth.record(root, golden, trees=("a", "b"))
    return root, golden, summary


def test_record_counts_only_real_files(tree):
    _, golden, summary = tree
    assert summary["trees"]["a"]["n_files"] == 2
    assert summary["trees"]["b"]["n_files"] == 1
    assert summary["trees"]["a"]["n_bytes"] == len("x = 1\n") + 10
    assert summary["excluded"] == {"dirs": list(fth.EXCLUDED_DIRS), "files": list(fth.EXCLUDED_FILES)}
    assert (golden / fth.SUMMARY).is_file() and (golden / fth.LISTING).is_file()


def test_unchanged_tree_verifies_quick_and_full(tree):
    root, golden, _ = tree
    assert fth.verify(root, golden) == []
    assert fth.verify(root, golden, full=True) == []


def test_caches_and_finder_files_are_ignored(tree):
    root, golden, _ = tree
    (root / "a" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"rewritten")
    (root / "a" / ".DS_Store").write_bytes(b"rewritten by Finder")
    (root / "b" / "__pycache__").mkdir()
    (root / "b" / "__pycache__" / "new.pyc").write_bytes(b"new")
    assert fth.verify(root, golden) == []
    assert fth.verify(root, golden, full=True) == []


def test_same_size_content_change_is_caught_by_full(tree):
    root, golden, _ = tree
    p = root / "a" / "x.py"
    st = p.stat()
    p.write_text("x = 2\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert fth.verify(root, golden) == []
    assert fth.verify(root, golden, full=True) == ["content changed: a/x.py"]


def test_any_write_is_caught_by_quick(tree):
    root, golden, _ = tree
    p = root / "a" / "x.py"
    st = p.stat()
    p.write_text("x = 1\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert fth.verify(root, golden) == ["mtime changed: a/x.py"]


def test_new_and_missing_files_are_caught(tree):
    root, golden, _ = tree
    (root / "b" / "z.txt").unlink()
    (root / "a" / "new.py").write_text("")
    assert fth.verify(root, golden) == ["missing: b/z.txt", "new file: a/new.py"]


def test_symlinks_are_refused(tmp_path):
    root = tmp_path / "legacy"
    _make_tree(root)
    (root / "b" / "link").symlink_to(root / "a" / "x.py")
    with pytest.raises(ValueError, match="symlink"):
        fth.record(root, tmp_path / "golden", trees=("a", "b"))


def test_pycache_fingerprint_sees_bytecode_writes(tmp_path):
    root = tmp_path / "legacy"
    _make_tree(root)
    before = fth.pycache_fingerprint([root / "a", root / "b"])
    (root / "b" / "__pycache__").mkdir()
    (root / "b" / "__pycache__" / "z.cpython-312.pyc").write_bytes(b"pyc")
    after = fth.pycache_fingerprint([root / "a", root / "b"])
    assert set(after) - set(before) == {str(root / "b" / "__pycache__" / "z.cpython-312.pyc")}


def test_recorded_summary_covers_every_frozen_tree():
    summary = fth.read_summary(GOLDEN)
    assert set(summary["trees"]) == set(fth.FROZEN_TREES)
    assert all(info["n_files"] > 0 for info in summary["trees"].values())
    assert fth.sha256_file(GOLDEN / summary["listing"]) == summary["listing_sha256"]


@requires_legacy
def test_frozen_trees_unchanged_quick():
    assert fth.verify(LEGACY, GOLDEN) == []


@requires_legacy
@pytest.mark.slow
def test_frozen_trees_unchanged_full_sha256():
    assert fth.verify(LEGACY, GOLDEN, full=True) == []
