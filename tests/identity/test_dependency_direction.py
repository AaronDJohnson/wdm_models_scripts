"""Spec section 1: the dependency direction ``wdm <- domainia <- wdm_models <- scripts``.

Each package is imported -- with every submodule -- in a fresh ``python -P`` subprocess
whose meta path refuses the downstream packages, and the sources are scanned (AST) for
upward imports.  A negative control plants an upward import in a scratch package and
checks that both guards catch it.  ``wdm_models`` arrives in S3/S5; this file then gains
its own probe.  (domainia's import identity, sentinel refusal, metadata and the root
``testpaths`` entry are D-A1's tests.)
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REL = Path(__file__).resolve().parents[2]
SRC = {"wdm": REL / "packages" / "wdm" / "src", "domainia": REL / "packages" / "domainia" / "src"}
#: What each package must never import (its downstream packages and the release harness).
DOWNSTREAM = {"wdm": ("domainia", "wdm_models", "scripts", "tools"),
              "domainia": ("wdm_models", "scripts", "tools")}

_PROBE = '''
import importlib, importlib.abc, json, pkgutil, sys

class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {hidden!r}:
            raise ModuleNotFoundError(f"{{name}} hidden by the dependency-direction test", name=name)
        return None

sys.meta_path.insert(0, _Hide())
pkg = importlib.import_module({package!r})
mods = sorted(m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."))
for name in mods:
    importlib.import_module(name)
print(json.dumps({{"file": pkg.__file__, "modules": mods,
                  "loaded": sorted(k.split(".")[0] for k in sys.modules if "." not in k)}}))
'''


def _run_probe(package: str, hidden: tuple[str, ...], pythonpath: str,
               cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=pythonpath)
    return subprocess.run([sys.executable, "-P", "-c", _PROBE.format(hidden=hidden, package=package)],
                          env=env, cwd=str(cwd), capture_output=True, text=True)


def _import_all(package: str) -> dict:
    out = _run_probe(package, DOWNSTREAM[package],
                     os.pathsep.join(str(SRC[p]) for p in ("domainia", "wdm")), REL.parent)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("package", ["wdm", "domainia"])
def test_every_module_imports_with_its_downstream_hidden(package):
    info = _import_all(package)
    assert Path(info["file"]).resolve().is_relative_to(SRC[package].resolve())
    assert info["modules"], "walk_packages found no submodules"
    assert not set(info["loaded"]) & set(DOWNSTREAM[package])
    assert "jax" not in info["loaded"]               # the NumPy-only lane imports every module


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("package", ["wdm", "domainia"])
def test_no_source_file_imports_downstream(package):
    files = sorted((SRC[package] / package).rglob("*.py"))
    assert files
    bad = {str(f.relative_to(SRC[package])): sorted(_imports(f) & set(DOWNSTREAM[package]))
           for f in files if _imports(f) & set(DOWNSTREAM[package])}
    assert not bad, bad


def test_the_guards_catch_an_upward_import(tmp_path):
    """Negative control: a scratch package whose submodules import downstream packages
    (one inside a function body, which only the AST scan sees) fails both guards."""
    pkg = tmp_path / "src" / "upward"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "bad.py").write_text("from wdm_models import registry  # noqa: F401\n")
    (pkg / "lazy.py").write_text("def f():\n    import scripts.golden  # noqa: F401\n")
    assert _imports(pkg / "bad.py") == {"wdm_models"}
    assert _imports(pkg / "lazy.py") == {"scripts"}
    out = _run_probe("upward", DOWNSTREAM["domainia"], str(tmp_path / "src"), tmp_path)
    assert out.returncode != 0
    assert "wdm_models hidden by the dependency-direction test" in out.stderr
