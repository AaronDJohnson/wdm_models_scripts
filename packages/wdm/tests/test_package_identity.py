"""C1 gates: which ``wdm`` is imported, its sentinels, and the frozen oracle."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent                                   # packages/wdm
SRC = PKG / "src"
ORACLE = TESTS / "oracle" / "vendored_2f7136c"
# The frozen tree the oracle was copied from (absent outside the workspace).
FROZEN = PKG.parents[2] / "toy_models" / "vendor" / "wdm" / "src" / "wdm"
#: sha256 of the frozen sources (toy_models/vendor/wdm/vendor-manifest.json).
FROZEN_SHA256 = {
    "filter.py": "17b74d1e6c812a1d44dc58cc87f42d23e48b14d5780d06673a5a892a55542792",
    "grid.py": "5d9f10a23af55fcba8e869944e541dfe178666d8a683463460e3f1b087fe2594",
    "noise.py": "b01390443b3b6d8b33c5a0bd72fe61a28ac1ec038b8d3aede2c53c0d257bf97d",
    "stationary.py": "e799278e9566484fa5bccb1d0d211bac3c0b103fd0fc5f7bb98ead1686282d7d",
    "transform.py": "852df5651eb30294e32acec8c0ed0bb2bb064df9183714ac67c87b1ff09e98f0",
}
#: sha256 of the oracle copies (relative imports rewritten to vendored_2f7136c.*).
ORACLE_SHA256 = {
    "filter.py": "ea52a6e394a5d8783f0dfe61bb038b706f737ed1beffee618083f883bfd7754b",
    "grid.py": "5d9f10a23af55fcba8e869944e541dfe178666d8a683463460e3f1b087fe2594",
    "noise.py": "5419255939429dc6e95277efc58eac9c1b96b06033005b548dd88c045501e368",
    "stationary.py": "bd7d2fbf60cb989ded1f7f5b4ff74f3edb30574aa6825bcdc5b9381b489bbd62",
    "transform.py": "b7d4922a5d7c47f7beef4e20e267fffc3a66d3ed2ce32bd69f4f258c6bcb9091",
}
_REWRITE = re.compile(r"^( *)from \.([a-z_]+) import", re.MULTILINE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(code: str, *paths: Path) -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(str(p) for p in (SRC, *paths)))
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(TESTS),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_wdm_resolves_to_this_package():
    import wdm

    assert Path(wdm.__file__).resolve().is_relative_to(SRC.resolve())


def test_release_sentinels():
    import wdm

    assert wdm.__version__ == "1.0.0"
    assert wdm.__wdm_api__ == "functions/1"
    assert wdm.__release__ == "wdm-paper-1"
    assert hasattr(wdm, "forward_time_many")


def test_import_wdm_does_not_import_jax():
    assert _run("import sys, wdm; print('jax' in sys.modules)") == "False"


def test_oracle_imports_as_its_own_package_without_wdm():
    out = _run(
        "import sys\n"
        "import vendored_2f7136c.transform, vendored_2f7136c.noise\n"
        "import vendored_2f7136c.stationary, vendored_2f7136c.filter, vendored_2f7136c.grid\n"
        "print('wdm' in sys.modules, vendored_2f7136c.transform.__file__)",
        ORACLE.parent,
    )
    loaded_wdm, path = out.split(" ", 1)
    assert loaded_wdm == "False"
    assert Path(path).resolve().is_relative_to(ORACLE.resolve())


@pytest.mark.parametrize("name", sorted(ORACLE_SHA256))
def test_oracle_bytes_are_pinned(name):
    assert _sha(ORACLE / name) == ORACLE_SHA256[name]


@pytest.mark.parametrize("name", sorted(ORACLE_SHA256))
def test_oracle_is_the_frozen_tree_with_imports_rewritten(name):
    src = FROZEN / name
    if not src.is_file():
        pytest.skip(f"frozen tree not present: {FROZEN}")
    assert _sha(src) == FROZEN_SHA256[name]
    rewritten = _REWRITE.sub(r"\1from vendored_2f7136c.\2 import", src.read_text())
    assert rewritten == (ORACLE / name).read_text()
