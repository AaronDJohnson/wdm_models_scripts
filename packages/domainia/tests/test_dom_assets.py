"""D-A9: fail-closed artifacts (spec section 3.5) on the shipped sinusoid caches."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from domainia.assets import Artifact, ArtifactError, ArtifactMissing, identity_key

ROOT = Path(__file__).resolve().parents[3]                            # wdm_models_scripts
CACHE = ROOT / "data" / "sinusoid" / "cache"
MANIFEST = ROOT / "data" / "MANIFEST.json"
BANDCOV = "sinusoid/cache/bandcov_v1_4bfb466c191c79f9879a.npz"
CN = "sinusoid/cache/cN_N16384_dt0.0009765625_fmin4_fref1.npy"


def bandcov_identity() -> dict:
    """WT ``BandCovariance._setup`` identity for the manuscript configuration (spec 5.1)."""
    f_min, f_nyq = 4.0, 512.0
    return dict(version=1, N=16384, dt=2.0**-10, f_min=f_min, f_ref=1.0, analysis_fmin=4.0,
                n_keep=16256, n_drop=128, support="real-orthonormal;f>=fmin;f<Nyquist;noDC", P=24,
                bounds=np.asarray([[-4.0, 0.0], [-2.0, 2.0]]).tolist(), drift_transform="asinh",
                drift_scale=min(.03, .384 * f_min / f_nyq),
                algorithm="latent-C0-PCG-discarded-precision-chebyshev")


@pytest.fixture(scope="module")
def shipped():
    if not (MANIFEST.is_file() and (ROOT / "data" / BANDCOV).is_file()):
        pytest.skip("shipped sinusoid caches not present")
    return json.loads(MANIFEST.read_text())["files"]


def test_identity_key_reproduces_the_shipped_bandcov_name():
    assert identity_key(bandcov_identity()) == "4bfb466c191c79f9879a"


def test_loads_the_shipped_bandcov_and_cN(shipped):
    art = Artifact.keyed(CACHE, "bandcov_v1_", bandcov_identity(), sha256=shipped[BANDCOV])
    assert art.path.name == "bandcov_v1_4bfb466c191c79f9879a.npz"
    arrays = art.load_npz()
    assert arrays["coefficients"].shape == (41, 65)
    assert json.loads(str(arrays["metadata"]))["c_N"] == 0.22629082684418084
    cN = Artifact(ROOT / "data" / CN, sha256=shipped[CN]).load_npy()
    assert float(cN) == 0.22629082684418084


def test_a_missing_artifact_is_never_created(tmp_path):
    ident = dict(bandcov_identity(), P=23)
    art = Artifact.keyed(tmp_path, "bandcov_v1_", ident)
    with pytest.raises(ArtifactMissing, match="never created implicitly"):
        art.load_npz()
    assert list(tmp_path.iterdir()) == []


def test_sha256_and_identity_mismatches_fail_closed(shipped, tmp_path):
    src = ROOT / "data" / BANDCOV
    tampered = tmp_path / src.name
    tampered.write_bytes(src.read_bytes()[:-1] + bytes([src.read_bytes()[-1] ^ 1]))
    with pytest.raises(ArtifactError, match="sha256"):
        Artifact(tampered, bandcov_identity(), shipped[BANDCOV]).load_npz()
    other = dict(bandcov_identity(), P=23)
    renamed = Artifact.keyed(tmp_path, "bandcov_v1_", other)
    shutil.copyfile(src, renamed.path)                                  # right bytes, wrong name
    with pytest.raises(ArtifactError, match="identity mismatch"):
        renamed.load_npz()
    with pytest.raises(ValueError, match="cannot carry an identity"):
        Artifact(ROOT / "data" / CN, identity={"N": 16384}).load_npy()
    # Unreadable or wrong-format files (no sha256 pinned) are ArtifactErrors too.
    truncated, empty, npy_named_npz = (tmp_path / n for n in ("t.npz", "e.npz", "cN.npz"))
    truncated.write_bytes(src.read_bytes()[:40])
    empty.write_bytes(b"")
    shutil.copyfile(ROOT / "data" / CN, npy_named_npz)
    for bad in (truncated, empty, npy_named_npz):
        with pytest.raises(ArtifactError, match="npz"):
            Artifact(bad).load_npz()
    with pytest.raises(ArtifactError, match="not a .npy"):
        Artifact(src).load_npy()


def test_writes_need_allow_write_and_are_atomic(tmp_path):
    ident = {"version": 1, "N": 8}
    art = Artifact.keyed(tmp_path / "cache", "toy_v1_", ident)
    with pytest.raises(PermissionError, match="allow_write=True"):
        art.save_npz({"a": np.arange(3.0)})
    assert not (tmp_path / "cache").exists()
    sha = art.save_npz({"a": np.arange(3.0)}, metadata={"note": "test"}, allow_write=True)
    assert sha == art.file_sha256() and sorted(p.name for p in art.path.parent.iterdir()) == [
        art.path.name]
    back = Artifact(art.path, ident, sha).load_npz()
    assert np.array_equal(back["a"], np.arange(3.0))
    assert json.loads(str(back["metadata"])) == {"identity": ident, "note": "test"}
    with pytest.raises(FileExistsError):
        art.save_npz({"a": np.zeros(1)}, allow_write=True)
    refused = tmp_path / "refused"                                      # invalid saves create nothing
    with pytest.raises(ValueError, match="metadata= argument"):
        Artifact(refused / "a.npz").save_npz({"metadata": np.zeros(1)}, allow_write=True)
    with pytest.raises(ValueError, match="differs from the artifact's identity"):
        Artifact(refused / "b.npz", ident).save_npz(
            {"a": np.zeros(1)}, metadata={"identity": {"N": 9}}, allow_write=True)
    with pytest.raises(ValueError, match="object arrays"):
        Artifact(refused / "c.npz").save_npz({"a": np.array([{}], dtype=object)}, allow_write=True)
    with pytest.raises(ValueError, match="object arrays"):
        Artifact(refused / "d.npy").save_npy(np.array([{}], dtype=object), allow_write=True)
    assert not refused.exists()
    npy = Artifact(tmp_path / "c.npy")
    with pytest.raises(PermissionError):
        npy.save_npy(np.float64(0.5))
    npy.save_npy(np.float64(0.5), allow_write=True)
    assert float(npy.load_npy()) == 0.5


def test_sha256_must_be_hex():
    with pytest.raises(ValueError, match="64 lowercase hex"):
        Artifact(Path("x.npz"), sha256="ABC")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        Artifact(Path("x.npz"), sha256=b"a" * 64)
