"""Fail-closed file artifacts (spec section 3.5).

An :class:`Artifact` is a path plus what the file must be: an **identity** (a
JSON-serialisable dict that the file records under ``metadata["identity"]`` and,
for keyed artifacts, that names the file) and/or the **sha256** of its bytes (from
``data/MANIFEST.json``).  Loading verifies both and raises on any mismatch or a
missing file; it never computes a replacement.  Nothing is written unless the caller
passes ``allow_write=True`` (unlike WT ``FastCovariance._load_or_compute_cN``, which
saved a missing ``cN_*.npy`` as a side effect).  Writes are atomic (temporary file in
the same directory, then ``os.replace``, as WT ``bandcov._atomic_npz``).

The model layer owns the concrete artifacts (``c_N``, the bandcov interpolant, the GB
``basis_L``); this module only knows files.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Artifact", "ArtifactError", "ArtifactMissing", "identity_key"]


class ArtifactError(ValueError):
    """The file exists but is not the artifact asked for (sha256, identity or format)."""


class ArtifactMissing(FileNotFoundError):
    """The artifact's file does not exist; it is never created implicitly."""


# What ``np.load`` (allow_pickle=False) raises on a truncated, empty or foreign file.
_READ_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile)


def identity_key(identity: Mapping[str, Any], length: int = 20) -> str:
    """``sha256(json.dumps(identity, sort_keys=True))[:length]`` -- the WT bandcov cache key
    (``BandCovariance.cache_path``, src/wdm_toys/bandcov.py:144-146)."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:length]


def _canonical(identity: Mapping[str, Any]) -> Any:
    return json.loads(json.dumps(identity, sort_keys=True))


@dataclass(frozen=True)
class Artifact:
    """A file that must exist with the given ``identity`` and/or ``sha256``."""

    path: Path
    identity: Mapping[str, Any] | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.identity is not None:
            object.__setattr__(self, "identity", _canonical(self.identity))
        if self.sha256 is not None and (not isinstance(self.sha256, str) or len(self.sha256) != 64
                                        or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError(f"sha256 must be 64 lowercase hex digits; got {self.sha256!r}")

    @classmethod
    def keyed(cls, root: str | Path, prefix: str, identity: Mapping[str, Any], suffix: str = ".npz",
              *, sha256: str | None = None, key_length: int = 20) -> Artifact:
        """``root / f"{prefix}{identity_key(identity)}{suffix}"`` (e.g. ``bandcov_v1_<key>.npz``)."""
        return cls(Path(root) / f"{prefix}{identity_key(identity, key_length)}{suffix}", identity,
                   sha256)

    # -- reading ---------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.is_file()

    def file_sha256(self) -> str:
        h = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify(self) -> None:
        """Raise :class:`ArtifactMissing` or :class:`ArtifactError` unless the file is there
        with the pinned sha256 (when one is given)."""
        if not self.exists():
            raise ArtifactMissing(f"artifact missing: {self.path} (it is never created implicitly; "
                                  "run the script that prepares it)")
        if self.sha256 is not None:
            got = self.file_sha256()
            if got != self.sha256:
                raise ArtifactError(f"{self.path}: sha256 {got} != pinned {self.sha256}")

    def load_npz(self) -> dict[str, np.ndarray]:
        """Every array of the ``.npz`` (``allow_pickle=False``), after :meth:`verify` and,
        when an identity is given, ``metadata["identity"] == identity``."""
        self.verify()
        try:                                  # our handle: closed even when np.load fails
            with self.path.open("rb") as fh:
                z = np.load(fh, allow_pickle=False)
                if isinstance(z, np.lib.npyio.NpzFile):
                    with z:
                        arrays = {k: np.array(z[k]) for k in z.files}
                else:
                    arrays = None
        except _READ_ERRORS as err:
            raise ArtifactError(f"{self.path}: not a readable .npz") from err
        if arrays is None:
            raise ArtifactError(f"{self.path}: not a .npz archive")
        if self.identity is not None:
            if "metadata" not in arrays:
                raise ArtifactError(f"{self.path}: records no metadata/identity")
            try:
                recorded = json.loads(str(arrays["metadata"])).get("identity")
            except (json.JSONDecodeError, AttributeError) as err:
                raise ArtifactError(f"{self.path}: unreadable metadata") from err
            if recorded != self.identity:
                raise ArtifactError(f"{self.path}: identity mismatch")
        return arrays

    def load_npy(self) -> np.ndarray:
        """The ``.npy`` array (``allow_pickle=False``) after :meth:`verify`.  An ``.npy``
        cannot record an identity: pin it by ``sha256`` (and its name)."""
        if self.identity is not None:
            raise ValueError("an .npy artifact cannot carry an identity; pin it by sha256")
        self.verify()
        try:
            with self.path.open("rb") as fh:
                array = np.load(fh, allow_pickle=False)
                if not isinstance(array, np.ndarray):                 # an .npz archive
                    array.close()
                    array = None
        except _READ_ERRORS as err:
            raise ArtifactError(f"{self.path}: not a readable .npy") from err
        if array is None:
            raise ArtifactError(f"{self.path}: not a .npy array")
        return array

    # -- writing (explicit only) -------------------------------------------------
    def _target(self, allow_write: bool, overwrite: bool) -> None:
        if not allow_write:
            raise PermissionError(f"refusing to write {self.path}: pass allow_write=True "
                                  "(artifacts are prepared by scripts, never as a side effect)")
        if self.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _storable(array: Any) -> np.ndarray:
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError("object arrays cannot be stored (they load only with allow_pickle)")
        return array

    def _atomic(self, write: Any) -> str:
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("wb") as fh:
                write(fh)
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return self.file_sha256()

    def save_npz(self, arrays: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
                 allow_write: bool = False, overwrite: bool = False) -> str:
        """Write ``arrays`` (+ ``metadata`` as a JSON string with this artifact's identity);
        returns the new file's sha256.  Everything is validated before anything is written."""
        payload = {k: self._storable(v) for k, v in arrays.items()}
        if "metadata" in payload:
            raise ValueError("pass metadata through the metadata= argument")
        if metadata is not None or self.identity is not None:
            meta = dict(metadata or {})
            if self.identity is not None:
                if "identity" in meta and _canonical(meta["identity"]) != self.identity:
                    raise ValueError("metadata['identity'] differs from the artifact's identity")
                meta["identity"] = self.identity
            payload["metadata"] = np.asarray(json.dumps(meta, sort_keys=True))
        self._target(allow_write, overwrite)
        return self._atomic(lambda fh: np.savez(fh, **payload))

    def save_npy(self, array: Any, *, allow_write: bool = False, overwrite: bool = False) -> str:
        """Write one array as ``.npy``; returns the new file's sha256."""
        if self.identity is not None:
            raise ValueError("an .npy artifact cannot carry an identity; pin it by sha256")
        array = self._storable(array)
        self._target(allow_write, overwrite)
        return self._atomic(lambda fh: np.save(fh, array, allow_pickle=False))
