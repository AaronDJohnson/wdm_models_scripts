"""The test-only gbonfly adapter (spec section 7.6, S6): gbonfly store templates computed on
demand by lineage C (``workers/gbw_templates.py``) and streamed into the release likelihood,
so that a G3 comparison separates the likelihood numerics from the waveform swap
(binary_galactica ``fourier`` = gbonfly to ~2e-16 relative, not bitwise).

    tables, meta = streamed_templates([("gb_3mo", P8), ...])   # one lineage-C subprocess
    lik = models.build("gb_3mo", "tri_raw", template=tables["gb_3mo"])

Nothing is written under the release tree: the rows and templates live in
``$TMPDIR/golden_run``.  Needs the frozen trees (``paths.legacy_available()``).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

from scripts.golden.goldens_io import read_json, read_npz, write_json
from scripts.golden.lineage import run_worker
from scripts.golden.paths import DATA, ROOT, legacy_available, tmp_root

__all__ = ["TemplateTable", "streamed_templates"]


class TemplateTable:
    """``p8 -> (2, store_n)`` by exact lookup of the row's float64 bytes; a row that was not
    streamed raises ``KeyError`` (never a silent fallback to another waveform).  The table
    holds a private read-only copy of ``H``: a caller can neither change it afterwards nor
    corrupt a later lookup by writing into a returned template."""

    def __init__(self, P8: np.ndarray, H: np.ndarray) -> None:
        P8, H = np.atleast_2d(np.asarray(P8, dtype=np.float64)), np.array(H)
        if P8.shape[1] != 8 or H.ndim != 3 or H.shape[:2] != (len(P8), 2):
            raise ValueError(f"p8 {P8.shape} and templates {H.shape} do not pair up")
        H.setflags(write=False)
        self._table = {p.tobytes(): h for p, h in zip(P8, H, strict=True)}

    def __len__(self) -> int:
        return len(self._table)

    def __call__(self, p8) -> np.ndarray:
        key = np.asarray(p8, dtype=np.float64).tobytes()
        if key not in self._table:
            raise KeyError("no streamed gbonfly template at this source row")
        return self._table[key]


def streamed_templates(rows: Sequence[tuple[str, np.ndarray]], *, timeout: float = 1800.0
                       ) -> tuple[dict[str, TemplateTable], dict]:
    """``({case: TemplateTable}, worker meta)`` for ``rows = [(case, P8 (n, 8)), ...]`` (rows of
    one case are merged; duplicates are computed once)."""
    if not legacy_available():
        raise RuntimeError("the gbonfly adapter needs the frozen legacy trees (lineage C)")
    by_case: dict[str, list[np.ndarray]] = {}
    for case, P8 in rows:
        by_case.setdefault(case, []).append(np.atleast_2d(np.asarray(P8, dtype=np.float64)))
    if not by_case:
        raise ValueError("no source rows to stream")
    stage = tmp_root() / "jobs" / uuid.uuid4().hex[:12]
    stage.mkdir(parents=True)
    spec = []
    for case, parts in by_case.items():
        P8 = np.unique(np.concatenate(parts), axis=0)
        np.savez(stage / f"{case}.npz", p8=P8)
        spec.append({"tag": case, "case": case, "points": str(stage / f"{case}.npz")})
    write_json(stage / "jobs.json", spec)
    out = run_worker("C", "gb_legacy", "templates", "--jobs", str(stage / "jobs.json"), "--workspace", str(ROOT),
                     "--data-dir", str(DATA), timeout=timeout)
    meta = read_json(out / "templates.json")
    tables = {}
    for case in by_case:
        z = read_npz(out / f"{case}.npz")
        tables[case] = TemplateTable(z["p8"], z["h"])
    return tables, meta
