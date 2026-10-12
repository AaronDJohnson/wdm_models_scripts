"""Package-wide test configuration for ``wdm-paper``.

Puts ``tests/oracle`` on ``sys.path`` so the frozen oracle imports as the
top-level package ``vendored_2f7136c`` (never as ``wdm``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))
