"""Versioned CSV output — the only way measurements reach ``paper/data``.

Paper figures are regenerated from committed CSVs, so the CSV *is* the record. Two
rules are enforced here rather than left to each script: the schema is explicit (a
row with an unexpected key is an error, not a silently dropped column), and the
environment that produced the numbers is captured alongside them.
"""

from __future__ import annotations

import csv
import platform
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> Path:
    """Write ``rows`` to ``path`` with exactly ``fields`` as the header.

    Raises on a row carrying a key outside ``fields``: a typo in a column name would
    otherwise drop that measurement from the CSV without a word.
    """
    rows = list(rows)
    allowed = set(fields)
    for i, row in enumerate(rows):
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"row {i} has fields not in the schema: {sorted(unknown)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def environment() -> dict[str, str]:
    """What produced the numbers: GPU, driver stack and interpreter.

    A throughput without the device it was measured on is not reproducible, and the
    committed CSVs are compared across machines (RTX 4070 vs A100) all the time.
    """
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": "(none)",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "(unknown)"
    except Exception:  # pragma: no cover - depends on environment
        pass
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover - depends on environment
        pass
    return info


def gpu_slug() -> str:
    """The GPU name as a filename fragment, e.g. ``nvidia_geforce_rtx_4070``."""
    name = environment()["gpu"]
    return name.lower().replace(" ", "_") if name != "(none)" else "nocuda"


def print_environment(stream: Any = sys.stdout) -> None:
    """Print the environment block measurement scripts put at the top of their output."""
    for key, value in environment().items():
        print(f"{key:9s}: {value}", file=stream)
