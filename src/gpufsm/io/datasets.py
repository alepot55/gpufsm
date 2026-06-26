"""Robust, checksummed dataset acquisition.

Replaces the legacy fragile SharePoint download with verifiable fetches: every
dataset declares a SHA-256, downloads are checksum-verified, and a cached copy is
reused. Small fixtures are vendored in the repo; the large ANMLZoo/AutomataZoo
suite is fetched on demand.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    sha256: str


# Known datasets — SHA-256 pinned from the public ANMLZoo repo (a trusted academic
# mirror, jackwadden/ANMLZoo). Each is a homogeneous ANML automaton loadable via
# gpufsm.io.anml. Add more entries (with their pinned SHA) as needed.
_ANMLZOO = "https://raw.githubusercontent.com/jackwadden/ANMLZoo/master"
DATASETS: dict[str, Dataset] = {
    # Levenshtein edit-distance automaton (k=24, 20x3): 2784 STEs, pure homogeneous,
    # all-input start states. Smallest ANMLZoo .anml; validated GPU==reference on it.
    "levenshtein": Dataset(
        name="levenshtein_24_20x3.1chip.anml",
        url=f"{_ANMLZOO}/Levenshtein/anml/24_20x3.1chip.anml",
        sha256="8d6ec59d7c57a6e41112f90c244b5c393ff71124df8062ab025c8f243f6a7370",
    ),
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (constant memory)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: str | Path, expected_sha256: str) -> bool:
    """True iff ``path`` exists and its SHA-256 matches ``expected_sha256``."""
    p = Path(path)
    return p.is_file() and sha256_file(p) == expected_sha256


def ensure(dataset: Dataset, dest_dir: str | Path) -> Path:
    """Return a checksum-verified local copy of ``dataset``, downloading if needed."""
    if not dataset.sha256:
        raise ValueError(
            f"dataset {dataset.name!r} has no SHA-256 pinned; refusing to download unverified data"
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dataset.name

    if verify(dest, dataset.sha256):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(dataset.url, tmp)  # noqa: S310 - pinned, checksum-verified below
    if sha256_file(tmp) != dataset.sha256:
        tmp.unlink(missing_ok=True)
        raise OSError(f"checksum mismatch for {dataset.name!r} downloaded from {dataset.url}")
    tmp.replace(dest)
    return dest
