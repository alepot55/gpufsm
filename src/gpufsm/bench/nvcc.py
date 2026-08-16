"""Compile a CUDA source to a shared library and load it through ctypes.

The witness experiments compare a Triton kernel against a hand-written CUDA one, so
each of them compiled a ``.cu`` at run time. Eight files carried the same block —
temp dir, nvcc flags, ``CDLL``, ``restype``/``argtypes`` — and the duplication was not
free: the ``-arch`` flag was hardcoded to ``sm_89`` for a long time, so on any other
GPU the cubin failed to load, the kernel silently never launched, the output buffer
stayed uninitialized and the oracle reported a 100% mismatch. The arch is derived from
the live device here, once.

Requires ``torch`` (to read the device capability) and a CUDA toolkit; import it only
from code that already needs a GPU.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "gpufsm" / "nvcc"


def arch_flag() -> str:
    """``-arch=sm_XY`` for the CUDA device actually present.

    Never hardcode this. A cubin built for the wrong architecture does not raise at
    load time in a way the caller notices — the kernel just does not run.
    """
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"-arch=sm_{major}{minor}"


def nvcc_path() -> str:
    """The nvcc to use: the toolkit's if installed in the usual place, else ``PATH``."""
    candidate = Path("/usr/local/cuda/bin/nvcc")
    return str(candidate) if candidate.exists() else "nvcc"


def compile_source(source: str, name: str, extra_flags: tuple[str, ...] = ()) -> Path:
    """Compile ``source`` to a ``.so`` and return its path, caching on content.

    The cache key is the hash of the source plus the flags plus the target arch, so a
    changed kernel or a different GPU produces a different library instead of silently
    reusing a stale one.
    """
    arch = arch_flag()
    key = hashlib.sha256("\0".join((source, arch, *extra_flags)).encode()).hexdigest()[:16]
    outdir = CACHE_ROOT / f"{name}-{key}"
    so = outdir / f"{name}.so"
    if so.exists():
        return so

    outdir.mkdir(parents=True, exist_ok=True)
    cu = outdir / f"{name}.cu"
    cu.write_text(source)
    result = subprocess.run(
        [
            nvcc_path(),
            "-O3",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            arch,
            *extra_flags,
            "-o",
            str(so),
            str(cu),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        so.unlink(missing_ok=True)
        raise RuntimeError(f"nvcc failed for {name} ({arch}):\n{result.stderr}")
    return so


def load_library(
    source: str,
    name: str,
    signatures: dict[str, tuple[type, list[type]]],
    extra_flags: tuple[str, ...] = (),
) -> ctypes.CDLL:
    """Compile ``source`` and return the loaded library with ``signatures`` applied.

    ``signatures`` maps an exported symbol to ``(restype, argtypes)``. Declaring them
    is not optional: without an explicit ``restype`` ctypes assumes ``int`` and a
    kernel returning a float elapsed time comes back as garbage.
    """
    lib = ctypes.CDLL(str(compile_source(source, name, extra_flags)))
    for symbol, (restype, argtypes) in signatures.items():
        fn = getattr(lib, symbol)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib
