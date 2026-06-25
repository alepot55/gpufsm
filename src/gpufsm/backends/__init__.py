"""Backend registration. The CPU reference is always available; Triton and CUDA
register themselves only if their dependencies (and, for CUDA, the compiled
extension) are importable — keeping the core installable on CPU-only machines.
"""

from __future__ import annotations

import importlib

from . import cpu  # noqa: F401  (always available)

# Optional backends: import by name so a missing module/dependency is a no-op
# rather than a hard failure (and mypy doesn't require them to exist yet).
for _optional in (
    "gpufsm.backends.triton_backend",
    "gpufsm.backends.cuda_backend",
    "gpufsm.backends.warp_backend",
):
    try:
        importlib.import_module(_optional)
    except Exception:  # pragma: no cover - depends on environment
        pass
