"""Shared helper: the nvcc -arch flag for the GPU actually present.

The landmark witnesses compile their CUDA thread-kernel with nvcc into a .so.
Hardcoding -arch=sm_89 (RTX 4070) makes the cubin fail to load on any other
architecture (e.g. A100 = sm_80, H100 = sm_90): the kernel silently does not
launch and the output stays uninitialized, so the CPU oracle sees a 100%
mismatch. Deriving the arch from the live device makes every witness portable
across GPUs (and still yields sm_89 on the 4070, so no change there).
"""

from __future__ import annotations


def cuda_arch_flag() -> str:
    """Return e.g. '-arch=sm_80' for the current CUDA device."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"-arch=sm_{major}{minor}"
