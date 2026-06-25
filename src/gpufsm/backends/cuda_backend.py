"""CUDA backend — registers only if the compiled ``_cuda`` extension is present.

The heavy lifting lives in ``backends/cuda/nfa_kernel.cu`` (built via
``GPUFSM_BUILD_CUDA=ON``). This module adapts the gpufsm NFA to that extension's
``run_dense`` entry point and exposes it through the registry.

Status: structurally complete; requires a CUDA toolkit + GPU to build/validate.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from ..nfa import NFA
from ..registry import Backend, register, register_availability
from ..result import Result

_EXT_NAME = "gpufsm.backends.cuda._cuda"


def _load_ext() -> Any | None:
    try:
        return importlib.import_module(_EXT_NAME)
    except Exception:
        return None


def _cuda_available() -> bool:
    return _load_ext() is not None


_cuda: Any = _load_ext()

if _cuda is not None:  # pragma: no cover - requires compiled extension + GPU

    class CUDAExecutor:
        def __init__(self, nfa: NFA, technique: str = "dense") -> None:
            self.nfa = nfa
            self.technique = technique

        def run(self, input_bytes: bytes) -> Result:
            nfa = self.nfa
            t0 = time.perf_counter()
            syms = np.frombuffer(input_bytes, dtype=np.uint8).astype(np.int32)
            transfer_ms = (time.perf_counter() - t0) * 1000.0
            accepted, match_len, kernel_ms = _cuda.run_dense(
                np.ascontiguousarray(nfa.sym_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_targets, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_symbols, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_targets, dtype=np.int32),
                np.ascontiguousarray(nfa.accept, dtype=np.int8),
                np.ascontiguousarray(syms, dtype=np.int32),
                int(nfa.num_states),
                int(nfa.start_state),
                int(nfa.uses_any_symbol),
            )
            return Result(
                accepted=bool(accepted),
                match_len=int(match_len),
                kernel_ms=float(kernel_ms),
                total_ms=float(kernel_ms) + transfer_ms,
                transfer_ms=transfer_ms,
            )

    @register(Backend.CUDA, "dense")
    def _make_cuda(nfa: NFA, technique: str) -> CUDAExecutor:
        return CUDAExecutor(nfa, technique)


register_availability(Backend.CUDA, _cuda_available)
