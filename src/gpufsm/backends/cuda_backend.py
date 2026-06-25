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

    def _pack_accept(nfa: NFA) -> np.ndarray:
        """Pack the accept set into 64-bit words (the kernel's working-set layout)."""
        nwords = (nfa.num_states + 63) // 64
        words = np.zeros(nwords, dtype=np.uint64)
        for s in range(nfa.num_states):
            if nfa.accept[s]:
                words[s >> 6] |= np.uint64(1) << np.uint64(s & 63)
        return words

    class CUDABitpackedExecutor:
        """Bit-packed working set (1 bit/state). The memory-centric thesis kernel."""

        def __init__(self, nfa: NFA, technique: str = "bitpacked") -> None:
            self.nfa = nfa
            self.technique = technique
            self._accept_words = _pack_accept(nfa)

        def run(self, input_bytes: bytes) -> Result:
            nfa = self.nfa
            t0 = time.perf_counter()
            syms = np.frombuffer(input_bytes, dtype=np.uint8).astype(np.int32)
            transfer_ms = (time.perf_counter() - t0) * 1000.0
            accepted, match_len, kernel_ms = _cuda.run_bitpacked(
                np.ascontiguousarray(nfa.sym_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_targets, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_symbols, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_targets, dtype=np.int32),
                np.ascontiguousarray(self._accept_words, dtype=np.uint64),
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

    @register(Backend.CUDA, "bitpacked")
    def _make_cuda_bitpacked(nfa: NFA, technique: str) -> CUDABitpackedExecutor:
        return CUDABitpackedExecutor(nfa, technique)

    def _pack_inputs(inputs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate a batch into one symbol buffer + CSR-style offsets."""
        offsets = np.zeros(len(inputs) + 1, dtype=np.int32)
        for i, b in enumerate(inputs):
            offsets[i + 1] = offsets[i] + len(b)
        if inputs:
            data = np.frombuffer(b"".join(inputs), dtype=np.uint8).astype(np.int32)
        else:
            data = np.zeros(0, dtype=np.int32)
        return data, offsets

    class CUDAMultistreamExecutor:
        """Multi-stream: one thread/string, whole batch in a single launch.

        ``run_batch`` is the real path; ``run`` is a batch of one. The batch-wide
        kernel time is reported on the first :class:`Result` (0 on the rest), so
        ``sum(r.kernel_ms)`` is the launch time for the whole batch. ``technique``
        selects the CSR placement: ``multistream`` (global CSR) or
        ``multistream_shared`` (read-only CSR staged into shared memory) — the
        global->shared CSR ablation axis.
        """

        _RUNNERS = {
            "multistream": "run_multistream",
            "multistream_shared": "run_multistream_shared",
            "multistream_async": "run_multistream_async",
        }

        def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
            self.nfa = nfa
            self.technique = technique
            self._accept_words = _pack_accept(nfa)
            self._runner = getattr(_cuda, self._RUNNERS[technique])

        def run(self, input_bytes: bytes) -> Result:
            return self.run_batch([input_bytes])[0]

        def run_batch(self, inputs: list[bytes]) -> list[Result]:
            nfa = self.nfa
            t0 = time.perf_counter()
            data, offsets = _pack_inputs(inputs)
            transfer_ms = (time.perf_counter() - t0) * 1000.0
            flags, lens, kernel_ms = self._runner(
                np.ascontiguousarray(nfa.sym_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_targets, dtype=np.int32),
                np.ascontiguousarray(nfa.sym_symbols, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_row_ptr, dtype=np.int32),
                np.ascontiguousarray(nfa.eps_targets, dtype=np.int32),
                np.ascontiguousarray(self._accept_words, dtype=np.uint64),
                np.ascontiguousarray(data, dtype=np.int32),
                np.ascontiguousarray(offsets, dtype=np.int32),
                int(nfa.num_states),
                int(nfa.start_state),
                int(nfa.uses_any_symbol),
            )
            results: list[Result] = []
            for i in range(len(inputs)):
                results.append(
                    Result(
                        accepted=bool(flags[i]),
                        match_len=int(lens[i]),
                        kernel_ms=float(kernel_ms) if i == 0 else 0.0,
                        total_ms=(float(kernel_ms) + transfer_ms) if i == 0 else 0.0,
                        transfer_ms=transfer_ms if i == 0 else 0.0,
                    )
                )
            return results

    @register(Backend.CUDA, "multistream")
    def _make_cuda_multistream(nfa: NFA, technique: str) -> CUDAMultistreamExecutor:
        return CUDAMultistreamExecutor(nfa, technique)

    @register(Backend.CUDA, "multistream_shared")
    def _make_cuda_multistream_shared(nfa: NFA, technique: str) -> CUDAMultistreamExecutor:
        return CUDAMultistreamExecutor(nfa, technique)

    @register(Backend.CUDA, "multistream_async")
    def _make_cuda_multistream_async(nfa: NFA, technique: str) -> CUDAMultistreamExecutor:
        return CUDAMultistreamExecutor(nfa, technique)


register_availability(Backend.CUDA, _cuda_available)
