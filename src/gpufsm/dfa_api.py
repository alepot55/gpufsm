"""Run a DFA over a batch of inputs on CPU (oracle) or CUDA (memory-bound kernel)."""

from __future__ import annotations

import time

import numpy as np

from .dfa import DFA, simulate_dfa
from .result import Result


def _pack(inputs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(inputs) + 1, dtype=np.int32)
    for i, b in enumerate(inputs):
        offsets[i + 1] = offsets[i] + len(b)
    data = (
        np.frombuffer(b"".join(inputs), dtype=np.uint8).astype(np.int32)
        if offsets[-1] > 0
        else np.zeros(0, dtype=np.int32)
    )
    return data, offsets


def run_dfa_batch(dfa: DFA, inputs: list[bytes], backend: str = "cuda") -> list[Result]:
    """Run ``dfa`` over a batch. ``backend``: ``cpu`` (reference oracle) or ``cuda``.

    The batch-wide kernel time is reported on the first :class:`Result` (0 on the rest),
    mirroring the NFA multi-stream executors.
    """
    if not inputs:
        return []
    if backend == "cpu":
        out = []
        for b in inputs:
            acc, mlen = simulate_dfa(dfa, b)
            out.append(Result(accepted=acc, match_len=mlen))
        return out
    if backend == "triton":
        from .dfa_backends import run_dfa_triton

        return run_dfa_triton(dfa, inputs)
    if backend == "warp":
        from .dfa_backends import run_dfa_warp

        return run_dfa_warp(dfa, inputs)
    if backend != "cuda":
        raise ValueError(f"unknown DFA backend {backend!r} (use cpu/cuda/triton/warp)")

    import gpufsm.backends.cuda._cuda as _cuda

    t0 = time.perf_counter()
    data, offsets = _pack(inputs)
    transfer_ms = (time.perf_counter() - t0) * 1000.0
    flags, lens, kernel_ms = _cuda.run_dfa(
        np.ascontiguousarray(dfa.trans, dtype=np.int32),
        np.ascontiguousarray(dfa.accept, dtype=np.int8),
        np.ascontiguousarray(data, dtype=np.int32),
        np.ascontiguousarray(offsets, dtype=np.int32),
        int(dfa.num_states),
        int(dfa.start_state),
    )
    return [
        Result(
            accepted=bool(flags[i]),
            match_len=int(lens[i]),
            kernel_ms=float(kernel_ms) if i == 0 else 0.0,
            total_ms=(float(kernel_ms) + transfer_ms) if i == 0 else 0.0,
            transfer_ms=transfer_ms if i == 0 else 0.0,
        )
        for i in range(len(inputs))
    ]
