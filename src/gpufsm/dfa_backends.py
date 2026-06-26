"""Triton and Warp DFA kernels — the cross-DSL comparison on the memory-bound workload.

Each runs one program/thread per string doing the scalar DFA walk
``cur = trans[cur*256 + byte]`` per byte (a data-dependent gather + a sequential chain).
This lets us measure abstraction regret on the MEMORY-bound face (DFA), complementing the
control-flow face (NFA). Registered lazily; used via :func:`gpufsm.dfa_api.run_dfa_batch`.
"""

from __future__ import annotations

import time

import numpy as np

from .dfa import DFA
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


def _results(flags, lens, kernel_ms: float, transfer_ms: float, n: int) -> list[Result]:
    return [
        Result(
            accepted=bool(flags[i]),
            match_len=int(lens[i]),
            kernel_ms=kernel_ms if i == 0 else 0.0,
            total_ms=(kernel_ms + transfer_ms) if i == 0 else 0.0,
            transfer_ms=transfer_ms if i == 0 else 0.0,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------- Triton
def triton_available() -> bool:
    try:
        import torch
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if triton_available():  # pragma: no cover - requires GPU
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _dfa_triton_kernel(
        trans, accept, input_data, input_offsets, num_strings, start_state, out_flags, out_lens
    ):
        pid = tl.program_id(0)
        if pid < num_strings:
            lo = tl.load(input_offsets + pid)
            hi = tl.load(input_offsets + pid + 1)
            cur = start_state
            out_f = 0
            out_l = 0
            done = 0
            if tl.load(accept + cur) != 0:
                out_f = 1
                done = 1
            for pos in range(lo, hi):
                if done == 0:
                    sym = tl.load(input_data + pos)
                    cur = tl.load(trans + cur * 256 + sym)
                    if tl.load(accept + cur) != 0:
                        out_f = 1
                        out_l = pos - lo + 1
                        done = 1
            tl.store(out_flags + pid, out_f)
            tl.store(out_lens + pid, out_l)

    def run_dfa_triton(dfa: DFA, inputs: list[bytes]) -> list[Result]:
        dev = torch.device("cuda")
        n = len(inputs)
        t0 = time.perf_counter()
        data_np, offsets = _pack(inputs)
        d_trans = torch.as_tensor(dfa.trans, device=dev)
        d_acc = torch.as_tensor(dfa.accept.astype(np.int32), device=dev)
        d_data = torch.as_tensor(data_np, device=dev)
        d_off = torch.as_tensor(offsets, device=dev)
        flags = torch.zeros(n, dtype=torch.int32, device=dev)
        lens = torch.zeros(n, dtype=torch.int32, device=dev)
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        _dfa_triton_kernel[(n,)](
            d_trans, d_acc, d_data, d_off, n, int(dfa.start_state), flags, lens
        )
        ev1.record()
        torch.cuda.synchronize()
        kernel_ms = float(ev0.elapsed_time(ev1))
        return _results(flags.cpu().numpy(), lens.cpu().numpy(), kernel_ms, transfer_ms, n)


# ---------------------------------------------------------------- Warp
def warp_available() -> bool:
    try:
        import warp as wp

        return bool(wp.get_cuda_device_count() > 0)
    except Exception:
        return False


if warp_available():  # pragma: no cover - requires GPU
    import warp as wp

    wp.config.quiet = True
    wp.init()

    @wp.kernel
    def _dfa_warp_kernel(
        trans: wp.array(dtype=wp.int32),
        accept: wp.array(dtype=wp.int32),
        input_data: wp.array(dtype=wp.int32),
        input_offsets: wp.array(dtype=wp.int32),
        num_strings: wp.int32,
        start_state: wp.int32,
        out_flags: wp.array(dtype=wp.int32),
        out_lens: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        lo = input_offsets[i]
        hi = input_offsets[i + 1]
        cur = start_state
        out_f = int(0)  # int(...) declares a mutable wp.int32 local; bare 0 miscompiles
        out_l = int(0)
        done = int(0)
        if accept[cur] != 0:
            out_f = 1
            done = 1
        pos = lo
        while pos < hi and done == 0:
            cur = trans[cur * 256 + input_data[pos]]
            if accept[cur] != 0:
                out_f = 1
                out_l = pos - lo + 1
                done = 1
            pos = pos + 1
        out_flags[i] = out_f
        out_lens[i] = out_l

    def run_dfa_warp(dfa: DFA, inputs: list[bytes]) -> list[Result]:
        n = len(inputs)
        t0 = time.perf_counter()
        data_np, offsets = _pack(inputs)
        d_trans = wp.from_numpy(np.ascontiguousarray(dfa.trans, np.int32), wp.int32, device="cuda")
        d_acc = wp.from_numpy(dfa.accept.astype(np.int32), wp.int32, device="cuda")
        d_data = wp.from_numpy(data_np, wp.int32, device="cuda")
        d_off = wp.from_numpy(offsets, wp.int32, device="cuda")
        flags = wp.zeros(n, dtype=wp.int32, device="cuda")
        lens = wp.zeros(n, dtype=wp.int32, device="cuda")
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        wp.synchronize()
        t1 = time.perf_counter()
        wp.launch(
            _dfa_warp_kernel,
            dim=n,
            inputs=[d_trans, d_acc, d_data, d_off, n, int(dfa.start_state), flags, lens],
            device="cuda",
        )
        wp.synchronize()
        kernel_ms = (time.perf_counter() - t1) * 1000.0
        return _results(flags.numpy(), lens.numpy(), kernel_ms, transfer_ms, n)
