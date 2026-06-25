"""NVIDIA Warp backend — the Python *thread-SIMT* probe on the abstraction spectrum.

Warp (warp-lang) JIT-compiles Python kernels to CUDA C++ with a **thread model**
(``wp.tid()``), so — unlike tile/SPMD DSLs (Triton) or tensor-only DSLs (cuTile,
CuTe, ThunderKittens) — it can express the data-dependent per-state control flow,
dynamic loops and bit manipulation an NFA needs. This makes it the key comparison
point for the "abstraction regret" thesis: same Python-level productivity as Triton,
but a thread model that *can* express the automata kernel directly.

Technique ``multistream``: one thread per input string (the natural Warp idiom), with
the active state-set held in a single register-resident ``uint64`` (≤64 states —
``NWORDS==1``). Mirrors the CUDA ``multistream`` kernel and the ``gpufsm.bitmap`` spec;
validated bit-identical to the reference oracle. NFAs with >64 states raise (a Warp
multi-word local bitset is future work — Warp lacks ergonomic per-thread arrays).

Registers only if ``warp`` imports and a CUDA device is present.
"""

from __future__ import annotations

import time

from ..nfa import ANY_SYMBOL, NFA
from ..registry import Backend, register, register_availability
from ..result import Result

WARP_MAX_STATES = 64


def _warp_available() -> bool:
    try:
        import warp as wp  # noqa: F401

        return bool(wp.get_cuda_device_count() > 0)
    except Exception:
        return False


if _warp_available():  # pragma: no cover - requires GPU + warp
    import numpy as np
    import warp as wp

    wp.config.quiet = True  # suppress the init/compile banner on the CLI
    wp.init()
    _DEV = "cuda"

    @wp.kernel
    def _multistream_kernel(
        sym_row_ptr: wp.array(dtype=wp.int32),
        sym_targets: wp.array(dtype=wp.int32),
        sym_symbols: wp.array(dtype=wp.int32),
        eps_row_ptr: wp.array(dtype=wp.int32),
        eps_targets: wp.array(dtype=wp.int32),
        accept_word: wp.uint64,
        input_data: wp.array(dtype=wp.int32),
        input_offsets: wp.array(dtype=wp.int32),
        num_states: wp.int32,
        start_state: wp.int32,
        uses_any: wp.int32,
        any_id: wp.int32,
        out_flags: wp.array(dtype=wp.int32),
        out_lens: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        lo = input_offsets[i]
        input_len = input_offsets[i + 1] - lo
        one = wp.uint64(1)
        zero = wp.uint64(0)

        cur = one << wp.uint64(start_state)
        # epsilon closure (num_states passes guarantee convergence)
        for _it in range(num_states):
            for s in range(num_states):
                if (cur & (one << wp.uint64(s))) != zero:
                    for k in range(eps_row_ptr[s], eps_row_ptr[s + 1]):
                        cur = cur | (one << wp.uint64(eps_targets[k]))

        # int(...) is intentional: it declares a *mutable* wp.int32 local. A bare
        # literal (0) makes Warp miscompile the later conditional reassignments.
        out_f = int(0)
        out_l = int(0)
        done = int(0)
        if (cur & accept_word) != zero:
            out_f = 1
            done = 1

        pos = int(0)
        while pos < input_len and done == 0:
            sym = input_data[lo + pos]
            nxt = zero
            for s in range(num_states):
                if (cur & (one << wp.uint64(s))) != zero:
                    for k in range(sym_row_ptr[s], sym_row_ptr[s + 1]):
                        tsym = sym_symbols[k]
                        hit = int(0)
                        if tsym == sym:
                            hit = 1
                        if uses_any == 1:
                            if tsym == any_id:
                                hit = 1
                        if hit == 1:
                            nxt = nxt | (one << wp.uint64(sym_targets[k]))
            for _it2 in range(num_states):
                for s in range(num_states):
                    if (nxt & (one << wp.uint64(s))) != zero:
                        for k in range(eps_row_ptr[s], eps_row_ptr[s + 1]):
                            nxt = nxt | (one << wp.uint64(eps_targets[k]))
            cur = nxt
            if (cur & accept_word) != zero:
                out_f = 1
                out_l = pos + 1
                done = 1
            pos = pos + 1

        out_flags[i] = out_f
        out_lens[i] = out_l

    def _accept_word(nfa: NFA) -> int:
        w = 0
        for s in range(nfa.num_states):
            if nfa.accept[s]:
                w |= 1 << s
        return w

    class WarpMultistreamExecutor:
        """One Warp thread per string; ``uint64`` register working set (≤64 states)."""

        def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
            if nfa.num_states > WARP_MAX_STATES:
                raise ValueError(
                    f"warp/multistream supports ≤{WARP_MAX_STATES} states "
                    f"(got {nfa.num_states}); multi-word Warp bitset is future work"
                )
            self.nfa = nfa
            self.technique = technique

            def _dev_i32(a: np.ndarray) -> wp.array:
                return wp.from_numpy(np.ascontiguousarray(a, np.int32), wp.int32, device=_DEV)

            self._srp = _dev_i32(nfa.sym_row_ptr)
            self._st = _dev_i32(nfa.sym_targets)
            self._ss = _dev_i32(nfa.sym_symbols)
            self._erp = _dev_i32(nfa.eps_row_ptr)
            self._et = _dev_i32(nfa.eps_targets)
            self._accept = wp.uint64(_accept_word(nfa))

        def run(self, input_bytes: bytes) -> Result:
            return self.run_batch([input_bytes])[0]

        def run_batch(self, inputs: list[bytes]) -> list[Result]:
            if not inputs:
                return []
            n = len(inputs)
            t0 = time.perf_counter()
            offsets = np.zeros(n + 1, dtype=np.int32)
            for i, b in enumerate(inputs):
                offsets[i + 1] = offsets[i] + len(b)
            data_np = (
                np.frombuffer(b"".join(inputs), dtype=np.uint8).astype(np.int32)
                if offsets[-1] > 0
                else np.zeros(0, dtype=np.int32)
            )
            data = wp.from_numpy(data_np, wp.int32, device=_DEV)
            off = wp.from_numpy(offsets, wp.int32, device=_DEV)
            out_flags = wp.zeros(n, dtype=wp.int32, device=_DEV)
            out_lens = wp.zeros(n, dtype=wp.int32, device=_DEV)
            transfer_ms = (time.perf_counter() - t0) * 1000.0

            wp.synchronize()
            t1 = time.perf_counter()
            wp.launch(
                _multistream_kernel,
                dim=n,
                inputs=[
                    self._srp, self._st, self._ss, self._erp, self._et,
                    self._accept, data, off,
                    int(self.nfa.num_states), int(self.nfa.start_state),
                    int(self.nfa.uses_any_symbol), int(ANY_SYMBOL),
                    out_flags, out_lens,
                ],
                device=_DEV,
            )
            wp.synchronize()
            kernel_ms = (time.perf_counter() - t1) * 1000.0

            flags_h = out_flags.numpy()
            lens_h = out_lens.numpy()
            results: list[Result] = []
            for i in range(n):
                results.append(
                    Result(
                        accepted=bool(flags_h[i]),
                        match_len=int(lens_h[i]),
                        kernel_ms=kernel_ms if i == 0 else 0.0,
                        total_ms=(kernel_ms + transfer_ms) if i == 0 else 0.0,
                        transfer_ms=transfer_ms if i == 0 else 0.0,
                    )
                )
            return results

    @register(Backend.WARP, "multistream")
    def _make_warp_multistream(nfa: NFA, technique: str) -> WarpMultistreamExecutor:
        return WarpMultistreamExecutor(nfa, technique)


register_availability(Backend.WARP, _warp_available)
