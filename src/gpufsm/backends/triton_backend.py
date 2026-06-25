"""Triton backend — high-level block-based DSL kernels for NFA simulation.

Status: ported and structurally complete, but **validated on GPU only** (this
backend registers itself solely when ``torch`` + ``triton`` + a CUDA device are
present). The kernel mirrors :func:`gpufsm.reference.simulate` exactly; a
``@pytest.mark.gpu`` test asserts verdict-equality with the CPU oracle.

Technique ``dense`` is the faithful, correctness-first kernel (one int8 slot per
state — the legacy default, intentionally kept as the *abstraction-regret* example).
The bit-packed Triton technique that embodies the memory thesis builds on the
spec in :mod:`gpufsm.bitmap` and is added once validated on hardware.
"""

from __future__ import annotations

import time

from ..nfa import ANY_SYMBOL, NFA
from ..registry import Backend, register, register_availability
from ..result import Result


def _triton_available() -> bool:
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if _triton_available():  # pragma: no cover - requires GPU
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _dense_kernel(
        sym_row_ptr,
        sym_targets,
        sym_symbols,
        eps_row_ptr,
        eps_targets,
        accept,
        input_symbols,
        out_flag,
        out_len,
        cur,
        nxt,
        input_len,
        start_state,
        NUM_STATES: tl.constexpr,
        ANY_ID: tl.constexpr,
        USES_ANY: tl.constexpr,
    ):
        # Single-program NFA simulation (latch-first-match). One int8 slot/state.
        for i in range(NUM_STATES):
            tl.store(cur + i, 0)
        tl.store(cur + start_state, 1)

        # Epsilon closure: NUM_STATES passes guarantee convergence.
        for _ in range(NUM_STATES):
            for s in range(NUM_STATES):
                if tl.load(cur + s) == 1:
                    lo = tl.load(eps_row_ptr + s)
                    hi = tl.load(eps_row_ptr + s + 1)
                    for k in range(lo, hi):
                        tl.store(cur + tl.load(eps_targets + k), 1)

        matched = 0
        for s in range(NUM_STATES):
            if (tl.load(cur + s) == 1) and (tl.load(accept + s) == 1):
                matched = 1
        if matched == 1:
            tl.store(out_flag, 1)
            tl.store(out_len, 0)
            return

        for pos in range(input_len):
            sym = tl.load(input_symbols + pos)
            for i in range(NUM_STATES):
                tl.store(nxt + i, 0)
            for s in range(NUM_STATES):
                if tl.load(cur + s) == 1:
                    lo = tl.load(sym_row_ptr + s)
                    hi = tl.load(sym_row_ptr + s + 1)
                    for k in range(lo, hi):
                        tsym = tl.load(sym_symbols + k)
                        hit = tsym == sym
                        if USES_ANY:
                            hit = hit or (tsym == ANY_ID)
                        if hit:
                            tl.store(nxt + tl.load(sym_targets + k), 1)
            # epsilon closure on nxt
            for _ in range(NUM_STATES):
                for s in range(NUM_STATES):
                    if tl.load(nxt + s) == 1:
                        lo = tl.load(eps_row_ptr + s)
                        hi = tl.load(eps_row_ptr + s + 1)
                        for k in range(lo, hi):
                            tl.store(nxt + tl.load(eps_targets + k), 1)
            for i in range(NUM_STATES):
                tl.store(cur + i, tl.load(nxt + i))
            m = 0
            for s in range(NUM_STATES):
                if (tl.load(cur + s) == 1) and (tl.load(accept + s) == 1):
                    m = 1
            if m == 1:
                tl.store(out_flag, 1)
                tl.store(out_len, pos + 1)
                return

        tl.store(out_flag, 0)
        tl.store(out_len, 0)

    class TritonExecutor:
        def __init__(self, nfa: NFA, technique: str = "dense") -> None:
            self.nfa = nfa
            self.technique = technique
            dev = torch.device("cuda")
            self._dev = dev
            self._sym_row_ptr = torch.as_tensor(nfa.sym_row_ptr, device=dev)
            self._sym_targets = torch.as_tensor(nfa.sym_targets, device=dev)
            self._sym_symbols = torch.as_tensor(nfa.sym_symbols, device=dev)
            self._eps_row_ptr = torch.as_tensor(nfa.eps_row_ptr, device=dev)
            self._eps_targets = torch.as_tensor(nfa.eps_targets, device=dev)
            self._accept = torch.as_tensor(nfa.accept.astype("int8"), device=dev)
            self._cur = torch.zeros(nfa.num_states, dtype=torch.int8, device=dev)
            self._nxt = torch.zeros(nfa.num_states, dtype=torch.int8, device=dev)

        def run(self, input_bytes: bytes) -> Result:
            import numpy as np

            dev = self._dev
            t0 = time.perf_counter()
            syms = np.frombuffer(input_bytes, dtype=np.uint8).astype(np.int32)
            inp = torch.as_tensor(syms, device=dev)
            transfer_ms = (time.perf_counter() - t0) * 1000.0

            flag = torch.zeros(1, dtype=torch.int32, device=dev)
            mlen = torch.zeros(1, dtype=torch.int32, device=dev)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _dense_kernel[(1,)](
                self._sym_row_ptr,
                self._sym_targets,
                self._sym_symbols,
                self._eps_row_ptr,
                self._eps_targets,
                self._accept,
                inp,
                flag,
                mlen,
                self._cur,
                self._nxt,
                int(syms.size),
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                ANY_ID=int(ANY_SYMBOL),
                USES_ANY=bool(self.nfa.uses_any_symbol),
            )
            end.record()
            torch.cuda.synchronize()
            kernel_ms = float(start.elapsed_time(end))
            return Result(
                accepted=bool(flag.item()),
                match_len=int(mlen.item()),
                kernel_ms=kernel_ms,
                total_ms=kernel_ms + transfer_ms,
                transfer_ms=transfer_ms,
            )

    @register(Backend.TRITON, "dense")
    def _make_triton(nfa: NFA, technique: str) -> TritonExecutor:
        return TritonExecutor(nfa, technique)


register_availability(Backend.TRITON, _triton_available)
