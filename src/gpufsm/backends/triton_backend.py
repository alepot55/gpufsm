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
    import numpy as np
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

        # latch-first-match: ``done`` freezes the verdict at the first accepting
        # state. Triton forbids ``return`` inside loops, so the per-position body
        # is guarded by ``done == 0`` and the loop runs to completion regardless.
        out_f = 0
        out_l = 0
        done = 0
        for s in range(NUM_STATES):
            if (tl.load(cur + s) == 1) and (tl.load(accept + s) == 1):
                done = 1
        if done == 1:
            out_f = 1
            out_l = 0

        for pos in range(input_len):
            if done == 0:
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
                    out_f = 1
                    out_l = pos + 1
                    done = 1

        tl.store(out_flag, out_f)
        tl.store(out_len, out_l)

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

    # ------------------------------------------------------------------
    # Bit-packed technique — the memory-centric thesis artifact.
    # The active state-set is a packed bitmask (1 bit/state, 64-bit words)
    # instead of one int8 slot/state: byte->bit, the first ablation axis.
    # Same CSR algorithm as ``dense`` (apples-to-apples); only the working-set
    # layout changes. Mirrors the executable spec in :mod:`gpufsm.bitmap`.
    # ------------------------------------------------------------------
    _WORD_BITS = 64

    @triton.jit
    def _bitpacked_kernel(
        sym_row_ptr,
        sym_targets,
        sym_symbols,
        eps_row_ptr,
        eps_targets,
        accept_words,
        input_symbols,
        out_flag,
        out_len,
        cur,
        nxt,
        input_len,
        start_state,
        NUM_STATES: tl.constexpr,
        NWORDS: tl.constexpr,
        ANY_ID: tl.constexpr,
        USES_ANY: tl.constexpr,
    ):
        # One int64 set-bit for runtime targets (avoids int32 overflow at bit 31+
        # and Python/Triton operator ambiguity on ``1 << <runtime>``).
        one = tl.full((), 1, tl.int64)

        # cur := { start_state }
        for w in range(NWORDS):
            tl.store(cur + w, 0)
        sw = start_state >> 6
        sb = one << (start_state & 63)
        tl.store(cur + sw, tl.load(cur + sw) | sb)

        # Epsilon closure: NUM_STATES passes guarantee convergence.
        for _ in range(NUM_STATES):
            for s in range(NUM_STATES):
                wi = s >> 6
                bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
                if (tl.load(cur + wi) & bit) != 0:
                    lo = tl.load(eps_row_ptr + s)
                    hi = tl.load(eps_row_ptr + s + 1)
                    for k in range(lo, hi):
                        t = tl.load(eps_targets + k)
                        twi = t >> 6
                        tbit = one << (t & 63)
                        tl.store(cur + twi, tl.load(cur + twi) | tbit)

        # Word-parallel accept test (NWORDS iterations, not NUM_STATES).
        out_f = 0
        out_l = 0
        done = 0
        for w in range(NWORDS):
            if (tl.load(cur + w) & tl.load(accept_words + w)) != 0:
                done = 1
        if done == 1:
            out_f = 1
            out_l = 0

        for pos in range(input_len):
            if done == 0:
                sym = tl.load(input_symbols + pos)
                for w in range(NWORDS):
                    tl.store(nxt + w, 0)
                for s in range(NUM_STATES):
                    wi = s >> 6
                    bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
                    if (tl.load(cur + wi) & bit) != 0:
                        lo = tl.load(sym_row_ptr + s)
                        hi = tl.load(sym_row_ptr + s + 1)
                        for k in range(lo, hi):
                            tsym = tl.load(sym_symbols + k)
                            hit = tsym == sym
                            if USES_ANY:
                                hit = hit or (tsym == ANY_ID)
                            if hit:
                                t = tl.load(sym_targets + k)
                                twi = t >> 6
                                tbit = one << (t & 63)
                                tl.store(nxt + twi, tl.load(nxt + twi) | tbit)
                # epsilon closure on nxt
                for _ in range(NUM_STATES):
                    for s in range(NUM_STATES):
                        wi = s >> 6
                        bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
                        if (tl.load(nxt + wi) & bit) != 0:
                            lo = tl.load(eps_row_ptr + s)
                            hi = tl.load(eps_row_ptr + s + 1)
                            for k in range(lo, hi):
                                t = tl.load(eps_targets + k)
                                twi = t >> 6
                                tbit = one << (t & 63)
                                tl.store(nxt + twi, tl.load(nxt + twi) | tbit)
                for w in range(NWORDS):
                    tl.store(cur + w, tl.load(nxt + w))
                m = 0
                for w in range(NWORDS):
                    if (tl.load(cur + w) & tl.load(accept_words + w)) != 0:
                        m = 1
                if m == 1:
                    out_f = 1
                    out_l = pos + 1
                    done = 1

        tl.store(out_flag, out_f)
        tl.store(out_len, out_l)

    def _pack_accept(nfa: NFA, nwords: int) -> np.ndarray:
        words = np.zeros(nwords, dtype=np.int64)
        for s in range(nfa.num_states):
            if nfa.accept[s]:
                words[s >> 6] |= np.int64(1) << np.int64(s & 63)
        return words

    class TritonBitpackedExecutor:
        def __init__(self, nfa: NFA, technique: str = "bitpacked") -> None:
            self.nfa = nfa
            self.technique = technique
            self._nwords = (nfa.num_states + _WORD_BITS - 1) // _WORD_BITS
            dev = torch.device("cuda")
            self._dev = dev
            self._sym_row_ptr = torch.as_tensor(nfa.sym_row_ptr, device=dev)
            self._sym_targets = torch.as_tensor(nfa.sym_targets, device=dev)
            self._sym_symbols = torch.as_tensor(nfa.sym_symbols, device=dev)
            self._eps_row_ptr = torch.as_tensor(nfa.eps_row_ptr, device=dev)
            self._eps_targets = torch.as_tensor(nfa.eps_targets, device=dev)
            self._accept_words = torch.as_tensor(_pack_accept(nfa, self._nwords), device=dev)
            self._cur = torch.zeros(self._nwords, dtype=torch.int64, device=dev)
            self._nxt = torch.zeros(self._nwords, dtype=torch.int64, device=dev)

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
            _bitpacked_kernel[(1,)](
                self._sym_row_ptr,
                self._sym_targets,
                self._sym_symbols,
                self._eps_row_ptr,
                self._eps_targets,
                self._accept_words,
                inp,
                flag,
                mlen,
                self._cur,
                self._nxt,
                int(syms.size),
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                NWORDS=int(self._nwords),
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

    @register(Backend.TRITON, "bitpacked")
    def _make_triton_bitpacked(nfa: NFA, technique: str) -> TritonBitpackedExecutor:
        return TritonBitpackedExecutor(nfa, technique)

    # ------------------------------------------------------------------
    # Multi-stream technique — single->multi-stream ablation axis.
    # grid=(num_strings,): one program per input string, all strings concurrent.
    # Same bit-packed working set, sliced per program (cur/nxt are N*NWORDS).
    # Inputs are a single concatenated buffer + per-string offsets.
    # ------------------------------------------------------------------
    @triton.jit
    def _multistream_kernel(
        sym_row_ptr,
        sym_targets,
        sym_symbols,
        eps_row_ptr,
        eps_targets,
        accept_words,
        input_data,
        input_offsets,
        num_strings,
        out_flags,
        out_lens,
        cur,
        nxt,
        start_state,
        NUM_STATES: tl.constexpr,
        NWORDS: tl.constexpr,
        ANY_ID: tl.constexpr,
        USES_ANY: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < num_strings:
            base = pid * NWORDS
            in_lo = tl.load(input_offsets + pid)
            input_len = tl.load(input_offsets + pid + 1) - in_lo
            one = tl.full((), 1, tl.int64)

            for w in range(NWORDS):
                tl.store(cur + base + w, 0)
            sw = start_state >> 6
            tl.store(cur + base + sw, tl.load(cur + base + sw) | (one << (start_state & 63)))

            for _ in range(NUM_STATES):
                for s in range(NUM_STATES):
                    wi = s >> 6
                    bit = one << (s & 63)
                    if (tl.load(cur + base + wi) & bit) != 0:
                        lo = tl.load(eps_row_ptr + s)
                        hi = tl.load(eps_row_ptr + s + 1)
                        for k in range(lo, hi):
                            t = tl.load(eps_targets + k)
                            twi = base + (t >> 6)
                            tl.store(cur + twi, tl.load(cur + twi) | (one << (t & 63)))

            out_f = 0
            out_l = 0
            done = 0
            for w in range(NWORDS):
                if (tl.load(cur + base + w) & tl.load(accept_words + w)) != 0:
                    done = 1
            if done == 1:
                out_f = 1
                out_l = 0

            for pos in range(input_len):
                if done == 0:
                    sym = tl.load(input_data + in_lo + pos)
                    for w in range(NWORDS):
                        tl.store(nxt + base + w, 0)
                    for s in range(NUM_STATES):
                        wi = s >> 6
                        bit = one << (s & 63)
                        if (tl.load(cur + base + wi) & bit) != 0:
                            lo = tl.load(sym_row_ptr + s)
                            hi = tl.load(sym_row_ptr + s + 1)
                            for k in range(lo, hi):
                                tsym = tl.load(sym_symbols + k)
                                hit = tsym == sym
                                if USES_ANY:
                                    hit = hit or (tsym == ANY_ID)
                                if hit:
                                    t = tl.load(sym_targets + k)
                                    twi = base + (t >> 6)
                                    tl.store(nxt + twi, tl.load(nxt + twi) | (one << (t & 63)))
                    for _ in range(NUM_STATES):
                        for s in range(NUM_STATES):
                            wi = s >> 6
                            bit = one << (s & 63)
                            if (tl.load(nxt + base + wi) & bit) != 0:
                                lo = tl.load(eps_row_ptr + s)
                                hi = tl.load(eps_row_ptr + s + 1)
                                for k in range(lo, hi):
                                    t = tl.load(eps_targets + k)
                                    twi = base + (t >> 6)
                                    tl.store(nxt + twi, tl.load(nxt + twi) | (one << (t & 63)))
                    for w in range(NWORDS):
                        tl.store(cur + base + w, tl.load(nxt + base + w))
                    m = 0
                    for w in range(NWORDS):
                        if (tl.load(cur + base + w) & tl.load(accept_words + w)) != 0:
                            m = 1
                    if m == 1:
                        out_f = 1
                        out_l = pos + 1
                        done = 1

            tl.store(out_flags + pid, out_f)
            tl.store(out_lens + pid, out_l)

    class TritonMultistreamExecutor:
        """Multi-stream: grid=(num_strings,), whole batch in a single launch.

        ``run_batch`` is the real path; ``run`` is a batch of one. The batch-wide
        kernel time is reported on the first :class:`Result` (0 on the rest).
        """

        def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
            self.nfa = nfa
            self.technique = technique
            self._nwords = (nfa.num_states + _WORD_BITS - 1) // _WORD_BITS
            dev = torch.device("cuda")
            self._dev = dev
            self._sym_row_ptr = torch.as_tensor(nfa.sym_row_ptr, device=dev)
            self._sym_targets = torch.as_tensor(nfa.sym_targets, device=dev)
            self._sym_symbols = torch.as_tensor(nfa.sym_symbols, device=dev)
            self._eps_row_ptr = torch.as_tensor(nfa.eps_row_ptr, device=dev)
            self._eps_targets = torch.as_tensor(nfa.eps_targets, device=dev)
            self._accept_words = torch.as_tensor(_pack_accept(nfa, self._nwords), device=dev)

        def run(self, input_bytes: bytes) -> Result:
            return self.run_batch([input_bytes])[0]

        def run_batch(self, inputs: list[bytes]) -> list[Result]:
            if not inputs:
                return []
            dev = self._dev
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
            data = torch.as_tensor(data_np, device=dev)
            off = torch.as_tensor(offsets, device=dev)
            transfer_ms = (time.perf_counter() - t0) * 1000.0

            flags = torch.zeros(n, dtype=torch.int32, device=dev)
            lens = torch.zeros(n, dtype=torch.int32, device=dev)
            cur = torch.zeros(n * self._nwords, dtype=torch.int64, device=dev)
            nxt = torch.zeros(n * self._nwords, dtype=torch.int64, device=dev)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _multistream_kernel[(n,)](
                self._sym_row_ptr,
                self._sym_targets,
                self._sym_symbols,
                self._eps_row_ptr,
                self._eps_targets,
                self._accept_words,
                data,
                off,
                n,
                flags,
                lens,
                cur,
                nxt,
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                NWORDS=int(self._nwords),
                ANY_ID=int(ANY_SYMBOL),
                USES_ANY=bool(self.nfa.uses_any_symbol),
            )
            end.record()
            torch.cuda.synchronize()
            kernel_ms = float(start.elapsed_time(end))

            flags_h = flags.cpu().numpy()
            lens_h = lens.cpu().numpy()
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

    @register(Backend.TRITON, "multistream")
    def _make_triton_multistream(nfa: NFA, technique: str) -> TritonMultistreamExecutor:
        return TritonMultistreamExecutor(nfa, technique)


register_availability(Backend.TRITON, _triton_available)
