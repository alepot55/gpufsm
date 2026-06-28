"""M2a — lane-packed Triton worklist: does supplying the missing primitive close the regret?

M1 pinned the 10x regret to H1: each Triton *program* runs ONE string redundantly across all
32 lanes of its warp (thread_inst_per_inst=32, all lanes the SAME string), so Triton issues
~90x more warp-instructions than CUDA (1 thread = 1 string, 32 strings/warp). The missing
primitive is "lane-level task parallelism": pack 32 independent strings into the 32 lanes of one
program.

This isolates that primitive cleanly on the DENSE NFA-scan algorithm (no ffs work-skipping, so
no active-set-union confound): the SAME kernel body run two ways —
  A (scalar):     1 string per program, scalar int64 state (the status quo Triton shape).
  B (lane-packed): BLOCK=32 strings per program, a [BLOCK] int64 state tile (lane j = string j).
Only the packing differs. The CSR (rowptr/targets/symbols) is SHARED across strings, so iterating
states uniformly `for s in range(NS)` keeps the inner CSR loop bounds SCALAR (a function of s, not
the lane) — expressible in Triton; only the active-set and the input symbol are per-lane tiles.

Both validated bit-for-bit vs the reference.py oracle BEFORE any throughput. NFAs are epsilon-free
(random_nfa), so eps-closure is a no-op and omitted (valid for these NFAs; eps is a uniform
extension). Writes paper2/data/m2_lane_packed_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m2_lane_packed.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from gpufsm.api import run
from gpufsm.nfa import ANY_SYMBOL, NFABuilder
from gpufsm.registry import Backend

SLEN = 256
N_STRINGS = int(os.environ.get("M2_N_STRINGS", "4096"))  # M2c sweeps batch to test occupancy
ALPHABET = "abcde"
WARMUP, SAMPLES = 3, 9
BLOCK = 32  # one warp: 32 strings packed into 32 lanes


@triton.jit
def _dense_scalar_kernel(
    rowptr,
    targets,
    symbols,
    accept_word,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    start_state,
    NS: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
    SLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_strings:
        in_lo = tl.load(input_offsets + pid)
        in_len = tl.load(input_offsets + pid + 1) - in_lo
        one = tl.full((), 1, tl.int64)
        zero = tl.full((), 0, tl.int64)
        cur = one << start_state
        out_f = 0
        out_l = 0
        done = 0
        if (cur & accept_word) != zero:
            out_f = 1
            done = 1
        for pos in range(SLEN):
            if pos < in_len and done == 0:
                sym = tl.load(input_data + in_lo + pos)
                nxt = zero
                for s in range(NS):
                    if ((cur >> s) & one) != zero:
                        for k in range(tl.load(rowptr + s), tl.load(rowptr + s + 1)):
                            tsym = tl.load(symbols + k)
                            hit = tsym == sym
                            if USES_ANY:
                                hit = hit or (tsym == ANY_ID)
                            if hit:
                                nxt = nxt | (one << tl.load(targets + k))
                cur = nxt
                if (cur & accept_word) != zero:
                    out_f = 1
                    out_l = pos + 1
                    done = 1
        tl.store(out_flags + pid, out_f)
        tl.store(out_lens + pid, out_l)


@triton.jit
def _dense_scalar_noskip_kernel(
    rowptr,
    targets,
    symbols,
    accept_word,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    start_state,
    NS: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
    SLEN: tl.constexpr,
):
    # Identical to _dense_scalar_kernel but WITHOUT the per-state `if active` skip: the CSR row
    # loop runs for every state unconditionally (contribution masked by s_active). This mirrors
    # EXACTLY the per-string work the lane-packed kernel must do (O(NS), no per-lane skipping),
    # so noskip-scalar vs lane-packed isolates the PURE lane-packing effect (1 string/warp ->
    # 32 strings/warp) with the work held identical.
    pid = tl.program_id(0)
    if pid < num_strings:
        in_lo = tl.load(input_offsets + pid)
        in_len = tl.load(input_offsets + pid + 1) - in_lo
        one = tl.full((), 1, tl.int64)
        zero = tl.full((), 0, tl.int64)
        cur = one << start_state
        out_f = 0
        out_l = 0
        done = 0
        if (cur & accept_word) != zero:
            out_f = 1
            done = 1
        for pos in range(SLEN):
            if pos < in_len and done == 0:
                sym = tl.load(input_data + in_lo + pos)
                nxt = zero
                for s in range(NS):
                    s_active = ((cur >> s) & one) != zero
                    for k in range(tl.load(rowptr + s), tl.load(rowptr + s + 1)):
                        tsym = tl.load(symbols + k)
                        hit = tsym == sym
                        if USES_ANY:
                            hit = hit or (tsym == ANY_ID)
                        nxt = nxt | tl.where(s_active and hit, one << tl.load(targets + k), zero)
                cur = nxt
                if (cur & accept_word) != zero:
                    out_f = 1
                    out_l = pos + 1
                    done = 1
        tl.store(out_flags + pid, out_f)
        tl.store(out_lens + pid, out_l)


@triton.jit
def _dense_lane_packed_kernel(
    rowptr,
    targets,
    symbols,
    accept_word,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    start_state,
    NS: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
    BLOCK: tl.constexpr,
    SLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    sidx = pid * BLOCK + lane  # [BLOCK]: which string each lane processes
    valid = sidx < num_strings
    in_lo = tl.load(input_offsets + sidx, mask=valid, other=0)
    in_hi = tl.load(input_offsets + sidx + 1, mask=valid, other=0)
    in_len = in_hi - in_lo
    one = tl.full((BLOCK,), 1, tl.int64)
    zero = tl.zeros((BLOCK,), tl.int64)
    cur = one << start_state  # [BLOCK]
    acc = (cur & accept_word) != 0
    done = acc & valid
    out_f = tl.where(done, 1, 0)
    out_l = tl.zeros((BLOCK,), tl.int32)
    for pos in range(SLEN):
        step = (pos < in_len) & (done == 0) & valid  # [BLOCK]
        sym = tl.load(input_data + in_lo + pos, mask=step, other=-1)  # [BLOCK]
        nxt = zero
        for s in range(NS):
            s_active = ((cur >> s) & 1) != 0  # [BLOCK]
            lo = tl.load(rowptr + s)
            hi = tl.load(rowptr + s + 1)
            for k in range(lo, hi):  # scalar bounds (shared CSR)
                tsym = tl.load(symbols + k)
                tgt = tl.load(targets + k)
                hit = sym == tsym  # [BLOCK]
                if USES_ANY:
                    hit = hit | (tsym == ANY_ID)
                nxt = nxt | tl.where(s_active & hit, one << tgt, zero)
        cur = tl.where(step, nxt, cur)
        acc = (cur & accept_word) != 0
        newly = acc & (done == 0) & step
        out_l = tl.where(newly, pos + 1, out_l)
        out_f = tl.where(newly, 1, out_f)
        done = done | newly
    tl.store(out_flags + sidx, out_f, mask=valid)
    tl.store(out_lens + sidx, out_l, mask=valid)


def random_nfa(n: int, seed: int):
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice(ALPHABET)), rng.randrange(n))
    return b.build()


def make_batch(seed: int):
    rng = np.random.default_rng(seed)
    flat = rng.integers(ord("a"), ord("a") + len(ALPHABET), size=N_STRINGS * SLEN, dtype=np.uint8)
    return flat.reshape(N_STRINGS, SLEN)


def to_device(nfa):
    dev = torch.device("cuda")
    acc = 0
    for s in range(nfa.num_states):
        if nfa.accept[s]:
            acc |= 1 << s
    return {
        "rowptr": torch.as_tensor(nfa.sym_row_ptr, device=dev),
        "targets": torch.as_tensor(nfa.sym_targets, device=dev),
        "symbols": torch.as_tensor(nfa.sym_symbols, device=dev),
        "accept_word": int(acc),
        "start": int(nfa.start_state),
        "NS": int(nfa.num_states),
        "ANY": int(ANY_SYMBOL),
        "USES_ANY": bool(nfa.uses_any_symbol),
    }


def launch(kernel_kind, g, data2d):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    if kernel_kind in ("scalar", "noskip"):
        kern = _dense_scalar_kernel if kernel_kind == "scalar" else _dense_scalar_noskip_kernel
        kern[(n,)](
            g["rowptr"],
            g["targets"],
            g["symbols"],
            g["accept_word"],
            flat,
            offsets,
            n,
            flags,
            lens,
            g["start"],
            NS=g["NS"],
            ANY_ID=g["ANY"],
            USES_ANY=g["USES_ANY"],
            SLEN=SLEN,
            num_warps=1,
        )
    else:
        grid = (triton.cdiv(n, BLOCK),)
        _dense_lane_packed_kernel[grid](
            g["rowptr"],
            g["targets"],
            g["symbols"],
            g["accept_word"],
            flat,
            offsets,
            n,
            flags,
            lens,
            g["start"],
            NS=g["NS"],
            ANY_ID=g["ANY"],
            USES_ANY=g["USES_ANY"],
            BLOCK=BLOCK,
            SLEN=SLEN,
            num_warps=1,
        )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok(nfa, data2d, kind, g, n_check=64) -> bool:
    flags, lens, _ = launch(kind, g, data2d[:n_check])
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            print(
                f"  MISMATCH {kind} string {i}: got ({bool(flags[i])},{lens[i]}) "
                f"ref ({ref.accepted},{ref.match_len})"
            )
            return False
    return True


def med_ms(kind, g, data2d):
    for _ in range(WARMUP):
        launch(kind, g, data2d)
    ts = [launch(kind, g, data2d)[2] for _ in range(SAMPLES)]
    return statistics.median(ts)


def profile_one(kind: str, ns: int) -> int:
    """Single-launch target for Nsight: `... profile <scalar|noskip|packed> <ns>`."""
    nfa = random_nfa(ns, seed=1000 + ns)
    g = to_device(nfa)
    data = make_batch(0)
    launch(kind, g, data)  # exactly one profiled launch
    return 0


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        return profile_one(sys.argv[2], int(sys.argv[3]))
    rows = []
    print(f"lane-packing the DENSE NFA scan (BLOCK={BLOCK} strings/program, 1 warp):")
    print("  A=skip-scalar (O(active)/string)  B=noskip-scalar (O(NS))  C=lane-packed (O(NS)/warp)")
    print(
        "  C/B = PURE lane-packing (work held equal);  C/A = realistic (vs work-efficient scalar)\n"
    )
    print(
        f"{'states':>7}{'seed':>5}{'A_skip':>9}{'B_nosk':>9}{'C_pack':>9}{'C/B':>7}{'C/A':>7}{'orac':>6}"
    )
    total_bits = N_STRINGS * SLEN * 8
    for ns in (8, 16, 24, 32):
        for seed in (0, 1):
            nfa = random_nfa(ns, seed=1000 + ns + seed)
            g = to_device(nfa)
            data = make_batch(seed)
            oks = all(oracle_ok(nfa, data, k, g) for k in ("scalar", "noskip", "packed"))
            if not oks:
                print(f"{ns:7d}{seed:5d}  ORACLE FAIL")
                continue
            ga = total_bits / (med_ms("scalar", g, data) * 1e-3) / 1e9
            gb = total_bits / (med_ms("noskip", g, data) * 1e-3) / 1e9
            gc = total_bits / (med_ms("packed", g, data) * 1e-3) / 1e9
            cb, ca = gc / gb, gc / ga
            print(f"{ns:7d}{seed:5d}{ga:9.1f}{gb:9.1f}{gc:9.1f}{cb:7.2f}{ca:7.2f}{'  ok':>6}")
            rows.append(
                (ns, seed, round(ga, 3), round(gb, 3), round(gc, 3), round(cb, 3), round(ca, 3))
            )
    if rows:
        cb_all = [r[5] for r in rows]
        ca_all = [r[6] for r in rows]
        print(
            f"\nPURE lane-packing  C/B: median {statistics.median(cb_all):.2f}x  "
            f"min {min(cb_all):.2f}x  max {max(cb_all):.2f}x"
        )
        print(
            f"realistic (vs skip) C/A: median {statistics.median(ca_all):.2f}x  "
            f"min {min(ca_all):.2f}x  max {max(ca_all):.2f}x"
        )
        outp = Path("paper2/data/m2_lane_packed_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("states,seed,A_skip_gbps,B_noskip_gbps,C_packed_gbps,C_over_B,C_over_A,gpu\n")
            for ns, seed, ga, gb, gc, cb, ca in rows:
                f.write(f"{ns},{seed},{ga},{gb},{gc},{cb},{ca},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
