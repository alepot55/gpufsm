"""M2e — lane-pack the WORK-EFFICIENT worklist: where the irreducible primitive bites.

M2c settled the DENSE scan: lane-packing recovers most of the warp-redundancy at scale. But the
M0 anchor (10x) was the WORKLIST (ffs over the active set, O(active)/string), not the dense scan.
The genuine missing primitive — per-lane data-dependent control flow — should bite HERE:

  - scalar worklist (WS): 1 string/program, `while bits: s=ffs(bits)` iterates ONLY that string's
    active states (O(active)). Per-lane work-skipping is natural in the thread shape.
  - lane-packed worklist (WP): BLOCK strings/program, one per lane. The warp is in lockstep, so it
    must iterate the UNION of active states across all 32 lanes (ffs over OR-reduced `cur`), doing
    each state for every lane (masked). For decorrelated strings the union -> all NS states, so WP
    DEGENERATES to the dense lane-packed kernel: lane-packing DESTROYS the ffs work-efficiency.

Decisive contrast (the paper's key figure):
  - scalar:      worklist BEATS dense (ffs-skip helps in the thread model).
  - lane-packed: worklist == dense (ffs-skip is lost; union cost dominates) -> WP/WS small.
=> the work-efficiency that the worklist buys is expressible only with per-lane control flow, which
   tile/SPMD lacks. That is the irreducible regret behind the M0 anchor.

All kernels oracle-gated (reference.py) before any Gbps. eps-free NFAs (eps-closure omitted, valid).
Writes paper2/data/m2e_worklist_packed_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m2e_worklist_packed.py
        .venv/bin/python experiments/cure/m2e_worklist_packed.py profile <ws|wp> <ns>
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
from triton.language.extra import libdevice

from gpufsm.api import run, run_batch
from gpufsm.nfa import ANY_SYMBOL, NFABuilder
from gpufsm.registry import Backend

SLEN = 256
N_STRINGS = int(os.environ.get("M2_N_STRINGS", "16384"))
ALPHABET = "abcde"
WARMUP, SAMPLES = 3, 9
BLOCK = 32


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _wl_scalar(
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
                bits = cur
                while bits != zero:  # iterate ONLY this string's active states (O(active))
                    s = libdevice.ffs(bits) - 1
                    bits = bits & (bits - 1)
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
def _wl_packed(
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
    sidx = pid * BLOCK + lane
    valid = sidx < num_strings
    in_lo = tl.load(input_offsets + sidx, mask=valid, other=0)
    in_len = tl.load(input_offsets + sidx + 1, mask=valid, other=0) - in_lo
    one = tl.full((BLOCK,), 1, tl.int64)
    zero = tl.zeros((BLOCK,), tl.int64)
    cur = one << start_state
    acc = (cur & accept_word) != 0
    done = acc & valid
    out_f = tl.where(done, 1, 0)
    out_l = tl.zeros((BLOCK,), tl.int32)
    for pos in range(SLEN):
        step = (pos < in_len) & (done == 0) & valid
        sym = tl.load(input_data + in_lo + pos, mask=step, other=-1)
        nxt = zero
        ucur = tl.where(step, cur, zero)
        union = tl.reduce(ucur, 0, _or_combine)  # scalar OR over lanes -> the active-set UNION
        bits = union
        while bits != 0:  # iterate the UNION (lockstep) — per-lane ffs-skip is impossible
            s = libdevice.ffs(bits) - 1
            bits = bits & (bits - 1)
            s_active = ((cur >> s) & 1) != 0  # [BLOCK]: which lanes actually have s
            for k in range(tl.load(rowptr + s), tl.load(rowptr + s + 1)):
                tsym = tl.load(symbols + k)
                tgt = tl.load(targets + k)
                hit = sym == tsym
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


def launch(kind, g, data2d):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    if kind == "ws":
        _wl_scalar[(n,)](
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
        _wl_packed[grid](
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


def profile_one(kind, ns):
    nfa = random_nfa(ns, seed=1000 + ns)
    g = to_device(nfa)
    launch(kind, g, make_batch(0))
    return 0


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        return profile_one(sys.argv[2], int(sys.argv[3]))
    rows = []
    print(f"4-way worklist head-to-head (batch={N_STRINGS}, BLOCK={BLOCK}), all oracle-gated:")
    print("  CU=cuda/worklist  WT=triton/worklist gpufsm (num_warps=4, the M0 baseline)")
    print("  WS=my scalar worklist (num_warps=1)  WP=lane-packed worklist")
    print("  regret_M0 = CU/WT (the 10x anchor)   regret_nw1 = CU/WS")
    print("  WP/CU = does lane-packed Triton beat hand-CUDA?\n")
    print(
        f"{'st':>4}{'sd':>3}{'CU':>9}{'WT':>9}{'WS':>9}{'WP':>9}{'CU/WT':>7}{'CU/WS':>7}{'WP/CU':>7}{'ok':>4}"
    )
    total_bits = N_STRINGS * SLEN * 8

    def gbps_runbatch(nfa, batch_bytes, technique):
        for _ in range(WARMUP):
            run_batch(
                nfa,
                batch_bytes,
                backend=Backend.CUDA if technique == "cu" else Backend.TRITON,
                technique="worklist",
            )
        be = Backend.CUDA if technique == "cu" else Backend.TRITON
        ms = statistics.median(
            [
                run_batch(nfa, batch_bytes, backend=be, technique="worklist")[0].kernel_ms
                for _ in range(SAMPLES)
            ]
        )
        return total_bits / (ms * 1e-3) / 1e9

    for ns in (8, 16, 24, 32):
        for seed in (0, 1):
            nfa = random_nfa(ns, seed=1000 + ns + seed)
            g = to_device(nfa)
            data = make_batch(seed)
            batch_bytes = [data[i].tobytes() for i in range(data.shape[0])]
            if not (oracle_ok(nfa, data, "ws", g) and oracle_ok(nfa, data, "wp", g)):
                print(f"{ns:4d}{seed:3d}  ORACLE FAIL")
                continue
            cu = gbps_runbatch(nfa, batch_bytes, "cu")
            wt = gbps_runbatch(nfa, batch_bytes, "tr")
            ws = total_bits / (med_ms("ws", g, data) * 1e-3) / 1e9
            wp = total_bits / (med_ms("wp", g, data) * 1e-3) / 1e9
            print(
                f"{ns:4d}{seed:3d}{cu:9.1f}{wt:9.1f}{ws:9.1f}{wp:9.1f}"
                f"{cu / wt:7.2f}{cu / ws:7.2f}{wp / cu:7.2f}{'  ok':>4}"
            )
            rows.append((ns, seed, round(cu, 2), round(wt, 2), round(ws, 2), round(wp, 2)))
    if rows:
        cuwt = [r[2] / r[3] for r in rows]
        cuws = [r[2] / r[4] for r in rows]
        wpcu = [r[5] / r[2] for r in rows]
        print(f"\nregret_M0  CU/WT (num_warps=4): median {statistics.median(cuwt):.2f}x")
        print(
            f"regret_nw1 CU/WS (num_warps=1): median {statistics.median(cuws):.2f}x  "
            f"<-- how much of the anchor was just launch config"
        )
        print(
            f"WP/CU lane-packed Triton vs CUDA: median {statistics.median(wpcu):.2f}x  "
            f"(>1 = Triton BEATS hand-CUDA worklist)"
        )
        outp = Path("paper2/data/m2e_worklist_packed_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        write_header = not outp.exists()
        with outp.open("a") as f:
            if write_header:
                f.write("states,seed,n_strings,cu_gbps,wt_gbps,ws_gbps,wp_gbps,gpu\n")
            for ns, seed, cu, wt, ws, wp in rows:
                f.write(f"{ns},{seed},{N_STRINGS},{cu},{wt},{ws},{wp},RTX4070\n")
        print(f"appended {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
