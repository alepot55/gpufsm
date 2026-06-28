"""M9 — multi-word lane-packed worklist: does the decomposition hold beyond 64 states?

The single-word WP2 prototype caps at 64 states (one int64 working set per lane). This generalizes
it to NWORDS int64 words/lane (working set = a [BLOCK, NWORDS] tile), removing the "<=64-state
prototype" threat. The per-lane next-state SCATTER (set bit `tgt` in word `tgt//64`, where `tgt`
differs per lane) uses a uniform masked accumulate over words -- viable for small NWORDS (<=8);
ANMLZoo-scale automata (Brill 42661 = 667 words) would make this O(NWORDS)/transition scatter
explode, itself a manifestation of the tile-scatter limit we study. We validate correctness up to
256 states. FINDING: the prototype generalizes + stays oracle-correct, but the >64-state compare
is CONFOUNDED -- cuda/worklist's register fast-path degrades past 64 states (register pressure), so
MW/CU flips above 1 not because Triton closed the residual but because CUDA lost its register
advantage. The clean ~2x residual is a <=64-state register-resident-regime result.

eps-free synthetic NFAs (random_nfa) -> eps-closure omitted (valid). Oracle-gated.
Writes paper2/data/m9_multiword_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m9_multiword.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from experiments.cure.m2e_worklist_packed import (
    SAMPLES,
    SLEN,
    WARMUP,
    make_batch,
    random_nfa,
    to_device,
)
from triton.language.extra import libdevice

from gpufsm.api import run, run_batch
from gpufsm.registry import Backend

N_STRINGS = 4096  # matches m2e.make_batch's batch size
BLOCK = 32


@triton.jit
def _wl_perlane_mw(
    rowptr,
    targets,
    symbols,
    accept_words,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    start_state,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
    BLOCK: tl.constexpr,
    SLEN: tl.constexpr,
    NWORDS: tl.constexpr,
    MAXDEG: tl.constexpr,
):
    pid = tl.program_id(0)
    widx = tl.arange(0, NWORDS)[None, :]  # [1,NWORDS]
    sidx = pid * BLOCK + tl.arange(0, BLOCK)
    valid = sidx < num_strings
    in_lo = tl.load(input_offsets + sidx, mask=valid, other=0)
    in_len = tl.load(input_offsets + sidx + 1, mask=valid, other=0) - in_lo
    valid2 = valid[:, None] & (widx >= 0)  # [BLOCK,NWORDS]
    zero = tl.zeros((BLOCK, NWORDS), tl.int64)
    one = tl.full((BLOCK, NWORDS), 1, tl.int64)
    accw = tl.load(accept_words + tl.arange(0, NWORDS))[None, :]  # [1,NWORDS]
    sw = start_state // 64
    sb = start_state - sw * 64
    cur = tl.where((widx == sw) & valid2, one << sb, zero)  # [BLOCK,NWORDS]
    done = (tl.sum(cur & accw, axis=1) != 0) & valid  # [BLOCK]
    out_f = tl.where(done, 1, 0)
    out_l = tl.zeros((BLOCK,), tl.int32)
    for pos in range(SLEN):
        step = (pos < in_len) & (done == 0) & valid  # [BLOCK]
        step2 = step[:, None] & (widx >= 0)
        sym = tl.load(input_data + in_lo + pos, mask=step, other=-1)  # [BLOCK]
        nxt = zero
        bits = tl.where(step2, cur, zero)  # [BLOCK,NWORDS]
        # process each word's active states; per-lane ffs within the word
        for w in range(NWORDS):
            bw = tl.sum(tl.where(widx == w, bits, zero), axis=1)  # [BLOCK]: word w, per lane
            while tl.max((bw != 0).to(tl.int32)) > 0:
                anyb = bw != 0  # [BLOCK]
                sl = libdevice.ffs(bw) - 1  # [BLOCK]: local bit
                s = w * 64 + sl  # [BLOCK]: global state per lane
                s_safe = tl.where(anyb, s, 0)
                bw = bw & (bw - 1)
                lo = tl.load(rowptr + s_safe, mask=anyb, other=0)  # [BLOCK]
                hi = tl.load(rowptr + s_safe + 1, mask=anyb, other=0)
                deg = hi - lo
                for kk in range(MAXDEG):
                    kact = anyb & (kk < deg)  # [BLOCK]
                    tsym = tl.load(symbols + lo + kk, mask=kact, other=-1)  # [BLOCK]
                    tgt = tl.load(targets + lo + kk, mask=kact, other=0)  # [BLOCK]
                    hit = tsym == sym
                    if USES_ANY:
                        hit = hit | (tsym == ANY_ID)
                    setlane = kact & hit  # [BLOCK]: this lane sets state tgt
                    tw = tgt // 64  # [BLOCK]: target word
                    tb = tgt - tw * 64  # [BLOCK]: target bit
                    # scatter: for each word column wt, set the bit if tw==wt
                    contrib = tl.where(
                        setlane[:, None] & (widx == tw[:, None]), one << tb[:, None], zero
                    )  # [BLOCK,NWORDS]
                    nxt = nxt | contrib
        cur = tl.where(step2, nxt, cur)
        acc = (tl.sum(cur & accw, axis=1) != 0) & (done == 0) & step  # [BLOCK]
        out_l = tl.where(acc, pos + 1, out_l)
        out_f = tl.where(acc, 1, out_f)
        done = done | acc
    tl.store(out_flags + sidx, out_f, mask=valid)
    tl.store(out_lens + sidx, out_l, mask=valid)


def pack_accept(nfa, nwords):
    a = np.zeros(nwords, dtype=np.int64)
    for s in range(nfa.num_states):
        if nfa.accept[s]:
            a[s // 64] |= np.int64(1) << np.int64(s % 64)
    return a


def max_outdeg(nfa):
    rp = np.asarray(nfa.sym_row_ptr)
    return int(np.max(rp[1:] - rp[:-1]))


def launch_mw(nfa, g, data2d, nwords, maxdeg):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    accw = torch.as_tensor(pack_accept(nfa, nwords), device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _wl_perlane_mw[(triton.cdiv(n, BLOCK),)](
        g["rowptr"],
        g["targets"],
        g["symbols"],
        accw,
        flat,
        offsets,
        n,
        flags,
        lens,
        g["start"],
        ANY_ID=g["ANY"],
        USES_ANY=g["USES_ANY"],
        BLOCK=BLOCK,
        SLEN=SLEN,
        NWORDS=nwords,
        MAXDEG=maxdeg,
        num_warps=1,
    )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok(nfa, data2d, g, nwords, maxdeg, n_check=64) -> bool:
    flags, lens, _ = launch_mw(nfa, g, data2d[:n_check], nwords, maxdeg)
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            print(
                f"  MISMATCH mw {i}: got ({bool(flags[i])},{lens[i]}) "
                f"ref ({ref.accepted},{ref.match_len})"
            )
            return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    total_bits = N_STRINGS * SLEN * 8
    print(f"M9 multi-word lane-packed worklist (batch={N_STRINGS}), oracle-gated:")
    print(f"{'states':>7}{'nwords':>7}{'MW_Gbps':>9}{'CU_Gbps':>9}{'MW/CU':>7}{'oracle':>8}")
    rows = []
    for ns in (96, 128, 192, 256):
        need = (ns + 63) // 64
        nwords = 1 << (need - 1).bit_length()  # next power of 2 (tl.arange needs pow2)
        nfa = random_nfa(ns, seed=1000 + ns)
        g = to_device(nfa)
        data = make_batch(0)
        md = max_outdeg(nfa)
        if not oracle_ok(nfa, data, g, nwords, md):
            print(f"{ns:7d}{nwords:7d}  ORACLE FAIL")
            continue
        batch_bytes = [data[i].tobytes() for i in range(data.shape[0])]

        def cu(bb=batch_bytes, nfa=nfa):
            # register-resident CUDA worklist (<=512 states) = the FAIR fast baseline (not the
            # slow global-memory worklist_global)
            for _ in range(WARMUP):
                run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")
            return statistics.median(
                [
                    run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")[0].kernel_ms
                    for _ in range(SAMPLES)
                ]
            )

        def mw(nfa=nfa, g=g, data=data, nwords=nwords, md=md):
            for _ in range(WARMUP):
                launch_mw(nfa, g, data, nwords, md)
            return statistics.median(
                [launch_mw(nfa, g, data, nwords, md)[2] for _ in range(SAMPLES)]
            )

        gmw = total_bits / (mw() * 1e-3) / 1e9
        gcu = total_bits / (cu() * 1e-3) / 1e9
        print(f"{ns:7d}{nwords:7d}{gmw:9.2f}{gcu:9.2f}{gmw / gcu:7.2f}{'  ok':>8}")
        rows.append((ns, nwords, round(gmw, 3), round(gcu, 3), round(gmw / gcu, 3)))
    if rows:
        r = [x[4] for x in rows]
        print(
            f"\nMW/CU (vs cuda/worklist-reg): median {statistics.median(r):.2f}x "
            f"(>64 states, NWORDS 2-4) — tests if the residual holds beyond 64 states"
        )
        outp = Path("paper2/data/m9_multiword_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("states,nwords,mw_gbps,cu_worklist_gbps,mw_over_cu,gpu\n")
            for ns, nw, gmw, gcu, ratio in rows:
                f.write(f"{ns},{nw},{gmw},{gcu},{ratio},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
