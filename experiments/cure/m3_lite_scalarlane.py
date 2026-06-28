"""M3-lite — a per-lane scalar worklist in PURE Triton: does removing the tile-tax close the gap?

M2e localized the irreducible residual to a per-instruction "tile tax": the lane-packed worklist
(WP) formed the active-set UNION with a cross-lane `tl.reduce` every position, then iterated the
union with masking. Nsight: at matched occupancy/warps/warp-inst WP was still 3.67x slower.

This kernel (WP2) is the cleanest pure-Triton approximation of a "scalar-lane program":
  - each lane keeps its OWN active set `bits` and computes its OWN next state `s = ffs(bits)` —
    NO cross-lane reduce, NO union.
  - per-lane CSR rows are GATHERED (`tl.load(rowptr + s)` with s a [BLOCK] tensor) — the
    "scalar-gather-in-tile" primitive made explicit.
  - the only uniform control is the outer `while tl.max(any active)` (lockstep to the busiest lane)
    and the inner `for kk in range(MAXDEG)` (uniform max out-degree, masked) — no union work.

If WP2 closes most of the M2e gap to CUDA, the primitive is LATENT (expressible, just not idiomatic)
and M3-full is about ergonomics. If a gap persists, it bounds what the real IR primitive must buy.
Compared head-to-head vs CUDA (CU) and the M2e union worklist (WP), oracle-gated, swept over batch.
Writes paper2/data/m3_lite_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m3_lite_scalarlane.py
        .venv/bin/python experiments/cure/m3_lite_scalarlane.py profile <ns>
"""

from __future__ import annotations

import os
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
from experiments.cure.m2e_worklist_packed import (
    launch as launch_m2e,  # WP (union) launcher
)
from triton.language.extra import libdevice

from gpufsm.api import run, run_batch
from gpufsm.registry import Backend

N_STRINGS = int(os.environ.get("M2_N_STRINGS", "16384"))
BLOCK = 32


@triton.jit
def _wl_perlane(
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
    MAXDEG: tl.constexpr,
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
    done = ((cur & accept_word) != 0) & valid
    out_f = tl.where(done, 1, 0)
    out_l = tl.zeros((BLOCK,), tl.int32)
    for pos in range(SLEN):
        step = (pos < in_len) & (done == 0) & valid
        sym = tl.load(input_data + in_lo + pos, mask=step, other=-1)
        nxt = zero
        bits = tl.where(step, cur, zero)  # each lane its OWN active set (no union)
        # lockstep to the busiest lane; each lane consumes its own active states via per-lane ffs
        while tl.max((bits != 0).to(tl.int32)) > 0:
            anyb = bits != 0
            s = libdevice.ffs(bits) - 1  # [BLOCK]: per-lane lowest active state
            s_safe = tl.where(anyb, s, 0)
            bits = bits & (bits - 1)
            lo = tl.load(rowptr + s_safe, mask=anyb, other=0)  # per-lane CSR row (gather)
            hi = tl.load(rowptr + s_safe + 1, mask=anyb, other=0)
            deg = hi - lo
            for kk in range(MAXDEG):  # uniform max out-degree, masked per lane
                kactive = anyb & (kk < deg)
                k = lo + kk
                tsym = tl.load(symbols + k, mask=kactive, other=-1)
                tgt = tl.load(targets + k, mask=kactive, other=0)
                hit = sym == tsym
                if USES_ANY:
                    hit = hit | (tsym == ANY_ID)
                nxt = nxt | tl.where(kactive & hit, one << tgt, zero)
        cur = tl.where(step, nxt, cur)
        newly = ((cur & accept_word) != 0) & (done == 0) & step
        out_l = tl.where(newly, pos + 1, out_l)
        out_f = tl.where(newly, 1, out_f)
        done = done | newly
    tl.store(out_flags + sidx, out_f, mask=valid)
    tl.store(out_lens + sidx, out_l, mask=valid)


def max_outdeg(nfa) -> int:
    rp = np.asarray(nfa.sym_row_ptr)
    return int(np.max(rp[1:] - rp[:-1])) if nfa.num_states > 0 else 1


def launch_wp2(g, data2d, maxdeg):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _wl_perlane[(triton.cdiv(n, BLOCK),)](
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
        MAXDEG=maxdeg,
        num_warps=1,
    )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok_wp2(nfa, data2d, g, maxdeg, n_check=64) -> bool:
    flags, lens, _ = launch_wp2(g, data2d[:n_check], maxdeg)
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            print(
                f"  MISMATCH wp2 string {i}: got ({bool(flags[i])},{lens[i]}) "
                f"ref ({ref.accepted},{ref.match_len})"
            )
            return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "profile":
        ns = int(sys.argv[2])
        nfa = random_nfa(ns, seed=1000 + ns)
        g = to_device(nfa)
        launch_wp2(g, make_batch(0), max_outdeg(nfa))
        return 0

    rows = []
    print("M3-lite per-lane scalar worklist (WP2) vs CUDA (CU) and M2e union worklist (WP)")
    print(f"batch={N_STRINGS}, BLOCK={BLOCK}, oracle-gated. WP2/CU>WP/CU means removing the union")
    print("cross-lane reduce helped close the tile-tax.\n")
    print(f"{'st':>4}{'sd':>3}{'CU':>9}{'WP':>9}{'WP2':>9}{'WP2/CU':>8}{'WP2/WP':>8}{'ok':>4}")
    total_bits = N_STRINGS * SLEN * 8

    def med_gbps_runbatch(nfa, batch_bytes):
        for _ in range(WARMUP):
            run_batch(nfa, batch_bytes, backend=Backend.CUDA, technique="worklist")
        ms = statistics.median(
            [
                run_batch(nfa, batch_bytes, backend=Backend.CUDA, technique="worklist")[0].kernel_ms
                for _ in range(SAMPLES)
            ]
        )
        return total_bits / (ms * 1e-3) / 1e9

    def med_gbps(launch_fn):
        for _ in range(WARMUP):
            launch_fn()
        ms = statistics.median([launch_fn()[2] for _ in range(SAMPLES)])
        return total_bits / (ms * 1e-3) / 1e9

    for ns in (8, 16, 24, 32):
        for seed in (0, 1):
            nfa = random_nfa(ns, seed=1000 + ns + seed)
            g = to_device(nfa)
            data = make_batch(seed)
            md = max_outdeg(nfa)
            if not oracle_ok_wp2(nfa, data, g, md):
                print(f"{ns:4d}{seed:3d}  ORACLE FAIL wp2")
                continue
            batch_bytes = [data[i].tobytes() for i in range(data.shape[0])]
            cu = med_gbps_runbatch(nfa, batch_bytes)
            wp = med_gbps(lambda g=g, data=data: launch_m2e("wp", g, data))
            wp2 = med_gbps(lambda g=g, data=data, md=md: launch_wp2(g, data, md))
            ratios = f"{wp2 / cu:8.2f}{wp2 / wp:8.2f}"
            print(f"{ns:4d}{seed:3d}{cu:9.1f}{wp:9.1f}{wp2:9.1f}{ratios}{'  ok':>4}")
            rows.append((ns, seed, round(cu, 2), round(wp, 2), round(wp2, 2)))
    if rows:
        wp2cu = [r[4] / r[2] for r in rows]
        wp2wp = [r[4] / r[3] for r in rows]
        print(f"\nWP2/CU (per-lane Triton vs CUDA): median {statistics.median(wp2cu):.2f}x")
        print(
            f"WP2/WP (per-lane vs union):       median {statistics.median(wp2wp):.2f}x  "
            f"(>1 = removing the cross-lane union reduce helped)"
        )
        outp = Path("paper2/data/m3_lite_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        write_header = not outp.exists()
        with outp.open("a") as f:
            if write_header:
                f.write("states,seed,n_strings,cu_gbps,wp_gbps,wp2_gbps,gpu\n")
            for ns, seed, cu, wp, wp2 in rows:
                f.write(f"{ns},{seed},{N_STRINGS},{cu},{wp},{wp2},RTX4070\n")
        print(f"appended {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
