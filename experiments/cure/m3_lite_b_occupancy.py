"""M3-lite-b — does raising occupancy close the WP2->CUDA gap, or is the intra-warp limit real?

M3-lite showed the per-lane worklist WP2 is latency-bound (issue_active 10%) at BLOCK=32/nw=1:
only ~11 warps/SM (22% occupancy), and a warp's 32 lockstep lanes give no independent work to
hide per-element load latency. Cross-warp hiding is the only lever left in pure Triton → MORE
warps/program (bigger BLOCK, nw=BLOCK//32 so each thread still = 1 string) raises occupancy.

Question: does WP2/CU climb toward ~0.8x (occupancy was binding) or plateau (the intra-warp
latency-hiding limit is fundamental, motivating M3-full)? Oracle-gated; CU = cuda/worklist.
Writes paper2/data/m3_lite_b_occupancy_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m3_lite_b_occupancy.py
        .venv/bin/python experiments/cure/m3_lite_b_occupancy.py profile <block> <ns>
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton
from experiments.cure.m2e_worklist_packed import (
    SAMPLES,
    SLEN,
    WARMUP,
    make_batch,
    random_nfa,
    to_device,
)
from experiments.cure.m3_lite_scalarlane import _wl_perlane, max_outdeg

from gpufsm.api import run, run_batch
from gpufsm.registry import Backend

N_STRINGS = int(os.environ.get("M2_N_STRINGS", "16384"))
BLOCKS = [int(b) for b in os.environ.get("M3B_BLOCKS", "32,64,128,256").split(",")]


def launch_block(g, data2d, maxdeg, block):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _wl_perlane[(triton.cdiv(n, block),)](
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
        BLOCK=block,
        SLEN=SLEN,
        MAXDEG=maxdeg,
        num_warps=max(1, block // 32),
    )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok(nfa, data2d, g, maxdeg, block, n_check=64) -> bool:
    flags, lens, _ = launch_block(g, data2d[:n_check], maxdeg, block)
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        block, ns = int(sys.argv[2]), int(sys.argv[3])
        nfa = random_nfa(ns, seed=1000 + ns)
        g = to_device(nfa)
        launch_block(g, make_batch(0), max_outdeg(nfa), block)
        return 0

    total_bits = N_STRINGS * SLEN * 8
    print(f"M3-lite-b WP2 occupancy sweep (batch={N_STRINGS}), per-lane worklist, oracle-gated.")
    print("  num_warps=BLOCK/32 so each thread = 1 string; bigger BLOCK = more warps/program.\n")
    hdr = "".join(f"{'B=' + str(b):>9}" for b in BLOCKS)
    print(f"{'st':>4}{'sd':>3}{'CU':>9}{hdr}{'best/CU':>9}")
    rows = []

    def cu_gbps(nfa, bb):
        for _ in range(WARMUP):
            run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")
        ms = statistics.median(
            [
                run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")[0].kernel_ms
                for _ in range(SAMPLES)
            ]
        )
        return total_bits / (ms * 1e-3) / 1e9

    def wp2_gbps(g, data, md, block):
        for _ in range(WARMUP):
            launch_block(g, data, md, block)
        ms = statistics.median([launch_block(g, data, md, block)[2] for _ in range(SAMPLES)])
        return total_bits / (ms * 1e-3) / 1e9

    for ns in (16, 32):
        for seed in (0, 1):
            nfa = random_nfa(ns, seed=1000 + ns + seed)
            g = to_device(nfa)
            data = make_batch(seed)
            md = max_outdeg(nfa)
            if not all(oracle_ok(nfa, data, g, md, b) for b in BLOCKS):
                print(f"{ns:4d}{seed:3d}  ORACLE FAIL")
                continue
            cu = cu_gbps(nfa, [data[i].tobytes() for i in range(data.shape[0])])
            gv = {b: wp2_gbps(g, data, md, b) for b in BLOCKS}
            best = max(gv.values())
            cells = "".join(f"{gv[b]:9.1f}" for b in BLOCKS)
            print(f"{ns:4d}{seed:3d}{cu:9.1f}{cells}{best / cu:9.2f}")
            rows.append((ns, seed, round(cu, 2), {b: round(gv[b], 2) for b in BLOCKS}))
    if rows:
        best_ratios = [max(r[3].values()) / r[2] for r in rows]
        print(f"\nbest WP2 (any BLOCK) / CU: median {statistics.median(best_ratios):.2f}x")
        for b in BLOCKS:
            rr = [r[3][b] / r[2] for r in rows]
            print(f"  BLOCK={b:4d}: WP2/CU median {statistics.median(rr):.2f}x")
        outp = Path("paper2/data/m3_lite_b_occupancy_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write(
                "states,seed,n_strings,cu_gbps,"
                + ",".join(f"wp2_b{b}_gbps" for b in BLOCKS)
                + ",gpu\n"
            )
            for ns, seed, cu, gv in rows:
                f.write(
                    f"{ns},{seed},{N_STRINGS},{cu},"
                    + ",".join(f"{gv[b]}" for b in BLOCKS)
                    + ",RTX4070\n"
                )
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
