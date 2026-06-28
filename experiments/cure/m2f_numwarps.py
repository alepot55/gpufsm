"""M2f — num_warps precision sweep on the scalar worklist: size the launch-config artifact.

M2e found the gpufsm triton/worklist defaults to num_warps=4, so 4 warps/program redundantly run
ONE string. This sweeps num_warps in {1,2,4,8} on the SAME scalar worklist kernel (1 string/program)
to size the artifact exactly: throughput should fall as num_warps rises (more redundant warps). The
nw=4 column is the gpufsm/paper-1 baseline; nw=1 is the tuned one. Oracle-gated. Sweeps batch since
the effect interacts with occupancy. Writes paper2/data/m2f_numwarps_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m2f_numwarps.py
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

# reuse the validated kernel + helpers from M2e (import does not run its main())
from experiments.cure.m2e_worklist_packed import (
    SAMPLES,
    SLEN,
    WARMUP,
    _wl_scalar,
    random_nfa,
    to_device,
)

from gpufsm.api import run
from gpufsm.registry import Backend

BATCHES = [int(b) for b in os.environ.get("M2F_BATCHES", "4096,16384,65536").split(",")]
NUM_WARPS = [1, 2, 4, 8]


def launch_nw(g, data2d, num_warps):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
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
        num_warps=num_warps,
    )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok(nfa, data2d, g, num_warps, n_check=64) -> bool:
    flags, lens, _ = launch_nw(g, data2d[:n_check], num_warps)
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            return False
    return True


def med_gbps(g, data2d, num_warps, total_bits):
    for _ in range(WARMUP):
        launch_nw(g, data2d, num_warps)
    ms = statistics.median([launch_nw(g, data2d, num_warps)[2] for _ in range(SAMPLES)])
    return total_bits / (ms * 1e-3) / 1e9


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    ns = 32
    rows = []
    print("num_warps sweep on the scalar worklist (1 string/program, ns=32), oracle-gated:")
    print("  throughput should FALL as num_warps rises (redundant warps/program)\n")
    hdr = "".join(f"{'nw=' + str(w):>10}" for w in NUM_WARPS)
    print(f"{'batch':>7}{hdr}{'nw1/nw4':>10}")
    for batch in BATCHES:
        data = make_batch_for(ns, batch)
        nfa = random_nfa(ns, seed=1000 + ns)
        g = to_device(nfa)
        total_bits = batch * SLEN * 8
        gv = {}
        ok = True
        for w in NUM_WARPS:
            if not oracle_ok(nfa, data, g, w):
                ok = False
                break
            gv[w] = med_gbps(g, data, w, total_bits)
        if not ok:
            print(f"{batch:7d}  ORACLE FAIL")
            continue
        cells = "".join(f"{gv[w]:10.1f}" for w in NUM_WARPS)
        ratio = gv[1] / gv[4]
        print(f"{batch:7d}{cells}{ratio:10.2f}")
        rows.append((batch, gv))
    if rows:
        ratios = [r[1][1] / r[1][4] for r in rows]
        print(
            f"\nnum_warps artifact (nw=1 / nw=4): median {statistics.median(ratios):.2f}x  "
            f"min {min(ratios):.2f}x  max {max(ratios):.2f}x"
        )
        outp = Path("paper2/data/m2f_numwarps_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("batch," + ",".join(f"nw{w}_gbps" for w in NUM_WARPS) + ",nw1_over_nw4,gpu\n")
            for batch, gv in rows:
                f.write(
                    f"{batch},"
                    + ",".join(f"{gv[w]:.3f}" for w in NUM_WARPS)
                    + f",{gv[1] / gv[4]:.3f},RTX4070\n"
                )
        print(f"wrote {outp}")
    return 0


def make_batch_for(ns, batch):
    # build a batch of an explicit size (m2e.make_batch reads a module-level N_STRINGS)
    rng = np.random.default_rng(0)
    flat = rng.integers(ord("a"), ord("a") + 5, size=batch * SLEN, dtype=np.uint8)
    return flat.reshape(batch, SLEN)


if __name__ == "__main__":
    sys.exit(main())
