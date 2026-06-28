"""M4 — does the decomposition generalize to the memory-bound DFA gather kernel?

The NFA is control-flow/latency-bound; M3-lite/-b showed lane-packing leaves a fundamental ~2x
intra-warp latency residual there. The DFA is paper-1's MEMORY-bound second face: one dependent
table gather per symbol (trans[cur*256 + sym]), no active set, no ffs, no union — the clean case
for lane-packing. PREDICTION: because the DFA is memory-bound, a tile gather issues 32 concurrent
memory requests (one/lane) = similar memory-level parallelism to CUDA's 32 independent threads, so
lane-packing should CLOSE the gap here (unlike the latency-bound NFA). If so, the residual depends
on the REGIME (control-flow vs memory) = paper-1's two faces, generalizing the cure decomposition.

Compares DFA scalar Triton (existing kernel, num_warps=4 default = the artifact), DFA lane-packed
Triton (new), and DFA CUDA, oracle-gated by simulate_dfa, across table sizes spanning the memory
hierarchy (cache -> L2 -> DRAM). Writes paper2/data/m4_dfa_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m4_dfa.py
        .venv/bin/python experiments/cure/m4_dfa.py profile <scalar|packed> <ns>
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

from gpufsm.dfa import random_dfa, simulate_dfa
from gpufsm.dfa_api import run_dfa_batch

SLEN = 256
N_STRINGS = int(os.environ.get("M2_N_STRINGS", "16384"))
WARMUP, SAMPLES = 3, 9
BLOCK = 32


@triton.jit
def _dfa_packed_kernel(
    trans,
    accept,
    input_data,
    input_offsets,
    num_strings,
    start_state,
    out_flags,
    out_lens,
    BLOCK: tl.constexpr,
    SLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    sidx = pid * BLOCK + lane
    valid = sidx < num_strings
    lo = tl.load(input_offsets + sidx, mask=valid, other=0)
    in_len = tl.load(input_offsets + sidx + 1, mask=valid, other=0) - lo
    cur = tl.full((BLOCK,), start_state, tl.int32)
    done = (tl.load(accept + cur, mask=valid, other=0) != 0) & valid
    out_f = tl.where(done, 1, 0)
    out_l = tl.zeros((BLOCK,), tl.int32)
    for pos in range(SLEN):
        step = (pos < in_len) & (done == 0) & valid
        sym = tl.load(input_data + lo + pos, mask=step, other=0)
        nxt = tl.load(
            trans + cur * 256 + sym, mask=step, other=0
        )  # per-lane gather (scalar-gather)
        cur = tl.where(step, nxt, cur)
        acc = tl.load(accept + cur, mask=step, other=0) != 0
        newly = acc & (done == 0) & step
        out_l = tl.where(newly, pos + 1, out_l)
        out_f = tl.where(newly, 1, out_f)
        done = done | newly
    tl.store(out_flags + sidx, out_f, mask=valid)
    tl.store(out_lens + sidx, out_l, mask=valid)


def make_batch(seed: int, n: int):
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, 256, size=n * SLEN, dtype=np.uint8)
    return flat.reshape(n, SLEN)


def launch_packed(dfa, data2d):
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    d_trans = torch.as_tensor(dfa.trans, device=dev)
    d_acc = torch.as_tensor(dfa.accept.astype(np.int32), device=dev)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _dfa_packed_kernel[(triton.cdiv(n, BLOCK),)](
        d_trans,
        d_acc,
        flat,
        offsets,
        n,
        int(dfa.start_state),
        flags,
        lens,
        BLOCK=BLOCK,
        SLEN=SLEN,
        num_warps=1,
    )
    ev1.record()
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ev0.elapsed_time(ev1))


def oracle_ok_packed(dfa, data2d, n_check=64) -> bool:
    flags, lens, _ = launch_packed(dfa, data2d[:n_check])
    for i in range(min(n_check, data2d.shape[0])):
        acc, mlen = simulate_dfa(dfa, data2d[i].tobytes())
        if (bool(flags[i]), int(lens[i])) != (acc, mlen):
            print(f"  MISMATCH packed string {i}: ({bool(flags[i])},{lens[i]}) vs ({acc},{mlen})")
            return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "profile":
        ns = int(sys.argv[2])
        dfa = random_dfa(ns, seed=0)
        launch_packed(dfa, make_batch(0, N_STRINGS))
        return 0

    total_bits = N_STRINGS * SLEN * 8
    print(f"M4 DFA decomposition (batch={N_STRINGS}), table sizes span the memory hierarchy:")
    print("  TR=scalar Triton (nw=4, the artifact)  PK=lane-packed Triton  CU=cuda")
    print("  PK/CU near 1.0 would show lane-packing CLOSES the gap in the memory-bound regime.\n")
    print(
        f"{'states':>7}{'tableKB':>9}{'TR':>9}{'PK':>9}{'CU':>9}{'PK/CU':>8}{'PK/TR':>8}{'ok':>4}"
    )
    rows = []
    for ns in (64, 1024, 4096, 16384):
        dfa = random_dfa(ns, seed=0)
        data = make_batch(0, N_STRINGS)
        if not oracle_ok_packed(dfa, data):
            print(f"{ns:7d}  ORACLE FAIL packed")
            continue
        batch_bytes = [data[i].tobytes() for i in range(data.shape[0])]

        def cu_tr(backend, dfa=dfa, bb=batch_bytes):
            for _ in range(WARMUP):
                run_dfa_batch(dfa, bb, backend=backend)
            return statistics.median(
                [run_dfa_batch(dfa, bb, backend=backend)[0].kernel_ms for _ in range(SAMPLES)]
            )

        def pk(dfa=dfa, data=data):
            for _ in range(WARMUP):
                launch_packed(dfa, data)
            return statistics.median([launch_packed(dfa, data)[2] for _ in range(SAMPLES)])

        tr = total_bits / (cu_tr("triton") * 1e-3) / 1e9
        cu = total_bits / (cu_tr("cuda") * 1e-3) / 1e9
        pkg = total_bits / (pk() * 1e-3) / 1e9
        tkb = ns * 256 * 4 / 1024
        rr = f"{pkg / cu:8.2f}{pkg / tr:8.2f}"
        print(f"{ns:7d}{tkb:9.0f}{tr:9.1f}{pkg:9.1f}{cu:9.1f}{rr}{'  ok':>4}")
        rows.append((ns, round(tkb, 1), round(tr, 2), round(pkg, 2), round(cu, 2)))
    if rows:
        pkcu = [r[3] / r[4] for r in rows]
        pktr = [r[3] / r[2] for r in rows]
        print(
            f"\nPK/CU (lane-packed vs CUDA): median {statistics.median(pkcu):.2f}x  "
            f"(near 1.0 = packing closes the DFA gap)"
        )
        print(f"PK/TR (packing vs scalar-nw4 Triton): median {statistics.median(pktr):.2f}x")
        outp = Path("paper2/data/m4_dfa_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("states,table_kb,triton_scalar_gbps,triton_packed_gbps,cuda_gbps,gpu\n")
            for ns, tkb, tr, pkg, cu in rows:
                f.write(f"{ns},{tkb},{tr},{pkg},{cu},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
