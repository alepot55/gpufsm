"""F3 gating measurement: is the lock-step `while`'s per-iteration `tt.reduce` gate worth removing
via an in-compiler reduce-hoist (`while tl.max(j<trip)>0`  ->  `mt=tl.max(trip); while j<mt`)?

The hoist computes the per-warp max trip ONCE (a scalar) and uses a scalar loop counter, so there is
no per-iteration cross-lane reduce, while per-warp early termination is preserved (each warp still
stops at its own busiest lane). It is provably equivalent (the body is already masked by j<trip).
A/B at the SOURCE level (oracle-identical) to decide if the MLIR pass is worth building:
  - k_while   : reduce-gated while (status quo).
  - k_for     : global-max bounded for (UNFAIR control -- forces every warp to the global max).
  - k_warpmax : the real transform (per-warp max hoisted once + scalar counter).
Scalar per-step payload (like MoE/rejection, where the tile loses); power-law trips.
"""

from __future__ import annotations

import statistics

import numpy as np
import torch
import triton
import triton.language as tl

N = 1 << 20
WARM, REP = 5, 15
BLOCK = 32
A_, B_, SEED = 1640531527, 1013904223, 12345


def build(seed=0):
    rng = np.random.default_rng(seed)
    raw = (rng.pareto(1.5, size=N) + 1).astype(np.int64)
    raw = np.clip(raw, 1, 256)
    trip = np.clip((raw * (8 * N / raw.sum())).round().astype(np.int64), 1, 256).astype(np.int32)
    return trip


def oracle(trip):
    out = np.zeros(N, dtype=np.int64)
    # acc = sum_{j<trip} ((i*A + j*B + SEED) & 0xFFFFFF)
    tr = trip.astype(np.int64)
    i = np.arange(N, dtype=np.int64)
    jmax = int(tr.max())
    for j in range(jmax):
        active = j < tr
        h = (i * A_ + j * B_ + SEED) & 0xFFFFFF
        out += np.where(active, h, 0)
    return out


@triton.jit
def k_while(trip, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    tr = tl.load(trip + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int64)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < tr).to(tl.int32)) > 0:  # reduce-gated lock-step
        active = valid & (j < tr)
        h = (i.to(tl.int64) * 1640531527 + j.to(tl.int64) * 1013904223 + 12345) & 0xFFFFFF
        acc = acc + tl.where(active, h, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


@triton.jit
def k_for(trip, out, n, maxtrip, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    tr = tl.load(trip + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int64)
    for j in range(maxtrip):  # bounded loop, NO per-iteration cross-lane reduce
        active = valid & (j < tr)
        h = (i.to(tl.int64) * 1640531527 + j * 1013904223 + 12345) & 0xFFFFFF
        acc = acc + tl.where(active, h, 0)
    tl.store(out + i, acc, mask=valid)


@triton.jit
def k_warpmax(trip, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    tr = tl.load(trip + i, mask=valid, other=0)
    mt = tl.max(tr)  # per-warp max, ONCE (the hoisted reduce)
    acc = tl.zeros((BLOCK,), tl.int64)
    j = 0
    while (
        j < mt
    ):  # SCALAR condition -> no per-iteration cross-lane reduce; keeps per-warp termination
        active = valid & (j < tr)
        h = (i.to(tl.int64) * 1640531527 + j * 1013904223 + 12345) & 0xFFFFFF
        acc = acc + tl.where(active, h, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def run(fn, d_trip, maxtrip=None):
    out = torch.zeros(N, dtype=torch.int64, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    if maxtrip is None:
        fn[(triton.cdiv(N, BLOCK),)](d_trip, out, N, BLOCK=BLOCK, num_warps=1)
    else:
        fn[(triton.cdiv(N, BLOCK),)](d_trip, out, N, maxtrip, BLOCK=BLOCK, num_warps=1)
    ev1.record()
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ev0.elapsed_time(ev1))


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    trip = build()
    d_trip = torch.as_tensor(trip, device="cuda")
    maxtrip = int(trip.max())
    ref = oracle(trip)
    ow, _ = run(k_while, d_trip)
    of, _ = run(k_for, d_trip, maxtrip)
    oc, _ = run(k_warpmax, d_trip)
    ok_w, ok_f, ok_c = (np.array_equal(ow, ref), np.array_equal(of, ref), np.array_equal(oc, ref))
    cv = trip.std() / trip.mean()
    print(f"oracle while={ok_w} for={ok_f} warpmax={ok_c} maxtrip={maxtrip} cv={cv:.2f}")
    if not (ok_w and ok_f and ok_c):
        print("ORACLE FAIL")
        return 1

    def med(fn, mt=None):
        for _ in range(WARM):
            run(fn, d_trip, mt)
        return statistics.median([run(fn, d_trip, mt)[1] for _ in range(REP)])

    tw = med(k_while)
    tc = med(k_warpmax)
    print(f"  while (reduce-gated, per-iter reduce): {tw * 1e3:8.1f} us")
    print(f"  warpmax (per-warp max hoisted once)  : {tc * 1e3:8.1f} us")
    g = tw / tc
    verdict = "WORTH a pass" if g > 1.08 else "hoisting the reduce does NOT help"
    print(f"  speedup warpmax/while = {g:.2f}x  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
