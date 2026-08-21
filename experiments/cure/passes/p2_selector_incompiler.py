"""The detect -> route -> lower loop, closed entirely inside libtriton.

The earlier selector (`p2_selector.py`) had to shell out: detection lived in a from-source
Triton carrying the research pass, and the lowering it routed to was an out-of-band nvcc
kernel. Those were two different toolchains, so the loop had a seam.

The pinned cure wheel carries BOTH halves, so the seam is gone:

  GPUFSM_THREAD_REGION=1   -> make_ttgir marks the lock-step loop `ttg.thread_region_candidate`
  per_lane_loop_retirement -> make_llir rewrites that loop's latch to the per-lane predicate

This script runs the whole decision in one process and one compiler: compile once to read the
detection attribute, route on it, recompile with retirement where detected, and leave the
negative control alone. Both kernels are oracle-gated in both modes.

Usage:  PYTHONPATH=<cure-tree>/python python p2_selector_incompiler.py
"""

from __future__ import annotations

import os
import statistics
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

os.environ["GPUFSM_THREAD_REGION"] = "1"

N, BLOCK = 1 << 20, 32
WARMUP, SAMPLES = 3, 9
CANDIDATE = "thread_region_candidate"


@triton.jit
def worklist_lockstep(trip, out, n, BLOCK: tl.constexpr):
    """Active-set shape: data-dependent trip count behind a tl.max latch."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    t = tl.load(trip + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < t).to(tl.int32)) > 0:
        acc = acc + tl.where(j < t, j, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


@triton.jit
def fixedtrip_control(trip, out, n, K: tl.constexpr, BLOCK: tl.constexpr):
    """Negative control: same body, same memory, but a compile-time trip count."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    t = tl.load(trip + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(K):
        acc = acc + tl.where(j < t, j, 0)
    tl.store(out + i, acc, mask=valid)


def _time(fn) -> float:
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(SAMPLES):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts) * 1e3


def run_lockstep(d, ref, retire: bool):
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(N, BLOCK),)
    kw = {"num_warps": 1}
    if retire:
        kw["per_lane_loop_retirement"] = True
    h = worklist_lockstep[grid](d, out, N, BLOCK=BLOCK, **kw)
    ok = np.array_equal(out.cpu().numpy(), ref)
    t = _time(lambda: worklist_lockstep[grid](d, out, N, BLOCK=BLOCK, **kw))
    return ok, t, h.asm["ttgir"], h.asm["ptx"]


def run_control(d, ref, retire: bool, K: int):
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(N, BLOCK),)
    kw = {"num_warps": 1}
    if retire:
        kw["per_lane_loop_retirement"] = True
    h = fixedtrip_control[grid](d, out, N, K=K, BLOCK=BLOCK, **kw)
    ok = np.array_equal(out.cpu().numpy(), ref)
    t = _time(lambda: fixedtrip_control[grid](d, out, N, K=K, BLOCK=BLOCK, **kw))
    return ok, t, h.asm["ttgir"], h.asm["ptx"]


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    gpu = torch.cuda.get_device_name(0)
    rng = np.random.default_rng(0)
    trip = np.clip(rng.geometric(1 / 24, size=N), 1, 512).astype(np.int32)
    d = torch.as_tensor(trip, device="cuda")
    ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)
    K = int(trip.max())
    warpmax = trip[: (N // 32) * 32].reshape(-1, 32).max(axis=1).mean()

    rows = []
    # --- lock-step kernel: detection should fire, so the selector routes it to retirement.
    ok0, t0, ttgir0, ptx0 = run_lockstep(d, ref, retire=False)
    detected = CANDIDATE in ttgir0
    ok1, t1, _, ptx1 = run_lockstep(d, ref, retire=True) if detected else (ok0, t0, "", ptx0)
    print(f"lock-step worklist : detected={detected} route={'retire' if detected else 'tile'}")
    print(f"  tile   {t0:8.1f}us oracle={'OK' if ok0 else 'FAIL'} redux={ptx0.count('redux.sync')}")
    print(
        f"  routed {t1:8.1f}us oracle={'OK' if ok1 else 'FAIL'} redux={ptx1.count('redux.sync')}"
        f" warp.sync={ptx1.count('bar.warp.sync')}"
    )
    print(f"  speedup {t0 / t1:.2f}x   E[warp-max]={warpmax:.1f}")
    rows.append(
        (
            "lockstep_worklist",
            int(detected),
            "retire" if detected else "tile",
            t0,
            t1,
            t0 / t1,
            "OK" if ok0 and ok1 else "FAIL",
        )
    )

    # --- negative control: detection must NOT fire, and the tile path must be kept.
    ok2, t2, ttgir2, ptx2 = run_control(d, ref, retire=False, K=K)
    detected2 = CANDIDATE in ttgir2
    print(f"fixed-trip control : detected={detected2} route={'retire' if detected2 else 'tile'}")
    print(f"  tile   {t2:8.1f}us oracle={'OK' if ok2 else 'FAIL'} redux={ptx2.count('redux.sync')}")
    rows.append(
        (
            "fixedtrip_control",
            int(detected2),
            "retire" if detected2 else "tile",
            t2,
            float("nan"),
            float("nan"),
            "OK" if ok2 else "FAIL",
        )
    )

    outp = Path("selector_incompiler.csv")
    with outp.open("w") as f:
        f.write(
            "# detect -> route -> lower, one process and one compiler (the pinned cure wheel).\n"
        )
        f.write(
            "# detection = ttg.thread_region_candidate in the TTGIR under GPUFSM_THREAD_REGION=1;\n"
        )
        f.write("# routing = recompile with per_lane_loop_retirement where detected.\n")
        f.write(f"# E[warp-max]={warpmax:.1f} over a geometric trip law, N={N}, BLOCK={BLOCK}.\n")
        f.write("kernel,detected,routed_to,tile_us,routed_us,speedup,oracle,gpu\n")
        for k, det, route, a, b, sp, o in rows:
            f.write(f"{k},{det},{route},{a:.1f},{b:.1f},{sp:.2f},{o},{gpu}\n")
    print(f"wrote {outp}")
    assert ok0 and ok1 and ok2, "oracle mismatch"
    assert detected and not detected2, "detection did not discriminate"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
