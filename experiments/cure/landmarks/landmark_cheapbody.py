"""Real kernels that meet BOTH halves of the scope condition: divergent trips, cheap body.

The built pass buys ~1.0x on power-law SpMV and MoE. Those inputs are not short of
stragglers (SpMV's per-warp straggler is 150 rows against a mean of 16); what defeats the
pass is that a DRAM gather per iteration dwarfs the control it sits beside. The condition
for the pass to pay is therefore a divergent trip count OVER A CHEAP BODY, and until now the
only kernels satisfying both halves in this paper were synthetic.

These two are not synthetic in shape:

  BINARY GCD (Stein's algorithm) -- the subtract-and-shift loop used in crypto libraries.
  Trip count depends on the operand pair; the body is shifts, a compare and a subtract, with
  no memory traffic at all.

  COLLATZ step count -- the canonical irregular-trip integer kernel, body is two integer
  operations. Included as the extreme of the divergence axis.

Both carry the tl.max lock-step latch the pass targets, both are checked against a numpy
oracle, and both are run with and without per-lane retirement from the same pinned wheel.

Usage:  PYTHONPATH=<cure-tree>/python python landmark_cheapbody.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

N, BLOCK = 1 << 20, 32
WARMUP, SAMPLES = 3, 9


@triton.jit
def binary_gcd(a_ptr, b_ptr, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    u = tl.load(a_ptr + i, mask=valid, other=1)
    v = tl.load(b_ptr + i, mask=valid, other=1)
    while tl.max((u != v).to(tl.int32)) > 0:
        live = u != v
        u_even = (u & 1) == 0
        v_even = (v & 1) == 0
        u_big = u > v
        # u even -> halve u; else v even -> halve v; else subtract the smaller and halve.
        nu = tl.where(u_even, u >> 1, tl.where(v_even, u, tl.where(u_big, (u - v) >> 1, u)))
        nv = tl.where(u_even, v, tl.where(v_even, v >> 1, tl.where(u_big, v, (v - u) >> 1)))
        u = tl.where(live, nu, u)
        v = tl.where(live, nv, v)
    tl.store(out + i, u, mask=valid)


@triton.jit
def collatz_steps(seed, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    x = tl.load(seed + i, mask=valid, other=1)
    steps = tl.zeros((BLOCK,), tl.int32)
    while tl.max((x != 1).to(tl.int32)) > 0:
        live = x != 1
        nxt = tl.where((x & 1) == 0, x >> 1, 3 * x + 1)
        x = tl.where(live, nxt, x)
        steps = steps + tl.where(live, 1, 0)
    tl.store(out + i, steps, mask=valid)


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


def collatz_oracle(seed: np.ndarray) -> np.ndarray:
    x = seed.astype(np.int64).copy()
    steps = np.zeros_like(x)
    live = x != 1
    while live.any():
        even = live & ((x & 1) == 0)
        odd = live & ~even
        x[even] >>= 1
        x[odd] = 3 * x[odd] + 1
        steps[live] += 1
        live = x != 1
    return steps.astype(np.int32)


def gcd_trips(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Iteration count of the same subtract-and-shift loop, for the straggler statistic."""
    u, v = a.astype(np.int64).copy(), b.astype(np.int64).copy()
    t = np.zeros_like(u)
    live = u != v
    while live.any():
        ue, ve = (u & 1) == 0, (v & 1) == 0
        big = u > v
        nu = np.where(ue, u >> 1, np.where(ve, u, np.where(big, (u - v) >> 1, u)))
        nv = np.where(ue, v, np.where(ve, v >> 1, np.where(big, v, (v - u) >> 1)))
        u = np.where(live, nu, u)
        v = np.where(live, nv, v)
        t[live] += 1
        live = u != v
    return t


def bench(name, launch, ref, trips, rows, gpu):
    warpmax = trips[: (N // 32) * 32].reshape(-1, 32).max(axis=1).mean()
    res = {}
    for retire in (False, True):
        out = torch.zeros(N, dtype=torch.int32, device="cuda")
        h = launch(out, retire)
        ok = np.array_equal(out.cpu().numpy(), ref)
        t = _time(lambda o=out, r=retire: launch(o, r))
        res[retire] = (ok, t, h.asm["ptx"].count("redux.sync"))
    (ok0, t0, r0), (ok1, t1, r1) = res[False], res[True]
    fired = r0 >= 1 and r1 == 0
    print(
        f"{name:>12}{trips.mean():9.1f}{warpmax:11.1f}{t0:11.1f}{t1:10.1f}{t0 / t1:8.2f}x"
        f"{'  OK' if ok0 and ok1 else '  FAIL':>8}{'  yes' if fired else '  NO':>6}"
    )
    rows.append(
        (
            name,
            trips.mean(),
            warpmax,
            t0,
            t1,
            t0 / t1,
            "OK" if ok0 and ok1 else "FAIL",
            int(fired),
            gpu,
        )
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    gpu = torch.cuda.get_device_name(0)
    rng = np.random.default_rng(0)
    rows = []
    print(
        f"{'kernel':>12}{'E[trip]':>9}{'E[warpmax]':>11}{'masked us':>11}{'cured us':>10}"
        f"{'speedup':>9}{'oracle':>8}{'fired':>6}"
    )

    a = (rng.integers(1, 1 << 30, size=N) | 1).astype(np.int32)
    b = (rng.integers(1, 1 << 30, size=N) | 1).astype(np.int32)
    da, db = torch.as_tensor(a, device="cuda"), torch.as_tensor(b, device="cuda")
    ref_gcd = np.gcd(a, b).astype(np.int32)
    grid = (triton.cdiv(N, BLOCK),)

    def launch_gcd(out, retire):
        kw = {"num_warps": 1}
        if retire:
            kw["per_lane_loop_retirement"] = True
        return binary_gcd[grid](da, db, out, N, BLOCK=BLOCK, **kw)

    bench("binary_gcd", launch_gcd, ref_gcd, gcd_trips(a, b), rows, gpu)

    seed = rng.integers(1, 1 << 16, size=N).astype(np.int32)  # 2^20 overflows int32
    ds = torch.as_tensor(seed, device="cuda")
    ref_col = collatz_oracle(seed)

    def launch_col(out, retire):
        kw = {"num_warps": 1}
        if retire:
            kw["per_lane_loop_retirement"] = True
        return collatz_steps[grid](ds, out, N, BLOCK=BLOCK, **kw)

    bench("collatz", launch_col, ref_col, ref_col.astype(np.int64), rows, gpu)

    outp = Path("cure_cheapbody.csv")
    with outp.open("w") as f:
        f.write("# Real-shaped kernels with divergent trips AND a cheap body: the case the\n")
        f.write("# scope condition predicts the pass should pay on. Oracle-gated in both modes;\n")
        f.write("# 'fired' means PTX redux.sync went 1 -> 0, i.e. the pass actually rewrote.\n")
        f.write("kernel,mean_trip,mean_warp_max,masked_us,cured_us,speedup,oracle,fired,gpu\n")
        for k, mt, wm, t0, t1, sp, o, fi, g in rows:
            f.write(f"{k},{mt:.2f},{wm:.2f},{t0:.1f},{t1:.1f},{sp:.2f},{o},{fi},{g}\n")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
