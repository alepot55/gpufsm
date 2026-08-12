"""R&D: is the built cure's speedup PREDICTABLE from intra-warp trip divergence?

Mechanism model. The masked lock-step loop `while tl.max(j < trip) > 0` makes EVERY lane of a
warp iterate until the warp's MAX trip (all 32 lanes execute warp-max iterations) AND pays a
cross-lane reduce (`tl.max`) every iteration. The cured per-lane-retirement loop lets each lane
stop at its OWN trip (work per warp = sum of trips = 32 x warp-mean) and branches on the per-lane
predicate (no per-iteration reduce). So the compute-work ratio the cure recovers is exactly

    D = E[warp-max] / E[trip]           (>= 1, = 1 iff no intra-warp divergence)

Falsifiable prediction: measured speedup masked/cured should track D once both times clear the
fixed launch/memory floor. We sweep controlled trip distributions, compute D from the DATA, and
measure both modes. Emits one CSV row per (dist, mode); a companion join computes speedup vs D.

Run per mode (the cure is compiled into the from-source Triton, gated by env):
  PYTHONPATH=$HOME/m3full_build/triton-src/python TRITON_ALWAYS_COMPILE=1 \
    GPUFSM_THREAD_REGION=off    .venv/bin/python experiments/cure/cure_predictive.py
  PYTHONPATH=$HOME/m3full_build/triton-src/python TRITON_ALWAYS_COMPILE=1 \
    GPUFSM_THREAD_REGION=retire .venv/bin/python experiments/cure/cure_predictive.py
"""

from __future__ import annotations

import csv
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

N, BLOCK = 1 << 20, 32
CSV = Path("paper2/data/landmark/cure_predictive_rtx4070.csv")
DISTS = ("constant", "lowvar", "uniform", "geometric", "pareto2", "pareto1.2")


@triton.jit
def _perlane_while(inp, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    trip = tl.load(inp + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < trip).to(tl.int32)) > 0:
        active = j < trip
        acc = acc + tl.where(active, j, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def make_trips(kind, rng):
    if kind == "constant":
        t = np.full(N, 64)
    elif kind == "lowvar":
        t = rng.integers(56, 73, size=N)  # ~uniform, narrow
    elif kind == "uniform":
        t = rng.integers(1, 257, size=N)  # wide uniform
    elif kind == "geometric":
        t = np.clip(rng.geometric(1 / 16, size=N), 1, 256)
    elif kind == "pareto2":
        raw = np.clip((rng.pareto(2.0, size=N) + 1), 1, 256)
        t = np.clip((raw * (16 * N / raw.sum())).round(), 1, 256)
    elif kind == "pareto1.2":
        raw = np.clip((rng.pareto(1.2, size=N) + 1), 1, 256)
        t = np.clip((raw * (16 * N / raw.sum())).round(), 1, 256)
    else:
        raise ValueError(kind)
    return t.astype(np.int32)


def warp_stats(trip):
    """Per-warp (32-lane) max mean, global trip mean, and divergence D."""
    w = trip[: (N // BLOCK) * BLOCK].reshape(-1, BLOCK)
    warpmax = float(w.max(axis=1).mean())
    mean = float(trip.mean())
    return warpmax, mean, warpmax / mean


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    rng = np.random.default_rng(0)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV.exists()
    rows = []
    for kind in DISTS:
        trip = make_trips(kind, rng)
        warpmax, mean, D = warp_stats(trip)
        d = torch.as_tensor(trip, device="cuda")
        ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)

        def run():
            o = torch.zeros(N, dtype=torch.int32, device="cuda")
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            _perlane_while[(triton.cdiv(N, BLOCK),)](d, o, N, BLOCK=BLOCK, num_warps=1)
            e1.record()
            torch.cuda.synchronize()
            return o.cpu().numpy(), float(e0.elapsed_time(e1))

        o, _ = run()
        ok = np.array_equal(o, ref)
        for _ in range(3):
            run()
        t = statistics.median([run()[1] for _ in range(9)]) * 1e3  # us
        rows.append((kind, mode, f"{warpmax:.2f}", f"{mean:.2f}", f"{D:.3f}", "OK" if ok else "FAIL", f"{t:.1f}"))
        print(f"dist={kind:10} D={D:6.2f} warpmax={warpmax:6.1f} mean={mean:5.1f} mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t:8.1f}us")

    with CSV.open("a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["dist", "mode", "warpmax_mean", "trip_mean", "D", "oracle", "time_us"])
        wr.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
