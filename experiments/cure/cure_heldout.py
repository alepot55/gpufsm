"""R&D: OUT-OF-SAMPLE validation of the straggler law.

Last experiment fit, on six distributions, masked_time = a + b*E[warp-max] (R^2=0.998) and a flat
cured floor. The strongest test of a law is a held-out prediction: FIX (a, b, floor) from the
train CSV, then on FOUR NEW distributions never used in the fit -- including an adversarial
single-straggler warp (31 lanes trip=2, 1 lane trip=256) -- predict each speedup *a priori* from
E[warp-max] alone and compare to measurement. If the law is real it predicts unseen shapes; if it
is an overfit it will not.

Run per mode (cure gated by env), then join off vs retire:
  PYTHONPATH=$HOME/m3full_build/triton-src/python TRITON_ALWAYS_COMPILE=1 \
    GPUFSM_THREAD_REGION=off    .venv/bin/python experiments/cure/cure_heldout.py
  PYTHONPATH=$HOME/m3full_build/triton-src/python TRITON_ALWAYS_COMPILE=1 \
    GPUFSM_THREAD_REGION=retire .venv/bin/python experiments/cure/cure_heldout.py
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
CSV = Path("paper2/data/landmark/cure_heldout_rtx4070.csv")
HELDOUT = ("bimodal", "lognormal", "single_straggler", "spike")


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
    if kind == "bimodal":
        pick = rng.random(N) < 0.5
        t = np.where(pick, rng.integers(4, 13, N), rng.integers(180, 221, N))
    elif kind == "lognormal":
        t = np.clip(np.round(rng.lognormal(2.5, 0.9, N)), 1, 256)
    elif kind == "single_straggler":
        # exactly one lane per 32-lane warp is a 256-trip straggler, the rest trip=2
        t = np.full(N, 2)
        t[:: BLOCK] = 256
    elif kind == "spike":
        t = np.where(rng.random(N) < 0.05, 250, rng.integers(2, 7, N))
    else:
        raise ValueError(kind)
    return t.astype(np.int32)


def warp_stats(trip):
    w = trip[: (N // BLOCK) * BLOCK].reshape(-1, BLOCK)
    warpmax = float(w.max(axis=1).mean())
    mean = float(trip.mean())
    return warpmax, mean, warpmax / mean


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    rng = np.random.default_rng(7)  # different seed than the train sweep
    CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV.exists()
    rows = []
    for kind in HELDOUT:
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
        t = statistics.median([run()[1] for _ in range(9)]) * 1e3
        rows.append((kind, mode, f"{warpmax:.2f}", f"{mean:.2f}", f"{D:.3f}", "OK" if ok else "FAIL", f"{t:.1f}"))
        print(f"dist={kind:16} D={D:7.2f} warpmax={warpmax:6.1f} mean={mean:6.1f} mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t:8.1f}us")

    with CSV.open("a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["dist", "mode", "warpmax_mean", "trip_mean", "D", "oracle", "time_us"])
        wr.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
