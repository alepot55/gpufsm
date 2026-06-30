"""F3 — verify the in-compiler reduce-hoist pass end-to-end (oracle + perf).

Runs the canonical lock-step kernel (`while tl.max(j<trip)>0`, scalar accumulate) through the
from-source Triton 3.8 that carries our `tritongpu-thread-region` pass, in three modes:
  off   : no pass (tile baseline)
  1     : detection only (tags the candidate; perf no-op)
  hoist : the REWRITE -- hoist reduce_max(trip) once + a scalar loop counter (no per-iteration
          cross-lane reduce, per-warp termination preserved, body unchanged/masked).
Oracle: acc[i] = sum_{j<trip[i]} j = trip[i]*(trip[i]-1)/2 (exact). The hoist must stay bit-exact
and be measurably faster. Cache-busted (TRITON_ALWAYS_COMPILE=1) so each mode recompiles.

Usage (run per mode):
  PYTHONPATH=$HOME/m3full_build/triton-src/python TRITON_ALWAYS_COMPILE=1 \
    GPUFSM_THREAD_REGION=hoist .venv/bin/python experiments/cure/f3_hoist_verify.py
"""

from __future__ import annotations

import os
import statistics

import numpy as np
import torch
import triton
import triton.language as tl


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


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    n = 1 << 20
    rng = np.random.default_rng(0)
    raw = np.clip((rng.pareto(1.5, size=n) + 1).astype(np.int64), 1, 256)
    trip = np.clip((raw * (16 * n / raw.sum())).round().astype(np.int64), 1, 256).astype(np.int32)
    d_trip = torch.as_tensor(trip, device="cuda")
    ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)

    def run():
        out = torch.zeros(n, dtype=torch.int32, device="cuda")
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        _perlane_while[(triton.cdiv(n, 32),)](d_trip, out, n, BLOCK=32, num_warps=1)
        ev1.record()
        torch.cuda.synchronize()
        return out.cpu().numpy(), float(ev0.elapsed_time(ev1))

    o, _ = run()
    ok = np.array_equal(o, ref)
    for _ in range(5):
        run()
    t = statistics.median([run()[1] for _ in range(15)])
    print(f"mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t * 1e3:8.1f}us")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
