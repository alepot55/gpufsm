"""M9: dispersion on the load-bearing cure number. Re-runs the lock-step kernel
(masked vs per_lane_loop_retirement) over 5 trip-distribution seeds and reports
per-seed speedup + median/IQR. Reuses the kernel + harness from bench_perlane_retire."""
from __future__ import annotations

import statistics

import numpy as np
import torch
from experiments.cure.bench_perlane_retire import N, run


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    sp = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        trip = np.clip(rng.geometric(1 / 16, size=N), 1, 256).astype(np.int32)
        d = torch.as_tensor(trip, device="cuda")
        ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)
        ok0, t0, _, _ = run(d, ref, retire=False)
        ok1, t1, rdx1, _ = run(d, ref, retire=True)
        assert ok0 and ok1 and rdx1 == 0, f"seed {seed}: oracle/pass failure"
        sp.append(t0 / t1)
        print(f"seed={seed} masked={t0:7.1f}us cured={t1:7.1f}us speedup={t0 / t1:.2f}x")
    med = statistics.median(sp)
    q1, q3 = np.percentile(sp, [25, 75])
    print(f"speedup median={med:.2f}x IQR=[{q1:.2f},{q3:.2f}] range=[{min(sp):.2f},{max(sp):.2f}] n={len(sp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
