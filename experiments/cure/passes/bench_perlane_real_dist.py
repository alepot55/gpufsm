"""Does the per-lane retirement pass pay on a REAL trip distribution?

The six distributions the straggler law was fitted on are synthetic. This runs the same
lock-step kernel on the trip counts of a real input: the row lengths of the power-law CSR
matrix the SpMV witness uses (`landmark_spmv.build_csr`, seed 0), whose per-warp straggler
is 150 rows against a mean of 16.

The point is a controlled separation of the two halves of the scope condition. The SpMV
witness holds the distribution fixed and pays a DRAM gather per iteration, and the pass
buys ~1.0x there. Here the distribution is identical and the body is cheap, so whatever
speedup appears is attributable to the body cost alone. Uniform row lengths are the
matched control: same body, same kernel, no divergence.

Usage:  PYTHONPATH=<triton-src>/python python bench_perlane_real_dist.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

N_ROWS, BLOCK = 1 << 20, 32
K_UNIFORM = 16
WARMUP, SAMPLES = 3, 9


@triton.jit
def perlane_while(inp, out, n, BLOCK: tl.constexpr):
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


def row_lengths(kind: str, seed: int = 0) -> np.ndarray:
    """Exactly landmark_spmv.build_csr's nnz-per-row, so the distribution is the same one."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return np.full(N_ROWS, K_UNIFORM, dtype=np.int64)
    raw = np.clip((rng.pareto(1.5, size=N_ROWS) + 1).astype(np.int64), 1, 4096)
    scale = (K_UNIFORM * N_ROWS) / raw.sum()
    return np.clip((raw * scale).round().astype(np.int64), 1, 4096)


def run(d, ref, retire: bool):
    out = torch.zeros(N_ROWS, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(N_ROWS, BLOCK),)
    kw = {"num_warps": 1}
    if retire:
        kw["per_lane_loop_retirement"] = True
    h = perlane_while[grid](d, out, N_ROWS, BLOCK=BLOCK, **kw)
    ok = np.array_equal(out.cpu().numpy(), ref)
    for _ in range(WARMUP):
        perlane_while[grid](d, out, N_ROWS, BLOCK=BLOCK, **kw)
    ts = []
    for _ in range(SAMPLES):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        perlane_while[grid](d, out, N_ROWS, BLOCK=BLOCK, **kw)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return ok, statistics.median(ts) * 1e3, h.asm["ptx"].count("redux.sync")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    gpu = torch.cuda.get_device_name(0)
    rows = []
    print(
        f"{'distribution':>16}{'E[trip]':>10}{'E[warpmax]':>12}{'masked us':>11}"
        f"{'cured us':>10}{'speedup':>9}{'oracle':>8}"
    )
    for kind in ("uniform", "powerlaw"):
        trip = row_lengths(kind)
        warpmax = trip[: (N_ROWS // 32) * 32].reshape(-1, 32).max(axis=1).mean()
        d = torch.as_tensor(trip.astype(np.int32), device="cuda")
        ref = (trip * (trip - 1) // 2).astype(np.int32)
        ok0, t0, rdx0 = run(d, ref, retire=False)
        ok1, t1, rdx1 = run(d, ref, retire=True)
        ok = ok0 and ok1 and rdx0 >= 1 and rdx1 == 0
        print(
            f"{kind:>16}{trip.mean():10.2f}{warpmax:12.2f}{t0:11.1f}{t1:10.1f}"
            f"{t0 / t1:8.2f}x{'  OK' if ok else '  FAIL':>8}"
        )
        rows.append((kind, trip.mean(), warpmax, t0, t1, t0 / t1, "OK" if ok else "FAIL"))
    outp = Path("paper2/data/landmark/cure_real_dist.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write("# per-lane retirement on the SpMV row-length distributions, cheap body.\n")
        f.write("# Same distributions as landmark_spmv (build_csr seed 0); the body here is\n")
        f.write("# an accumulate, not a DRAM gather, so this isolates the body cost.\n")
        f.write("distribution,mean_trip,mean_warp_max,masked_us,cured_us,speedup,oracle,gpu\n")
        for k, mt, wm, t0, t1, sp, o in rows:
            f.write(f"{k},{mt:.2f},{wm:.2f},{t0:.1f},{t1:.1f},{sp:.2f},{o},{gpu}\n")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
