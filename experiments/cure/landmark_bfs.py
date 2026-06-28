"""LANDMARK P1 — generality witness #4: graph pointer-chase (bounded random walk).

THE canonical dependent-load / MLP-bound pattern on a REAL irregular graph (GAP/SuiteSparse-class),
and a sharp re-test of the MLP hypothesis the hash-probe witness diluted. Each of N walkers takes a
FIXED number of steps (no trip divergence) but each step's gather address DEPENDS on the previous
load (pointer chase): v -> neighbor[ rowptr[v] + h(v,step) % deg(v) ]. So this isolates the pure
dependent-load face: irregular MEMORY (variable degree, scattered colidx), data-dependent addresses,
but ZERO control-flow trip divergence. If the lock-step tile's gather already gives full intra-warp
MLP (the hash-probe finding), regret stays modest (~baseline); if pointer-chasing starves the tile,
regret is large. Either outcome is a clean, oracle-gated data point.

  TILE (Triton, lane-packed): one walker/lane, BLOCK lanes lockstep over the fixed step count.
  THREAD (CUDA, one thread/walker): independent dependent-load chains -> intra-warp latency hiding.
  ORACLE (numpy, exact integer arithmetic): the final vertex per walker.
The step hash uses only multiply+add+mask so draws are identical across numpy/CUDA/Triton 64-bit
two's-complement wrap. Oracle-gated (exact). Writes paper2/data/landmark/bfs_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/landmark_bfs.py
        .venv/bin/python experiments/cure/landmark_bfs.py profile <tile|thread>
"""

from __future__ import annotations

import ctypes
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

# odd mixing constants, both < 2^31 so int32 literals never overflow in Triton; products are
# forced to int64 via .to(tl.int64) on the tensors (matches CUDA long long / numpy int64 wrap).
A, B, SEED = 1640531527, 1013904223, 12345
M24 = 0xFFFFFF
NV = 1 << 20  # 1M vertices
AVG_DEG = 16  # power-law degrees, same total edges as uniform AVG_DEG
N_WALK = 1 << 20  # 1M independent walkers
STEPS = 64  # fixed steps -> NO trip divergence (isolates dependent-load MLP)
WARMUP, SAMPLES = 3, 9
BLOCK = int(os.environ.get("BFS_BLOCK", "32"))
NUM_WARPS = max(1, BLOCK // 32)


def build_graph(seed: int):
    """Power-law-degree CSR graph; colidx scattered (irregular gather)."""
    rng = np.random.default_rng(seed)
    raw = (rng.pareto(1.5, size=NV) + 1).astype(np.int64)
    raw = np.clip(raw, 1, 4096)
    scale = (AVG_DEG * NV) / raw.sum()
    deg = np.clip((raw * scale).round().astype(np.int64), 1, 4096)
    rowptr = np.zeros(NV + 1, dtype=np.int32)
    rowptr[1:] = np.cumsum(deg)
    total = int(rowptr[-1])
    colidx = rng.integers(0, NV, size=total, dtype=np.int32)
    return rowptr, colidx, deg


def oracle(rowptr, colidx):
    """Exact final vertex per walker, vectorized 64-bit-wrap arithmetic (matches kernels).

    numpy int64 multiply/add wrap two's-complement = CUDA `long long`; `& M24` extracts the same
    low 24 bits as CUDA `(unsigned long long)m & 0xFFFFFF`, so no explicit 64-bit mask is needed.
    """
    rp = rowptr.astype(np.int64)
    ci = colidx.astype(np.int64)
    v = np.arange(N_WALK, dtype=np.int64) % NV
    for step in range(STEPS):
        lo = rp[v]
        deg = rp[v + 1] - lo
        h = (v * A + step * B + SEED) & M24
        nb = lo + (h % deg)
        v = ci[nb]
    return v


@triton.jit
def _walk_tile(rowptr, colidx, n, out, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    w = pid * BLOCK + tl.arange(0, BLOCK)
    valid = w < n
    v = w % (1 << 20)  # NV, start vertex
    for step in range(STEPS):  # constexpr range, unrolled
        lo = tl.load(rowptr + v, mask=valid, other=0)
        hi = tl.load(rowptr + v + 1, mask=valid, other=1)
        deg = hi - lo
        h = (v.to(tl.int64) * 1640531527 + step * 1013904223 + 12345) & 0xFFFFFF
        nb = lo + (h % deg.to(tl.int64)).to(tl.int32)
        v = tl.load(colidx + nb, mask=valid, other=0)
    tl.store(out + w, v, mask=valid)


def _compile_thread():
    src = """
extern "C" __global__ void walk_kernel(
    const int* rowptr, const int* colidx, int n, int steps, int* out) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;  // one thread == one walker
  if (w >= n) return;
  int v = w % (1 << 20);
  for (int step = 0; step < steps; step++) {
    int lo = rowptr[v];
    int deg = rowptr[v + 1] - lo;
    long long m = (long long)v * 1640531527LL + (long long)step * 1013904223LL + 12345LL;
    unsigned long long h = (unsigned long long)m & 0xFFFFFFULL;
    int nb = lo + (int)(h % (unsigned long long)deg);
    v = colidx[nb];  // dependent load (pointer chase)
  }
  out[w] = v;
}
extern "C" float wk_launch(const int* rowptr, const int* colidx, int n, int steps, int* out) {
  int th = 256, bl = (n + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  walk_kernel<<<bl, th>>>(rowptr, colidx, n, steps, out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}
"""
    cache = Path.home() / ".cache" / "landmark_bfs"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="wk_", dir=str(cache)))
    cu, so = d / "wk.cu", d / "wk.so"
    cu.write_text(src)
    nvcc = "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.wk_launch.restype = ctypes.c_float
    lib.wk_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


_THR = None


def run_tile(rp, ci):
    out = torch.zeros(N_WALK, dtype=torch.int32, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _walk_tile[(triton.cdiv(N_WALK, BLOCK),)](
        rp, ci, N_WALK, out, STEPS=STEPS, BLOCK=BLOCK, num_warps=NUM_WARPS
    )
    ev1.record()
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ev0.elapsed_time(ev1))


def run_thread(rp, ci):
    global _THR
    if _THR is None:
        _THR = _compile_thread()
    out = torch.zeros(N_WALK, dtype=torch.int32, device="cuda")
    ms = _THR.wk_launch(rp.data_ptr(), ci.data_ptr(), N_WALK, STEPS, out.data_ptr())
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ms)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    rp_h, ci_h, deg = build_graph(0)
    d = torch.device("cuda")
    rp = torch.as_tensor(rp_h, device=d)
    ci = torch.as_tensor(ci_h, device=d)
    if len(sys.argv) >= 3 and sys.argv[1] == "profile":
        (run_tile if sys.argv[2] == "tile" else run_thread)(rp, ci)
        return 0

    ref = oracle(rp_h, ci_h)
    for name, fn in (("tile", run_tile), ("thread", run_thread)):
        got, _ = fn(rp, ci)
        if not np.array_equal(got, ref):
            print(f"ORACLE FAIL {name}: {int((got != ref).sum())}/{N_WALK} mismatch")
            return 1
    cv = float(deg.std() / deg.mean())
    print(
        f"graph pointer-chase ({N_WALK} walkers x {STEPS} steps, deg cv={cv:.2f}, "
        f"dependent-load MLP, NO trip divergence):"
    )

    def med(fn):
        for _ in range(WARMUP):
            fn(rp, ci)
        return statistics.median([fn(rp, ci)[1] for _ in range(SAMPLES)])

    tile_ms, thr_ms = med(run_tile), med(run_thread)
    tile_me = N_WALK * STEPS / (tile_ms * 1e-3) / 1e6
    thr_me = N_WALK * STEPS / (thr_ms * 1e-3) / 1e6
    regret = thr_me / tile_me
    print(f"  TILE (Triton)  {tile_me:8.1f} Mstep/s")
    print(f"  THREAD (CUDA)  {thr_me:8.1f} Mstep/s")
    print(f"  regret (thread/tile) = {regret:.2f}x  (pure dependent-load MLP, real graph)")
    outp = Path("paper2/data/landmark/bfs_rtx4070.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write("workload,n_walk,steps,deg_cv,tile_mstep_s,thread_mstep_s,regret,gpu\n")
        f.write(
            f"pointer_chase,{N_WALK},{STEPS},{cv:.3f},{tile_me:.2f},{thr_me:.2f},{regret:.3f},RTX4070\n"
        )
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
