"""LANDMARK P1 — generality witness #3: SpMV (CSR) = THE NEGATIVE CONTROL (the decisive falsifier).

The sharp test: regret tracks CONTROL-FLOW DIVERGENCE, not memory irregularity.
SpMV's per-row work is an irregular GATHER (x[colidx[j]]) + multiply-add reduce -- irregular MEMORY
access, but the only per-element control is the row-length loop. We run TWO matrices, same kernels:
  - UNIFORM nnz/row  -> irregular access, NO control divergence.
  - POWER-LAW nnz/row -> control divergence (variable trip).
HONEST RESULT (predicted uniform~1x; FALSIFIED): uniform is ~1.9x, a BASELINE tile-lowering
overhead (occupancy/register/masking; tile DRAM 29% vs thread 50%, occ 50% vs 94%; NOT
num_warps-fixable). The within-workload contrast isolates divergence: power-law adds an increment
(2.2x@32 -> 5.8x@256). So regret = tile-lowering baseline + divergence increment; the clean ~1x
negative control is DENSE-REGULAR work (Triton ~= cuBLAS), not irregular SpMV.

  TILE (Triton): one row/lane, lock-step over the longest row in the warp; x[colidx] gather.
  THREAD (CUDA): one row/thread, independent.
  ORACLE (numpy segmented sum, float32, same order as the kernels).
Oracle-gated (allclose, float32 reduction). x is DRAM-resident (gathers pay memory latency).
Writes paper2/data/landmark/spmv_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/landmark_spmv.py
        .venv/bin/python experiments/cure/landmark_spmv.py profile <uniform|powerlaw> <tile|thread>
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

N_ROWS = 1 << 20  # 1M rows
NCOLS = 1 << 22  # x has 4M float32 = 16 MB -> DRAM-resident (gather pays memory latency)
K_UNIFORM = 16  # uniform nnz/row
WARMUP, SAMPLES = 3, 9
import os  # noqa: E402

BLOCK = int(
    os.environ.get("SPMV_BLOCK", "32")
)  # rows/program; num_warps=BLOCK/32 to match occupancy
NUM_WARPS = max(1, BLOCK // 32)


def build_csr(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        nnz_per_row = np.full(N_ROWS, K_UNIFORM, dtype=np.int64)
    else:  # power-law (zipf-ish), same TOTAL nnz as uniform for a fair memory comparison
        raw = (rng.pareto(1.5, size=N_ROWS) + 1).astype(np.int64)
        raw = np.clip(raw, 1, 4096)
        scale = (K_UNIFORM * N_ROWS) / raw.sum()
        nnz_per_row = np.clip((raw * scale).round().astype(np.int64), 1, 4096)
    rowptr = np.zeros(N_ROWS + 1, dtype=np.int32)
    rowptr[1:] = np.cumsum(nnz_per_row)
    total = int(rowptr[-1])
    colidx = rng.integers(0, NCOLS, size=total, dtype=np.int32)
    values = rng.standard_normal(total).astype(np.float32)
    x = rng.standard_normal(NCOLS).astype(np.float32)
    return rowptr, colidx, values, x, nnz_per_row


def oracle(rowptr, colidx, values, x):
    prod = values * x[colidx]  # float32
    return np.add.reduceat(prod, rowptr[:-1].astype(np.int64)).astype(np.float32)


@triton.jit
def _spmv_tile(rowptr, colidx, values, x, n_rows, y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    valid = r < n_rows
    lo = tl.load(rowptr + r, mask=valid, other=0)
    hi = tl.load(rowptr + r + 1, mask=valid, other=0)
    nnz = hi - lo
    acc = tl.zeros((BLOCK,), tl.float32)
    k = tl.zeros((BLOCK,), tl.int32)
    while tl.max((k < nnz).to(tl.int32)) > 0:  # lock-step to the longest row in the warp
        active = valid & (k < nnz)
        idx = lo + k
        col = tl.load(colidx + idx, mask=active, other=0)
        val = tl.load(values + idx, mask=active, other=0.0)
        xv = tl.load(x + col, mask=active, other=0.0)  # irregular gather
        acc = acc + tl.where(active, val * xv, 0.0)
        k = k + 1
    tl.store(y + r, acc, mask=valid)


def _compile_thread():
    src = """
extern "C" __global__ void spmv_kernel(
    const int* rowptr, const int* colidx, const float* values, const float* x,
    int n_rows, float* y) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;  // one thread == one row
  if (r >= n_rows) return;
  float acc = 0.0f;
  for (int k = rowptr[r]; k < rowptr[r + 1]; k++) acc += values[k] * x[colidx[k]];
  y[r] = acc;
}
extern "C" float sp_launch(const int* rowptr, const int* colidx, const float* values,
                           const float* x, int n_rows, float* y) {
  int th = 256, bl = (n_rows + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  spmv_kernel<<<bl, th>>>(rowptr, colidx, values, x, n_rows, y);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}
"""
    cache = Path.home() / ".cache" / "landmark_spmv"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="sp_", dir=str(cache)))
    cu, so = d / "sp.cu", d / "sp.so"
    cu.write_text(src)
    nvcc = "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.sp_launch.restype = ctypes.c_float
    lib.sp_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
    return lib


_THR = None


def to_dev(rowptr, colidx, values, x):
    d = torch.device("cuda")
    return (
        torch.as_tensor(rowptr, device=d),
        torch.as_tensor(colidx, device=d),
        torch.as_tensor(values, device=d),
        torch.as_tensor(x, device=d),
    )


def run_tile(g):
    rp, ci, vl, x = g
    y = torch.zeros(N_ROWS, dtype=torch.float32, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _spmv_tile[(triton.cdiv(N_ROWS, BLOCK),)](
        rp, ci, vl, x, N_ROWS, y, BLOCK=BLOCK, num_warps=NUM_WARPS
    )
    ev1.record()
    torch.cuda.synchronize()
    return y.cpu().numpy(), float(ev0.elapsed_time(ev1))


def run_thread(g):
    global _THR
    if _THR is None:
        _THR = _compile_thread()
    rp, ci, vl, x = g
    y = torch.zeros(N_ROWS, dtype=torch.float32, device="cuda")
    ms = _THR.sp_launch(
        rp.data_ptr(), ci.data_ptr(), vl.data_ptr(), x.data_ptr(), N_ROWS, y.data_ptr()
    )
    torch.cuda.synchronize()
    return y.cpu().numpy(), float(ms)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        rp, ci, vl, x, _ = build_csr(sys.argv[2], 0)
        g = to_dev(rp, ci, vl, x)
        (run_tile if sys.argv[3] == "tile" else run_thread)(g)
        return 0

    print("SpMV negative control: regret vs CONTROL-FLOW DIVERGENCE (same memory irregularity)")
    print(
        f"{'matrix':>10}{'nnz_cv':>9}{'tile_Gflop':>12}{'thread_Gflop':>13}{'regret':>9}{'oracle':>8}"
    )
    rows = []
    for kind in ("uniform", "powerlaw"):
        rp, ci, vl, x, nnz = build_csr(kind, 0)
        g = to_dev(rp, ci, vl, x)
        ref = oracle(rp, ci, vl, x)
        ok = all(np.allclose(fn(g)[0], ref, rtol=1e-3, atol=1e-2) for fn in (run_tile, run_thread))
        if not ok:
            print(f"{kind:>10}  ORACLE FAIL")
            continue
        # coefficient of variation of row length = the divergence predictor
        cv = float(nnz.std() / nnz.mean())
        total_flop = 2 * int(rp[-1])  # mul + add per nnz

        def med(fn, g=g):
            for _ in range(WARMUP):
                fn(g)
            return statistics.median([fn(g)[1] for _ in range(SAMPLES)])

        tflop = total_flop / (med(run_tile) * 1e-3) / 1e9
        hflop = total_flop / (med(run_thread) * 1e-3) / 1e9
        regret = hflop / tflop
        print(f"{kind:>10}{cv:9.2f}{tflop:12.1f}{hflop:13.1f}{regret:9.2f}{'  ok':>8}")
        rows.append((kind, round(cv, 3), round(tflop, 1), round(hflop, 1), round(regret, 3)))
    if rows:
        print("\n=> uniform (no control divergence) regret ~1x; power-law (divergent) regret > 1x.")
        print()
        outp = Path("paper2/data/landmark/spmv_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("matrix,nnz_cv,tile_gflop_s,thread_gflop_s,regret,gpu\n")
            for kind, cv, tf, hf, rg in rows:
                f.write(f"{kind},{cv},{tf},{hf},{rg},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
