"""LANDMARK P1 — generality witness #2: rejection sampling (PURE control-flow divergence).

Each element loops a data-dependent # of times: at iter j it draws a deterministic
value h(i,j) and accepts when h(i,j) < thresh[i]; the per-element trip count is data-dependent and
divergent, with ~no memory traffic (thresh read once, then pure integer compute). ISOLATES
the masked-lane-waste sub-mechanism (the hash-probe finding): the lock-step tile must run to the
busiest lane's trip count while early-accepting lanes sit idle (masked), whereas independent threads
retire as they accept.

  TILE (Triton, lane-packed): while tl.max(active) -> lockstep to busiest lane.
  THREAD (CUDA, one thread/element): independent trip counts.
  ORACLE (Python, exact 64-bit-wrap arithmetic): the accept iteration per element.
The hash uses only multiply+add+mask (no shifts) so low-24-bit draws are identical across
numpy/CUDA/Triton 64-bit two's-complement wrap. Oracle-gated. Predict: regret like hash-probe
(masked-waste). Writes paper2/data/landmark/rejection_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/landmark_rejection.py
        .venv/bin/python experiments/cure/landmark_rejection.py profile <tile|thread>
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

A, B, C, E = 2654435761, 2246822519, 3266489917, 668265263  # odd mixing constants
SEED = 12345
M24 = 0xFFFFFF
N = 1 << 20  # 1M elements
MAXITER = 256
WARMUP, SAMPLES = 3, 9
BLOCK = 32


def make_thresh(seed: int):
    rng = np.random.default_rng(seed)
    # accept prob in [0.125, 0.375] -> mean trip ~2.7-8, divergent
    return ((1 << 21) + rng.integers(0, 1 << 22, size=N)).astype(np.int64)


def oracle(thresh):
    """Exact reference trip count (accept iter), pure-Python 64-bit-wrap arithmetic."""
    out = np.empty(thresh.size, dtype=np.int64)
    mask = (1 << 64) - 1
    for i in range(thresh.size):
        th = int(thresh[i])
        acc = MAXITER
        for j in range(MAXITER):
            t = (i * A + j * B + SEED) & mask
            t = (t * C) & mask
            t = (t * E) & mask
            if (t & M24) < th:
                acc = j
                break
        out[i] = acc
    return out


@triton.jit
def _rejection_tile(
    thresh,
    n,
    out,
    A: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    E: tl.constexpr,
    SEED: tl.constexpr,
    M24: tl.constexpr,
    MAXITER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    th = tl.load(thresh + i, mask=valid, other=0)
    acc = tl.full((BLOCK,), MAXITER, tl.int64)
    done = ~valid
    j = tl.zeros((BLOCK,), tl.int64)
    while tl.max((~done).to(tl.int32)) > 0:
        active = ~done
        t = i.to(tl.int64) * A + j * B + SEED
        t = t * C
        t = t * E
        draw = t & M24
        hit = active & (draw < th) & (j < MAXITER)
        acc = tl.where(hit, j, acc)
        done = done | hit | (j >= MAXITER - 1)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def _compile_thread():
    src = """
extern "C" __global__ void rejection_kernel(
    const long long* thresh, int n, long long* out,
    long long A, long long B, long long C, long long E, long long SEED, int MAXITER) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // one thread == one element
  if (i >= n) return;
  long long th = thresh[i];
  long long acc = MAXITER;
  for (int j = 0; j < MAXITER; j++) {
    unsigned long long t = (unsigned long long)((long long)i * A + (long long)j * B + SEED);
    t = t * (unsigned long long)C;
    t = t * (unsigned long long)E;
    if ((long long)(t & 0xFFFFFFULL) < th) { acc = j; break; }
  }
  out[i] = acc;
}
extern "C" float rj_launch(const long long* thresh, int n, long long* out,
    long long A, long long B, long long C, long long E, long long SEED, int MAXITER) {
  int th = 256, bl = (n + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  rejection_kernel<<<bl, th>>>(thresh, n, out, A, B, C, E, SEED, MAXITER);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}
"""
    cache = Path.home() / ".cache" / "landmark_rejection"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="rj_", dir=str(cache)))
    cu, so = d / "rj.cu", d / "rj.so"
    cu.write_text(src)
    nvcc = "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.rj_launch.restype = ctypes.c_float
    lib.rj_launch.argtypes = (
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p] + [ctypes.c_longlong] * 5 + [ctypes.c_int]
    )
    return lib


_THR = None


def run_tile(d_thresh, n):
    out = torch.zeros(n, dtype=torch.int64, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _rejection_tile[(triton.cdiv(n, BLOCK),)](
        d_thresh,
        n,
        out,
        A=A,
        B=B,
        C=C,
        E=E,
        SEED=SEED,
        M24=M24,
        MAXITER=MAXITER,
        BLOCK=BLOCK,
        num_warps=1,
    )
    ev1.record()
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ev0.elapsed_time(ev1))


def run_thread(d_thresh, n):
    global _THR
    if _THR is None:
        _THR = _compile_thread()
    out = torch.zeros(n, dtype=torch.int64, device="cuda")
    ms = _THR.rj_launch(d_thresh.data_ptr(), n, out.data_ptr(), A, B, C, E, SEED, MAXITER)
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ms)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    thresh = make_thresh(0)
    d_thresh = torch.as_tensor(thresh, device="cuda")
    if len(sys.argv) >= 3 and sys.argv[1] == "profile":
        (run_tile if sys.argv[2] == "tile" else run_thread)(d_thresh, N)
        return 0

    ref = oracle(thresh[:2048])
    for name, fn in (("tile", run_tile), ("thread", run_thread)):
        got, _ = fn(d_thresh, N)
        if not np.array_equal(got[:2048], ref):
            print(f"ORACLE FAIL {name}: {int((got[:2048] != ref).sum())}/2048 mismatch")
            return 1
    got_all, _ = run_thread(d_thresh, N)
    print(
        f"oracle OK; trips: min {got_all.min()} mean {got_all.mean():.1f} max {got_all.max()} "
        f"(divergent across lanes -> masked-waste test)\n"
    )

    def med(fn):
        for _ in range(WARMUP):
            fn(d_thresh, N)
        return statistics.median([fn(d_thresh, N)[1] for _ in range(SAMPLES)])

    tile_ms, thr_ms = med(run_tile), med(run_thread)
    tile_me = N / (tile_ms * 1e-3) / 1e6
    thr_me = N / (thr_ms * 1e-3) / 1e6
    regret = thr_me / tile_me
    print(f"rejection sampling ({N} elements, pure control-flow divergence, ~no memory):")
    print(f"  TILE (Triton)  {tile_me:8.1f} Melem/s")
    print(f"  THREAD (CUDA)  {thr_me:8.1f} Melem/s")
    print(f"  regret (thread/tile) = {regret:.2f}x  (isolates masked-lane waste from divergence)")
    outp = Path("paper2/data/landmark/rejection_rtx4070.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write("workload,n,tile_melem_s,thread_melem_s,regret,gpu\n")
        f.write(f"rejection,{N},{tile_me:.2f},{thr_me:.2f},{regret:.3f},RTX4070\n")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
