"""F3 — bound the FULL-cure payoff vs the built reduce-hoist, on the lock-step kernel.

Three points for `acc[i] = sum_{j<trip[i]} j` (power-law trips), oracle-identical:
  tile baseline (Triton lock-step while)      -- measured elsewhere (155.6 us, tipi 32)
  reduce-hoist (built in-compiler pass)       -- measured elsewhere (100.4 us = 1.55x, tipi 32)
  thread (CUDA, one thread/element, retiring) -- THIS script: per-lane retirement (tipi < 32)
The thread version is the target of the FULL below-TritonGPU lowering. The gap from reduce-hoist to
thread is exactly the per-lane (sub-warp) retirement the structural wall blocks in TritonGPU.
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

N = 1 << 20


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    rng = np.random.default_rng(0)
    raw = np.clip((rng.pareto(1.5, size=N) + 1).astype(np.int64), 1, 256)
    trip = np.clip((raw * (16 * N / raw.sum())).round().astype(np.int64), 1, 256).astype(np.int32)
    ref = (trip.astype(np.int64) * (trip.astype(np.int64) - 1) // 2).astype(np.int32)
    d_trip = torch.as_tensor(trip, device="cuda")
    src = """
extern "C" __global__ void k(const int* trip, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int t = trip[i], acc = 0;
  for (int j = 0; j < t; j++) acc += j;   // per-lane trip; the thread retires when its loop ends
  out[i] = acc;
}
extern "C" float launch(const int* trip, int* out, int n) {
  int th = 32, bl = (n + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); k<<<bl, th>>>(trip, out, n); cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms; }
"""
    cache = Path.home() / ".cache" / "f3_full_cure"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=str(cache)))
    cu, so = d / "k.cu", d / "k.so"
    cu.write_text(src)
    subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-O3",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-arch=sm_89",
            "-o",
            str(so),
            str(cu),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.launch.restype = ctypes.c_float
    lib.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    def run():
        out = torch.zeros(N, dtype=torch.int32, device="cuda")
        ms = lib.launch(d_trip.data_ptr(), out.data_ptr(), N)
        torch.cuda.synchronize()
        return out.cpu().numpy(), float(ms)

    o, _ = run()
    ok = np.array_equal(o, ref)
    for _ in range(5):
        run()
    t = statistics.median([run()[1] for _ in range(15)])
    print(f"thread (per-lane retiring) oracle={'OK' if ok else 'FAIL'} time={t * 1e3:.1f}us")
    print("  vs tile 155.6us (1.0x) / reduce-hoist 100.4us (1.55x) -> full-cure target ~5.6x")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
