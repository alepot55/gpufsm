"""LANDMARK F2 — ML-domain witness #2: MoE top-k expert routing (SCALAR-control irregular).

The scalar-control counterpart to the attention witness, on a workload NVIDIA cares about (MoE).
Each token is routed to a DATA-DEPENDENT number of experts (top-k by a threshold on per-token
router scores -> variable, imbalanced "expert load" like real MoE); the kernel then accumulates
each selected expert's scalar contribution. SAME ragged variable-count structure as attention,
but the per-step work is SCALAR (one integer multiply-add), not a dense head-dim -- isolating
control divergence -> the law predicts regret > 1 here (tile loses), the OPPOSITE of attention. The
attention(dense,<1) vs MoE(scalar,>1) contrast at identical ragged structure is the cleanest
demonstration of the mechanism (per-step work density, not "irregularity", decides the sign).

  TILE (Triton): one token/lane, lock-step `while j < n_sel` to the busiest token in the warp.
  THREAD (CUDA, one token/thread): independent; retires when its expert list ends.
  ORACLE (numpy, exact integer routing). Bit-exact (deterministic hash, products fit int64).
Two variants (isolate divergence): UNIFORM n_sel (no divergence) vs POWER-LAW n_sel (imbalance).
Writes paper2/data/landmark/moe_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/landmark_moe.py
        .venv/bin/python experiments/cure/landmark_moe.py profile <uniform|powerlaw> <tile|thr>
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

A, B, SEED = 1640531527, 1013904223, 12345  # odd mixing constants < 2^31 (int32-literal safe)
M24 = 0xFFFFFF
N = 1 << 20  # tokens
E = 64  # experts
MEAN_K = 8  # mean selected experts/token
WARMUP, SAMPLES = 3, 9
BLOCK = int(os.environ.get("MOE_BLOCK", "32"))
NUM_WARPS = max(1, BLOCK // 32)


def build(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        nsel = np.full(N, MEAN_K, dtype=np.int64)
    else:  # power-law selected counts: realistic MoE load imbalance
        raw = (rng.pareto(1.5, size=N) + 1).astype(np.int64)
        raw = np.clip(raw, 1, E)
        scale = (MEAN_K * N) / raw.sum()
        nsel = np.clip((raw * scale).round().astype(np.int64), 1, E)
    start = np.zeros(N, dtype=np.int32)
    start[1:] = np.cumsum(nsel)[:-1]
    total = int(nsel.sum())
    # each token's selected experts = a random subset of size nsel[i] (ids in [0,E))
    pool = np.empty(total, dtype=np.int32)
    off = 0
    for i in range(N):
        k = int(nsel[i])
        pool[off : off + k] = rng.choice(E, size=k, replace=False).astype(np.int32)
        off += k
    w = ((np.arange(E, dtype=np.int64) * A + SEED) & M24).astype(np.int64)  # expert weights
    return pool, w, start.astype(np.int32), nsel.astype(np.int32)


def oracle(pool, w, start, nsel):
    tok = np.repeat(np.arange(N, dtype=np.int64), nsel)  # token id per pool entry
    pe = pool.astype(np.int64)
    h = (tok * A + pe * B + SEED) & M24
    contrib = h * w[pe]  # < 2^48, fits int64
    return np.add.reduceat(contrib, start.astype(np.int64))


@triton.jit
def _moe_tile(pool, W, start, nsel, n, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    lo = tl.load(start + i, mask=valid, other=0)
    ns = tl.load(nsel + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int64)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < ns).to(tl.int32)) > 0:  # lock-step to the busiest token in the warp
        active = valid & (j < ns)
        e = tl.load(pool + lo + j, mask=active, other=0)
        h = (i.to(tl.int64) * 1640531527 + e.to(tl.int64) * 1013904223 + 12345) & 0xFFFFFF
        w = tl.load(W + e, mask=active, other=0)
        acc = acc + tl.where(active, h * w, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def _compile_thread():
    src = """
extern "C" __global__ void moe_kernel(
    const int* pool, const long long* W, const int* start, const int* nsel, int n, long long* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // one token == one thread
  if (i >= n) return;
  int lo = start[i], ns = nsel[i];
  long long acc = 0;
  for (int j = 0; j < ns; j++) {  // independent: retires when this token's list ends
    int e = pool[lo + j];
    long long m = (long long)i * 1640531527LL + (long long)e * 1013904223LL + 12345LL;
    long long h = m & 0xFFFFFFLL;
    acc += h * W[e];
  }
  out[i] = acc;
}
extern "C" float moe_launch(const int* pool, const long long* W, const int* start,
                            const int* nsel, int n, long long* out) {
  int th = 256, bl = (n + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  moe_kernel<<<bl, th>>>(pool, W, start, nsel, n, out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}
"""
    cache = Path.home() / ".cache" / "landmark_moe"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="moe_", dir=str(cache)))
    cu, so = d / "moe.cu", d / "moe.so"
    cu.write_text(src)
    nvcc = "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.moe_launch.restype = ctypes.c_float
    lib.moe_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
    return lib


_THR = None


def to_dev(pool, w, start, nsel):
    dv = torch.device("cuda")
    return (
        torch.as_tensor(pool, device=dv),
        torch.as_tensor(w, device=dv),
        torch.as_tensor(start, device=dv),
        torch.as_tensor(nsel, device=dv),
    )


def run_tile(g):
    pool, w, start, nsel = g
    out = torch.zeros(N, dtype=torch.int64, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _moe_tile[(triton.cdiv(N, BLOCK),)](
        pool, w, start, nsel, N, out, BLOCK=BLOCK, num_warps=NUM_WARPS
    )
    ev1.record()
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ev0.elapsed_time(ev1))


def run_thread(g):
    global _THR
    if _THR is None:
        _THR = _compile_thread()
    pool, w, start, nsel = g
    out = torch.zeros(N, dtype=torch.int64, device="cuda")
    ms = _THR.moe_launch(
        pool.data_ptr(), w.data_ptr(), start.data_ptr(), nsel.data_ptr(), N, out.data_ptr()
    )
    torch.cuda.synchronize()
    return out.cpu().numpy(), float(ms)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        g = to_dev(*build(sys.argv[2], 0))
        (run_tile if sys.argv[3] == "tile" else run_thread)(g)
        return 0

    print("MoE top-k routing (ML-irregular, SCALAR per-step): regret vs expert-count divergence")
    print(f"{'matrix':>10}{'k_cv':>8}{'tile_Mp':>10}{'thr_Mp':>10}{'regret':>9}{'oracle':>8}")
    rows = []
    for kind in ("uniform", "powerlaw"):
        pool, w, start, nsel = build(kind, 0)
        g = to_dev(pool, w, start, nsel)
        ref = oracle(pool, w, start, nsel)
        ok = all(np.array_equal(fn(g)[0], ref) for fn in (run_tile, run_thread))
        if not ok:
            print(f"{kind:>10}  ORACLE FAIL")
            continue
        cv = float(nsel.std() / nsel.mean())
        pairs = float(int(nsel.sum()))  # (token,expert) pairs processed

        def med(fn, g=g):
            for _ in range(WARMUP):
                fn(g)
            return statistics.median([fn(g)[1] for _ in range(SAMPLES)])

        tile_mp = pairs / (med(run_tile) * 1e-3) / 1e6
        thr_mp = pairs / (med(run_thread) * 1e-3) / 1e6
        regret = thr_mp / tile_mp
        print(f"{kind:>10}{cv:8.2f}{tile_mp:10.1f}{thr_mp:10.1f}{regret:9.2f}{'  ok':>8}")
        rows.append((kind, round(cv, 3), round(tile_mp, 1), round(thr_mp, 1), round(regret, 3)))
    if rows:
        outp = Path("paper2/data/landmark/moe_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("matrix,k_cv,tile_mpair_s,thread_mpair_s,regret,gpu\n")
            for kind, cv, tf, hf, rg in rows:
                f.write(f"{kind},{cv},{tf},{hf},{rg},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
