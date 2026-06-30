"""LANDMARK F2 — ML-domain generality witness: ragged/variable-context attention.

The regret law, on a workload NVIDIA actually cares about. Variable-length attention contexts
(PagedAttention / ragged batching / prefix caching) are THE hot irregular ML kernel: each query
attends over its own key/value slice of a pooled cache; the per-query inner loop length is
data-dependent -- exactly the lock-step `while tl.max(j < seqlen)` signature of the regret law.

  TILE (Triton): one query/lane, flash-style online softmax, lock-steps to the LONGEST context in
    the warp (idle lanes masked); head_dim D in registers.
  THREAD (CUDA, one query/thread): independent online softmax, retires as its context ends.
  ORACLE (numpy, exact softmax attention per query over its ragged slice). Float; allclose-gated.
Two matrices, same kernels (isolates divergence like SpMV uniform-vs-powerlaw):
  UNIFORM seqlen  -> no trip divergence (baseline tile-lowering cost).
  POWER-LAW seqlen -> divergent contexts (adds a divergence increment).
Writes paper2/data/landmark/attention_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/landmark_attention.py
        .venv/bin/python experiments/cure/landmark_attention.py profile <uniform|powerlaw> <tile|thr>
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

N = 1 << 16  # queries
D = 8  # head dim (small but real; isolates the variable-context lock-step)
MEAN_SL = 64  # mean context length
N_CHECK = 2048  # oracle-gated subset (exact python-loop reference)
WARMUP, SAMPLES = 3, 9
BLOCK = int(os.environ.get("ATTN_BLOCK", "32"))
NUM_WARPS = max(1, BLOCK // 32)


def build(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        sl = np.full(N, MEAN_SL, dtype=np.int64)
    else:  # power-law context lengths, same total as uniform (fair memory)
        raw = (rng.pareto(1.5, size=N) + 1).astype(np.int64)
        raw = np.clip(raw, 1, 8192)
        scale = (MEAN_SL * N) / raw.sum()
        sl = np.clip((raw * scale).round().astype(np.int64), 1, 8192)
    start = np.zeros(N, dtype=np.int32)
    start[1:] = np.cumsum(sl)[:-1]
    total = int(sl.sum())
    q = rng.standard_normal((N, D)).astype(np.float32) * 0.5
    k = rng.standard_normal((total, D)).astype(np.float32) * 0.5
    v = rng.standard_normal((total, D)).astype(np.float32)
    return q, k, v, start.astype(np.int32), sl.astype(np.int32)


def oracle(q, k, v, start, sl, n_check):
    out = np.empty((n_check, D), dtype=np.float32)
    for i in range(n_check):
        lo, ln = int(start[i]), int(sl[i])
        ks, vs = k[lo : lo + ln], v[lo : lo + ln]
        s = ks @ q[i]
        s = s - s.max()
        e = np.exp(s)
        w = e / e.sum()
        out[i] = (w[:, None] * vs).sum(0)
    return out


@triton.jit
def _attn_tile(Q, K, V, start, seqlen, n, Out, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    qi = pid * BLOCK + tl.arange(0, BLOCK)
    valid = qi < n
    d = tl.arange(0, D)
    qoff = qi[:, None] * D + d[None, :]
    q = tl.load(Q + qoff, mask=valid[:, None], other=0.0)
    lo = tl.load(start + qi, mask=valid, other=0)
    sl = tl.load(seqlen + qi, mask=valid, other=0)
    m = tl.full((BLOCK,), -1e30, tl.float32)
    lsum = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK, D), tl.float32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < sl).to(tl.int32)) > 0:  # lock-step to the longest context in the warp
        active = valid & (j < sl)
        idx = lo + j
        kvoff = idx[:, None] * D + d[None, :]
        k = tl.load(K + kvoff, mask=active[:, None], other=0.0)
        v = tl.load(V + kvoff, mask=active[:, None], other=0.0)
        s = tl.sum(q * k, axis=1)
        s = tl.where(active, s, -1e30)
        m_new = tl.maximum(m, s)
        corr = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        lsum = lsum * corr + p
        acc = acc * corr[:, None] + p[:, None] * v
        m = m_new
        j = j + 1
    out = acc / lsum[:, None]
    tl.store(Out + qoff, out, mask=valid[:, None])


def _compile_thread():
    src = """
extern "C" __global__ void attn_kernel(
    const float* Q, const float* K, const float* V,
    const int* start, const int* seqlen, int n, float* O, int D) {
  int qi = blockIdx.x * blockDim.x + threadIdx.x;  // one query == one thread
  if (qi >= n) return;
  const float* q = Q + (long)qi * D;
  int lo = start[qi], sl = seqlen[qi];
  float m = -1e30f, l = 0.0f;
  float acc[16];
  for (int dd = 0; dd < D; dd++) acc[dd] = 0.0f;
  for (int j = 0; j < sl; j++) {            // independent: retires when this context ends
    const float* k = K + (long)(lo + j) * D;
    const float* v = V + (long)(lo + j) * D;
    float s = 0.0f;
    for (int dd = 0; dd < D; dd++) s += q[dd] * k[dd];
    float m_new = fmaxf(m, s);
    float corr = expf(m - m_new), p = expf(s - m_new);
    l = l * corr + p;
    for (int dd = 0; dd < D; dd++) acc[dd] = acc[dd] * corr + p * v[dd];
    m = m_new;
  }
  float* o = O + (long)qi * D;
  for (int dd = 0; dd < D; dd++) o[dd] = acc[dd] / l;
}
extern "C" float at_launch(const float* Q, const float* K, const float* V,
    const int* start, const int* seqlen, int n, float* O, int D) {
  int th = 256, bl = (n + th - 1) / th;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  attn_kernel<<<bl, th>>>(Q, K, V, start, seqlen, n, O, D);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}
"""
    cache = Path.home() / ".cache" / "landmark_attention"
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="at_", dir=str(cache)))
    cu, so = d / "at.cu", d / "at.so"
    cu.write_text(src)
    nvcc = "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.at_launch.restype = ctypes.c_float
    lib.at_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    return lib


_THR = None


def to_dev(q, k, v, start, sl):
    dv = torch.device("cuda")
    return (
        torch.as_tensor(q, device=dv),
        torch.as_tensor(k, device=dv),
        torch.as_tensor(v, device=dv),
        torch.as_tensor(start, device=dv),
        torch.as_tensor(sl, device=dv),
    )


def run_tile(g):
    q, k, v, start, sl = g
    o = torch.zeros((N, D), dtype=torch.float32, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    _attn_tile[(triton.cdiv(N, BLOCK),)](
        q, k, v, start, sl, N, o, D=D, BLOCK=BLOCK, num_warps=NUM_WARPS
    )
    ev1.record()
    torch.cuda.synchronize()
    return o.cpu().numpy(), float(ev0.elapsed_time(ev1))


def run_thread(g):
    global _THR
    if _THR is None:
        _THR = _compile_thread()
    q, k, v, start, sl = g
    o = torch.zeros((N, D), dtype=torch.float32, device="cuda")
    ms = _THR.at_launch(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        start.data_ptr(),
        sl.data_ptr(),
        N,
        o.data_ptr(),
        D,
    )
    torch.cuda.synchronize()
    return o.cpu().numpy(), float(ms)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "profile":
        q, k, v, start, sl = build(sys.argv[2], 0)
        g = to_dev(q, k, v, start, sl)
        (run_tile if sys.argv[3] == "tile" else run_thread)(g)
        return 0

    print("ragged attention (ML-irregular): regret vs context-length divergence")
    print(f"{'matrix':>10}{'sl_cv':>8}{'tile_Mp':>10}{'thr_Mp':>10}{'regret':>9}{'oracle':>8}")
    rows = []
    for kind in ("uniform", "powerlaw"):
        q, k, v, start, sl = build(kind, 0)
        g = to_dev(q, k, v, start, sl)
        ref = oracle(q, k, v, start, sl, N_CHECK)
        ok = all(
            np.allclose(fn(g)[0][:N_CHECK], ref, rtol=1e-2, atol=2e-3)
            for fn in (run_tile, run_thread)
        )
        if not ok:
            print(f"{kind:>10}  ORACLE FAIL")
            continue
        cv = float(sl.std() / sl.mean())
        pairs = float(int(sl.sum()))  # query-key pairs processed

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
        outp = Path("paper2/data/landmark/attention_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("matrix,sl_cv,tile_mpair_s,thread_mpair_s,regret,gpu\n")
            for kind, cv, tf, hf, rg in rows:
                f.write(f"{kind},{cv},{tf},{hf},{rg},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
