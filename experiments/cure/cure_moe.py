"""Harden the built cure on MoE top-k routing (control-bound ML workload), power-law expert counts.
Same _moe_tile lock-step kernel; masked (GPUFSM_THREAD_REGION unset) vs cured (=retire). int64-exact oracle."""
from __future__ import annotations
import os, statistics
import numpy as np, torch, triton, triton.language as tl

A, B, SEED, M24 = 1640531527, 1013904223, 12345, 0xFFFFFF
N, E, MEAN_K = 1 << 20, 64, 8


@triton.jit
def _moe_tile(pool, W, start, nsel, n, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    lo = tl.load(start + i, mask=valid, other=0)
    ns = tl.load(nsel + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int64)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < ns).to(tl.int32)) > 0:
        active = valid & (j < ns)
        e = tl.load(pool + lo + j, mask=active, other=0)
        h = (i.to(tl.int64) * 1640531527 + e.to(tl.int64) * 1013904223 + 12345) & 0xFFFFFF
        w = tl.load(W + e, mask=active, other=0)
        acc = acc + tl.where(active, h * w, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    rng = np.random.default_rng(0)
    raw = np.clip((rng.pareto(1.5, size=N) + 1).astype(np.int64), 1, E)
    nsel = np.clip((raw * (MEAN_K * N / raw.sum())).round().astype(np.int64), 1, E)
    start = np.zeros(N, dtype=np.int32); start[1:] = np.cumsum(nsel)[:-1]
    total = int(nsel.sum())
    pool = np.empty(total, dtype=np.int32); off = 0
    for i in range(N):
        k = int(nsel[i]); pool[off:off + k] = rng.choice(E, size=k, replace=False).astype(np.int32); off += k
    w = ((np.arange(E, dtype=np.int64) * A + SEED) & M24).astype(np.int64)
    tok = np.repeat(np.arange(N, dtype=np.int64), nsel); pe = pool.astype(np.int64)
    ref = np.add.reduceat(((tok * A + pe * B + SEED) & M24) * w[pe], start.astype(np.int64))
    d_pool = torch.as_tensor(pool, device="cuda"); d_w = torch.as_tensor(w, device="cuda")
    d_start = torch.as_tensor(start, device="cuda"); d_nsel = torch.as_tensor(nsel.astype(np.int32), device="cuda")
    BLOCK = 32

    def run():
        out = torch.zeros(N, dtype=torch.int64, device="cuda")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); _moe_tile[(triton.cdiv(N, BLOCK),)](d_pool, d_w, d_start, d_nsel, N, out, BLOCK=BLOCK, num_warps=1); e1.record()
        torch.cuda.synchronize(); return out.cpu().numpy(), float(e0.elapsed_time(e1))

    o, _ = run(); ok = np.array_equal(o, ref)
    for _ in range(3): run()
    t = statistics.median([run()[1] for _ in range(9)])
    print(f"moe powerlaw mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t*1e3:8.1f}us")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
