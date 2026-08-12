"""M4c — generalize the built cure across trip distributions (de-risk one-kernel-trick).
Same per-lane-while kernel; masked (GPUFSM_THREAD_REGION unset) vs cured (=retire).
Oracle: acc[i]=trip[i]*(trip[i]-1)/2. Reports oracle + median time per (dist, mode)."""
from __future__ import annotations
import os, statistics
import numpy as np, torch, triton, triton.language as tl


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


def trips(dist, n, rng):
    if dist == "uniform":
        return rng.integers(1, 257, size=n).astype(np.int32)
    if dist == "geometric":
        return np.clip(rng.geometric(1 / 16, size=n), 1, 256).astype(np.int32)
    if dist == "pareto":
        raw = np.clip((rng.pareto(1.5, size=n) + 1), 1, 256)
        return np.clip((raw * (16 * n / raw.sum())).round(), 1, 256).astype(np.int32)
    raise ValueError(dist)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    n, BLOCK = 1 << 20, 32
    for dist in ("uniform", "geometric", "pareto"):
        rng = np.random.default_rng(0)
        tr = trips(dist, n, rng)
        d = torch.as_tensor(tr, device="cuda")
        ref = (tr.astype(np.int64) * (tr.astype(np.int64) - 1) // 2).astype(np.int32)

        def run():
            out = torch.zeros(n, dtype=torch.int32, device="cuda")
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); _perlane_while[(triton.cdiv(n, BLOCK),)](d, out, n, BLOCK=BLOCK, num_warps=1); e1.record()
            torch.cuda.synchronize(); return out.cpu().numpy(), float(e0.elapsed_time(e1))

        o, _ = run(); ok = np.array_equal(o, ref)
        for _ in range(5): run()
        t = statistics.median([run()[1] for _ in range(15)])
        print(f"dist={dist:9} mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t*1e3:8.1f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
