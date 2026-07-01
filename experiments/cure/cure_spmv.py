"""Harden the built cure on a REAL sparse workload: SpMV (CSR), power-law nnz/row.
Same _spmv_tile lock-step kernel; masked (GPUFSM_THREAD_REGION unset) vs cured (=retire).
Oracle = numpy segmented sum (float32). Prints oracle + median tile time per mode."""
from __future__ import annotations
import os, statistics
import numpy as np, torch, triton, triton.language as tl

N_ROWS = 1 << 20
NCOLS = 1 << 22
K = 16  # mean nnz/row


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
    while tl.max((k < nnz).to(tl.int32)) > 0:
        active = valid & (k < nnz)
        idx = lo + k
        col = tl.load(colidx + idx, mask=active, other=0)
        val = tl.load(values + idx, mask=active, other=0.0)
        xv = tl.load(x + col, mask=active, other=0.0)
        acc = acc + tl.where(active, val * xv, 0.0)
        k = k + 1
    tl.store(y + r, acc, mask=valid)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    rng = np.random.default_rng(0)
    raw = np.clip((rng.pareto(1.5, size=N_ROWS) + 1).astype(np.int64), 1, 4096)
    nnz = np.clip((raw * (K * N_ROWS / raw.sum())).round().astype(np.int64), 1, 4096)
    rowptr = np.zeros(N_ROWS + 1, dtype=np.int32); rowptr[1:] = np.cumsum(nnz)
    total = int(rowptr[-1])
    colidx = rng.integers(0, NCOLS, size=total, dtype=np.int32)
    values = rng.standard_normal(total).astype(np.float32)
    x = rng.standard_normal(NCOLS).astype(np.float32)
    ref = np.add.reduceat(values * x[colidx], rowptr[:-1].astype(np.int64)).astype(np.float32)
    d_rp = torch.as_tensor(rowptr, device="cuda"); d_ci = torch.as_tensor(colidx, device="cuda")
    d_v = torch.as_tensor(values, device="cuda"); d_x = torch.as_tensor(x, device="cuda")
    BLOCK = 32

    def run():
        y = torch.zeros(N_ROWS, dtype=torch.float32, device="cuda")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); _spmv_tile[(triton.cdiv(N_ROWS, BLOCK),)](d_rp, d_ci, d_v, d_x, N_ROWS, y, BLOCK=BLOCK, num_warps=1); e1.record()
        torch.cuda.synchronize(); return y.cpu().numpy(), float(e0.elapsed_time(e1))

    o, _ = run(); ok = np.allclose(o, ref, rtol=1e-3, atol=1e-3)
    for _ in range(3): run()
    t = statistics.median([run()[1] for _ in range(9)])
    print(f"spmv powerlaw mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t*1e3:8.1f}us")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
