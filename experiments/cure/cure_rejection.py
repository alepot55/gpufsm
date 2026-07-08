"""Cure on the rejection-sampling witness (pure control-flow divergence, ~no memory).
Same _rejection_tile lock-step kernel as landmark_rejection; masked
(TRITON_ENABLE_PERLANE_LOOP_RETIREMENT unset) vs cured (=1). The regret law predicts the
LARGEST real-workload cure gain here (regret 4.0x was pure masked-lane waste).
Oracle: numpy-vectorized exact 64-bit-wrap accept iteration. Prints median time per mode."""
from __future__ import annotations

import os
import statistics

import numpy as np
import torch
import triton
import triton.language as tl

A, B, C, E = 2654435761, 2246822519, 3266489917, 668265263
SEED = 12345
M24 = 0xFFFFFF
N = 1 << 20
MAXITER = 256
BLOCK = 32


@triton.jit
def _rejection_tile(
    thresh, n, out,
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, E: tl.constexpr,
    SEED: tl.constexpr, M24: tl.constexpr, MAXITER: tl.constexpr, BLOCK: tl.constexpr,
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


def oracle(thresh: np.ndarray) -> np.ndarray:
    i = np.arange(N, dtype=np.uint64)
    acc = np.full(N, MAXITER, dtype=np.int64)
    done = np.zeros(N, dtype=bool)
    for j in range(MAXITER):
        t = (i * np.uint64(A) + np.uint64(j) * np.uint64(B) + np.uint64(SEED))
        t = t * np.uint64(C)
        t = t * np.uint64(E)
        draw = (t & np.uint64(M24)).astype(np.int64)
        hit = ~done & (draw < thresh)
        acc[hit] = j
        done |= hit
        if done.all():
            break
    return acc


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    mode = os.environ.get("GPUFSM_THREAD_REGION", "off")
    rng = np.random.default_rng(SEED)
    thresh = ((1 << 21) + rng.integers(0, 1 << 22, size=N)).astype(np.int64)
    ref = oracle(thresh)
    d_th = torch.as_tensor(thresh, device="cuda")

    def run():
        out = torch.zeros(N, dtype=torch.int64, device="cuda")
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _rejection_tile[(triton.cdiv(N, BLOCK),)](
            d_th, N, out, A=A, B=B, C=C, E=E, SEED=SEED, M24=M24,
            MAXITER=MAXITER, BLOCK=BLOCK, num_warps=1,
        )
        e1.record()
        torch.cuda.synchronize()
        return out.cpu().numpy(), float(e0.elapsed_time(e1))

    o, _ = run()
    ok = np.array_equal(o, ref)
    for _ in range(3):
        run()
    t = statistics.median([run()[1] for _ in range(9)])
    print(f"rejection mode={mode:6} oracle={'OK' if ok else 'FAIL'} time={t*1e3:8.1f}us")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
