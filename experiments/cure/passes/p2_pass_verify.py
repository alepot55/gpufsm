"""LANDMARK P2 — verify the ThreadRegion DETECTION pass fires inside libtriton.

Compiles the per-lane lock-step kernel (the thread_region target) with the from-source Triton 3.8.0
that contains our new TritonGPU pass `tritongpu-thread-region`. With GPUFSM_THREAD_REGION=1 the pass
runs in make_ttgir and tags the matched scf.while with `ttg.thread_region_candidate`; we assert the
attribute appears in the TTGIR. We also confirm a normal compile (env unset) still produces correct
code, i.e. the pass is a gated no-op for everyone else.

Requires the from-source build (PYTHONPATH=$HOME/m3full_build/triton-src/python):
  PYTHONPATH=... .venv/bin/python experiments/cure/p2_pass_verify.py
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl


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


# Identical body, distinct JITFunction => distinct compile-cache key, so the env-gated ON/OFF
# compiles below don't alias through Triton's kernel cache.
@triton.jit
def _perlane_while_off(inp, out, n, BLOCK: tl.constexpr):
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


def _ttgir(fn) -> str:
    src = triton.compile(
        triton.compiler.ASTSource(
            fn=fn,
            signature={"inp": "*i32", "out": "*i32", "n": "i32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 32},
        )
    )
    return src.asm["ttgir"]


def main() -> int:
    print(f"triton {triton.__version__} from {triton.__file__}")
    if "_C/libtriton" not in triton.__file__ and "m3full_build" not in triton.__file__:
        print("WARN: not the from-source build; set PYTHONPATH to the hacked Triton.")

    # 1) pass ON -> attribute must appear.
    os.environ["GPUFSM_THREAD_REGION"] = "1"
    ttgir_on = _ttgir(_perlane_while)
    fired = "ttg.thread_region_candidate" in ttgir_on
    print(f"  pass ON : thread_region_candidate attribute present = {fired}")

    # 2) pass OFF -> attribute must NOT appear (gated no-op), and kernel still runs correctly.
    del os.environ["GPUFSM_THREAD_REGION"]
    ttgir_off = _ttgir(_perlane_while_off)
    absent = "ttg.thread_region_candidate" not in ttgir_off
    print(f"  pass OFF: attribute absent (gated no-op)            = {absent}")

    ok = True
    if torch.cuda.is_available():
        n = 4096
        trip = torch.arange(n, device="cuda", dtype=torch.int32) % 17
        out = torch.empty(n, dtype=torch.int32, device="cuda")
        _perlane_while[(triton.cdiv(n, 32),)](trip, out, n, BLOCK=32)
        torch.cuda.synchronize()
        ref = (trip * (trip - 1) // 2).to(torch.int32)  # sum_{j<trip} j
        ok = bool(torch.equal(out, ref))
        print(f"  kernel correctness (sum_{{j<trip}} j)               = {ok}")
    else:
        print("  (no CUDA: skipped runtime correctness)")

    passed = fired and absent and ok
    print(
        f"\n=> ThreadRegion detection pass {'VERIFIED' if passed else 'FAILED'} inside libtriton."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
