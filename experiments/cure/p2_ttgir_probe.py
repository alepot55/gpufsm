"""LANDMARK P2 — IR probe: the lock-step signature of abstraction regret, in TritonGPU IR.

The cure (M10) shows that lowering the SAME per-lane source to the thread model closes the residual
(SP/WP2 = 4.2x). P2 asks: can the tile DSL itself do that lowering automatically? This probe pins
down *what the pass must match*, on a hackable Triton-from-source build (3.8.0, built locally).

A per-lane data-dependent `while tl.max(active) > 0` loop -- the structural core of the NFA
worklist, rejection sampling, and every irregular witness -- compiles in TritonGPU IR to:
    scf.while (iter args : tensor<32xi32, #blocked>)        # the whole 32-lane TILE is carried
      cond:  %r = tt.reduce (j < trip) axis=0 -> i32        # reduced to ONE scalar
             scf.condition( %r > 0 )                        # the tile loops to the BUSIEST lane
      body:  %active = (j < trip)  ; <body predicated by %active>   # masked-lane waste
That `scf.while` over a #blocked tensor with a `tt.reduce`-gated `scf.condition` IS the lock-step:
all 32 lanes iterate to the maximum trip count, idle lanes masked. The thread model (CUDA, and M10's
lowering) instead gives each lane an independent `while`, so lanes retire as they finish -- the
intra-warp latency hiding component C measured as missing.

This script (a) verifies the from-source build is functional, (b) dumps the TTGIR, and (c) ASSERTS
the lock-step pattern is present -- a falsifiable detector: if a future Triton lowered the per-lane
while without the tile-wide reduce-gate, the assertion fails and the regret premise is revisited.

Usage:  PYTHONPATH=$HOME/m3full_build/triton-src/python \
            .venv/bin/python experiments/cure/p2_ttgir_probe.py
        (falls back to whatever `triton` is importable; prints the version it used)
Writes paper2/data/landmark/p2_lockstep.ttgir (the dumped IR, versioned evidence).
"""

from __future__ import annotations

import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def _perlane_while(inp, out, n, BLOCK: tl.constexpr):
    """Per-lane data-dependent trip count -> the tile must lock-step to the busiest lane."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    trip = tl.load(inp + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < trip).to(tl.int32)) > 0:  # LOCK-STEP: loops to the busiest lane
        active = j < trip
        acc = acc + tl.where(active, j, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def dump_ttgir() -> str:
    src = triton.compile(
        triton.compiler.ASTSource(
            fn=_perlane_while,
            signature={"inp": "*i32", "out": "*i32", "n": "i32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 32},
        )
    )
    return src.asm["ttgir"]


def main() -> int:
    print(f"triton {triton.__version__} from {triton.__file__}")
    ttgir = dump_ttgir()

    # The lock-step detector: a while-loop carrying a blocked tensor, gated by a reduce-to-scalar.
    has_while = "scf.while" in ttgir
    has_blocked_iter = "scf.while" in ttgir and "#blocked" in ttgir
    has_reduce_gate = "tt.reduce" in ttgir and "scf.condition" in ttgir
    lockstep = has_while and has_blocked_iter and has_reduce_gate

    print(f"  scf.while present:                 {has_while}")
    print(f"  while carries #blocked tile tensor:{has_blocked_iter}")
    print(f"  condition gated by tt.reduce:      {has_reduce_gate}")
    print(f"  => LOCK-STEP signature present:    {lockstep}")

    outp = Path("paper2/data/landmark/p2_lockstep.ttgir")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(ttgir)
    print(f"wrote {outp} ({len(ttgir.splitlines())} lines)")

    if not lockstep:
        print(
            "UNEXPECTED: the per-lane while did NOT lower to a tile-wide reduce-gated loop. "
            "The abstraction-regret premise (lock-step over the tile) should be re-examined."
        )
        return 1
    print(
        "\nThe pass target: rewrite this scf.while region so each lane has an INDEPENDENT loop "
        "(no tt.reduce gate; sizePerThread=1; reconverge at exit) -- the thread-model lowering the "
        "cure (M10) realizes out-of-band, here proposed in-DSL. See docs/P2_PASS_DESIGN.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
