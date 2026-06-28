"""LANDMARK P2 — the in-IR lowering wall, as a falsifiable probe.

The thread_region DETECTION pass is done (p2_pass_verify.py). The LOWERING half wants each lane to
terminate its loop independently. The natural in-TritonGPU rewrite -- give the matched `scf.while`'s
`scf.condition` a per-lane `tensor<...xi1>` predicate instead of the `tt.reduce`-to-scalar `i1` --
is STRUCTURALLY REJECTED by MLIR: `scf.condition` takes a single `i1`. This probe runs the rewrite
(perlane_while_attempt.mlir) through the built `triton-opt` and asserts the rejection, the way the
Gluon probe asserts its compile error. Exit 0 = wall confirmed (expected); exit 1 = a future
toolchain accepted per-lane control and the "structural wall" claim must be revisited.

Why this matters: the carried tile tensors are ALREADY sizePerThread=1 (one element per lane), so
the lock-step is NOT a layout choice -- it is the loop construct. Per-lane loop termination is
inexpressible in TritonGPU's structured tile control flow; the cure must lower below TritonGPU to
the thread model (ITS) -- exactly what M10 (experiments/cure/m10_scalar_program.py) does, at 4.2x.

Usage:  .venv/bin/python experiments/cure/p2_lowering_wall.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MLIR = HERE / "triton_thread_region_pass" / "perlane_while_attempt.mlir"
EXPECTED = "expects different type"  # the verifier's i1-vs-tensor rejection


def find_triton_opt() -> Path | None:
    for c in [
        Path.home() / "m3full_build" / "triton-src" / "bin" / "triton-opt",
        Path.home() / "m3full_build" / "triton-src" / "build" / "bin" / "triton-opt",
    ]:
        if c.exists():
            return c
    return None


def main() -> int:
    opt = find_triton_opt()
    if opt is None:
        print("SKIP: built triton-opt not found (needs the from-source Triton build).")
        return 0
    proc = subprocess.run([str(opt), str(MLIR)], capture_output=True, text=True)
    err = proc.stderr + proc.stdout
    rejected = EXPECTED in err and ("i1" in err and "tensor<8xi1" in err)
    print(f"triton-opt: {opt}")
    print("attempted rewrite: per-lane scf.condition over a tile tensor")
    if rejected:
        line = next((ln for ln in err.splitlines() if EXPECTED in ln), "")
        # show just the verifier message, not the long absolute path prefix
        msg = line.split("error:", 1)[-1].strip() if "error:" in line else line.strip()
        print(f"  REJECTED (expected): error:{msg[:110]}")
        print(
            "\n=> WALL CONFIRMED: per-lane loop termination is inexpressible in TritonGPU "
            "structured control flow (scf.condition requires i1). The regret is in the loop "
            "construct, not the layout. The cure lowers below TritonGPU (M10, thread model, 4.2x)."
        )
        return 0
    print("  NOT rejected -- a future toolchain may accept per-lane control. Revisit the claim.")
    print(err[:400])
    return 1


if __name__ == "__main__":
    sys.exit(main())
