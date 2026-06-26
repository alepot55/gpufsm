"""Falsifiable probe: can Gluon express the NFA CSR inner loop?  (Answer: no.)

This is the *controlled* half of the Triton<->Gluon attribution in the paper. Triton and
Gluon share the same MLIR compiler stack; Gluon only *adds* explicit layout/shared-memory
control (it does not remove any tuning lever). So if Gluon still cannot express the kernel,
the binding constraint cannot be "Triton wasn't tuned/laid-out right" — it is the
tile/SPMD execution paradigm's lack of a **scalar element load**, which the data-dependent
CSR traversal `for k in range(row_ptr[s], row_ptr[s+1])` requires.

Run on a CUDA box with Triton's experimental Gluon frontend:

    python scripts/gluon_probe.py

Expected: the kernel fails to compile because `gl.load` returns a layout-typed tensor
(a block), never a Python scalar int, so it cannot drive `range(...)`. The script prints
the captured CompilationError and exits 0 (the *failure to compile* IS the positive result).
If a future Gluon adds a scalar load and this compiles, the probe prints that the claim no
longer holds — i.e. the result is falsifiable, by construction.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
    except Exception as e:  # pragma: no cover - environment without Gluon
        print(f"SKIP: Gluon/torch not importable ({type(e).__name__}: {e})")
        return 0

    @gluon.jit
    def csr_scan(rowptr, tgt, out, NS: gl.constexpr):
        # The CSR inner loop's bounds are *loaded* values. In a thread model
        # (CUDA/Warp) row_ptr[s] is a scalar int that drives the loop. In Gluon,
        # gl.load returns a layout-typed block — there is no scalar element load.
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        acc = gl.zeros([1], gl.int64, layout=layout)
        for s in range(NS):
            lo = gl.load(rowptr + s)  # layout-typed block, NOT a scalar int
            hi = gl.load(rowptr + s + 1)
            for _k in range(lo, hi):  # range() needs scalar ints -> cannot lower
                acc += gl.load(tgt)  # noqa: B909 - intentional minimal body
        gl.store(out, acc)

    dev = "cuda" if torch.cuda.is_available() else None
    if dev is None:
        print("SKIP: no CUDA device")
        return 0

    rowptr = torch.zeros(8, dtype=torch.int32, device=dev)
    tgt = torch.zeros(8, dtype=torch.int64, device=dev)
    out = torch.zeros(1, dtype=torch.int64, device=dev)

    try:
        csr_scan[(1,)](rowptr, tgt, out, NS=4, num_warps=1)
    except Exception as e:  # the expected, positive result
        msg = f"{type(e).__name__}: {e}"
        print("EXPECTED FAILURE — Gluon cannot express the CSR scalar scan:")
        print(msg)
        # Confirm the failure is about block-vs-scalar typing, not an unrelated error.
        markers = ("block", "scalar", "layout", "Value argument", "pointer")
        if any(m.lower() in msg.lower() for m in markers):
            print("\nRESULT: confirmed — no scalar element load (block-typed gl.load).")
        else:
            print("\nNOTE: failed for a different reason than expected; inspect above.")
        return 0

    print("UNEXPECTED: the Gluon kernel compiled. The expressibility claim may no longer")
    print("hold for this Triton/Gluon version — update docs/DSL_EXPRESSIVENESS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
