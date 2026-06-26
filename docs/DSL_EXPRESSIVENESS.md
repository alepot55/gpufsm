# DSL Expressiveness on Irregular Automata — the Abstraction Spectrum

Generated: 2026-06-25 (session 2). Hardware: RTX 4070 (sm_89). Triton 3.5.1, Warp 1.14.0,
CUDA toolkit 13.3 / driver 580 (max CUDA 13.0).

This note records *which GPU programming models can even express* the irregular NFA
kernel, and at what cost — the empirical backbone of the "abstraction regret" thesis:
**for irregular automata the binding constraint is the data-dependent control flow a DSL
lets you express, not only the memory layout.** Each row is backed by a real, run attempt
in this repo (not a literature claim).

## The kernel's irreducible needs

An NFA step over CSR transitions requires, per active state `s`:
1. a **scalar** membership test on the active-set bitmask,
2. a **data-dependent inner loop** `for k in range(row_ptr[s], row_ptr[s+1])` whose bounds
   are *loaded values*, and
3. a **scatter** `next |= 1 << target[k]` into the next-state bitmask.

This is scalar, branch-heavy, scatter-heavy work — the opposite of dense tile/tensor algebra.

## Results (this repo, validated against the `reference.py` oracle)

| Model | Abstraction | Expresses the NFA kernel? | Evidence | Verdict |
|---|---|---|---|---|
| **CUDA** | low (C++ SIMT) | **Yes**, fully | `dense`, `bitpacked`, `multistream`, `multistream_shared`, `multistream_async` all pass | Full control: scalar branches, dynamic loops, register/shared/global, bit ops, streams |
| **NVIDIA Warp** | high (Python, **thread-SIMT**) | **Yes** | `warp/multistream` passes (≤64 states, register `uint64`) | Python productivity *and* a thread model that expresses per-state control flow + bit ops |
| **Triton** | high (Python, **tile/SPMD**) | **Partially** — only as a single unrolled program | `dense`, `bitpacked`, `multistream` pass, but: `return` forbidden in loops (needs a `done`-latch rewrite); int literals truncate to int32 (bit masks must be `int64` scalars); cannot place CSR in shared memory (compiler-owned) | Works but fights the model; no explicit memory-layout control |
| **Gluon** (Triton experimental low-level) | mid (tile + **explicit layouts/shared mem**) | **No** (for this kernel) | see probe below | Exposes layout/shared-mem control Triton hides, but still tile-only: **no scalar load**, so data-dependent control flow over loaded CSR values is inexpressible |

Update (work-efficient worklist): Triton **can** express the work-efficient active-set
kernel too — `libdevice.ffs` + a data-dependent `while` loop iterate set bits — and it is
validated against the oracle. But it still pays **~6.5× regret vs CUDA** on that kernel
(CUDA worklist 164–170 Gbps vs Triton worklist 24–25 Gbps at ≤64 states), essentially the
same as the 6–8× on the full-scan kernel. So for Triton **expressibility ≠ efficiency**: even when it expresses
the right algorithm, the tile/SPMD model imposes a large constant penalty on scalar,
data-dependent automata work. Gluon, by contrast, cannot express it at all.
| **Tensor-only DSLs** (cuTile/Tile IR, CUTLASS CuTe DSL, ThunderKittens, JAX/Pallas, TileLang) | high (tile/tensor) | **No** | not attempted — dense-tile/tensor-core model; automata scatter/branch must be faked as masked dense ops | Off-axis for irregular automata; discuss in related work, do not benchmark |

## Gluon probe (concrete evidence)

Minimal attempt at the CSR scalar scan (the core NFA inner loop) in Gluon:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def probe(rowptr, tgt, out, NS: gl.constexpr):
    acc = gl.zeros([1], gl.int64, layout=gl.BlockedLayout([1],[32],[1],[0]))
    for s in range(NS):
        lo = gl.load(rowptr + s)            # <- returns a layout-typed BLOCK, not a scalar
        hi = gl.load(rowptr + s + 1)
        for k in range(lo, hi):             # <- range() needs scalar ints, not blocks
            t = gl.load(tgt + k)
    gl.store(out, acc)
```

Compilation fails:

```
triton ... CompilationError: Value argument cannot be block type if pointer argument is not a block
```

Root cause: Gluon's `gl.load` always returns a **layout-typed tensor** — there is no
scalar element load. Therefore the loaded `row_ptr[s]` values cannot drive a Python
`range(...)`, and the data-dependent inner loop that defines CSR traversal cannot be
expressed. Gluon **relaxes the memory-layout constraint** (it adds `allocate_shared_memory`,
explicit `BlockedLayout`/`SwizzledSharedLayout`, `thread_barrier`, `warp_specialize`) but
**does not relax the control-flow constraint**, which is the binding one for automata.

## Thesis implication

Placing the models on a 2-D map — *abstraction level* (x) vs *expressible control flow +
memory layout* (y):

- **High control:** CUDA, Warp (Warp is high-abstraction yet thread-model → expresses automata).
- **Layout control but tile-only:** Gluon (more memory control than Triton, same control-flow wall).
- **Tile/SPMD, layout hidden:** Triton (expresses automata only as a strained single program).
- **Tensor-only (off-axis):** cuTile, CuTe DSL, ThunderKittens, Pallas, TileLang.

Automata throughput tracks the **y-axis (expressible control flow + layout)**, not the
x-axis (how "high-level" the DSL looks). Crucially, Warp (high abstraction) expresses the
kernel while Gluon and the tensor DSLs (also high/mid abstraction) cannot — so the regret is
*not* a simple function of abstraction height; it is set by **what control flow + memory
layout the model forbids you to express**. For irregular automata the **control-flow**
limit dominates: Gluon's extra layout control buys nothing because the kernel cannot be
written at all.

## Reproduce

`backends/warp_backend.py` (Warp, works) and the probe above (Gluon, fails to compile) are
both runnable on this machine. The CUDA/Triton techniques are validated by `pytest -m gpu`.
