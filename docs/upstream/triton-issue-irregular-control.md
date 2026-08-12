<!-- Reviewer-ready DESIGN ISSUE for github.com/triton-lang/triton.
     Posting = user action. Frame: a characterized + partially-closed gap, not a merge request.
     Suggested labels: enhancement, discussion. Suggested cc: @ptillet @lezcano @ThomasRaoux -->

# [RFC/Discussion] Data-dependent per-lane control flow is structurally inexpressible in TritonGPU — characterization + a partial in-IR mitigation

## Summary

Triton's tile/SPMD model cannot express **per-lane early retirement of a data-dependent loop**: when lanes
of a tile run loops with *different, data-dependent* trip counts, the only available lowering is to mask and
keep every lane issuing the *union* of all control paths at full width. The SIMT model (CUDA, via Independent
Thread Scheduling) retires a lane the instant its loop ends. We measured the resulting gap on a family of
irregular workloads, found it is set by the **execution paradigm, not the abstraction level**, built a small
in-IR pass that captures part of it, and proved (constructively) why the rest cannot be expressed inside
TritonGPU. Filing this to (a) check our reading of the IR with maintainers and (b) discuss whether a
per-lane region construct is in scope — especially now that the CUDA Tile-IR backend makes "what the tile
model structurally cannot express" a live question.

This is a *characterization + partial fix*, not a request to merge niche code.

## Where it bites (measured, oracle-gated, RTX 4070)

Tile (Triton) vs thread (CUDA), same algorithm, same data; regret = tile-time / thread-time:

| workload | regret | note |
|---|---|---|
| pointer-chase (fixed trip) | **1.00×** | negative control — no divergence, must be 1.00 |
| NFA automata (active-set) | 1.96× | divergent scalar control |
| MoE top-k routing (ragged) | 2.36× | ragged expert counts, scalar per-step |
| rejection sampling | 4.00× | divergent trip counts (max/mean ≈ 5) |
| flash-attention (varlen) | **0.64×** | dense vectorizable step → **tile WINS** |

The sign flip is the point: tile *wins* when the divergent step is a dense vectorizable reduction
(attention head-dim) and *loses* when it is scalar control (automata, MoE, rejection). Same compiler,
opposite sign — which is why Triton excels at flash-attention yet collapses on automata. The driver is
per-step instruction-issue efficiency (threads-per-instruction), not arithmetic.

## The IR diagnosis (please sanity-check this)

For an `scf.while`/`scf.for` whose per-lane continuation differs across the tile, per-lane retirement would
need a **per-lane branch**. As far as we can tell that is not expressible in TritonGPU:

- `scf.condition` takes a **single `i1`**, not a per-lane vector predicate. The loop continues until the
  *last* lane is done; finished lanes keep issuing masked work.
- The relevant `#blocked` tensors are already `sizePerThread = 1`, so there is no sub-tile structure left
  to peel into independent per-lane loops at the TritonGPU level.
- A related expressiveness wall: Gluon's `gl.load` only returns layout-carrying tiles (no scalar load), so a
  data-dependent `for k in range(lo, hi)` over a CSR row is also inexpressible. Minimal reproducer:
  `gluon_probe.py` (exits 0 on the expected failure; exits 1 if a future Triton compiles it).

If we are wrong and there is an existing idiom for per-lane data-dependent termination, that would resolve
this — pointers very welcome.

## What we built that *does* fit today (and what doesn't)

- A research pass `tritongpu-thread-region` that detects the lock-step signature (an `scf.while` over
  `#blocked` iter-args whose `scf.condition` derives from a `tt.reduce` of a per-lane predicate) and applies
  a **reduce-hoist** rewrite: replace the per-iteration cross-lane `tt.reduce` with a once-hoisted
  `reduce_max(%trip)` + a scalar counter. Provably equivalent; **~1.55×** on the automata kernel.
- Honest scope: this only helps the masked-loop case, and it does **not** generalize to canonical kernels
  (we verified `-triton-licm` already hoists a genuinely *loop-invariant* reduce, since `tt.reduce` is
  `Pure`; our rewrite is only interesting because the reduce is loop-*variant*). The remaining gap to the
  SIMT baseline (we bound the full thread lowering at **~5.6×** on this kernel) is exactly the per-lane
  retirement the points above say TritonGPU can't express — a below-TritonGPU (`make_llir`) lowering.

## Question for maintainers

1. Is our reading correct that per-lane data-dependent termination is currently inexpressible in TritonGPU
   (single-`i1` `scf.condition`, `sizePerThread=1` tiles)?
2. Is a **per-lane / scalar sub-region** construct (lanes retire independently; `convergent`; reconverge at
   the region terminator; lowered below TritonGPU via per-lane `cond_br` + convergence tokens +
   `bar.warp.sync`) something the project would consider — or is the intended answer "drop to a SIMT/Tile-IR
   fallback for these regions"? The CUDA Tile-IR backend seems to face the same gap, so this may be
   complementary to that effort rather than Triton-specific.

Happy to share the full artifact (witnesses, oracle, Nsight attribution, the pass source, and the ~5.6×
lowering bound) and to prototype whichever direction maintainers prefer.
