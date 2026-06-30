# [RFC] A per-lane sub-tile region for data-dependent control flow

*Draft — intended to be posted as a `[RFC]` issue on `triton-lang/triton`. Status: proposal +
reference detection pass built; the lowering is specified, not yet upstreamed. Author evidence and a
runnable structural-impossibility probe accompany this draft (see "Evidence" and "Reference work").*

## Summary

Triton's tile/SPMD model has no way to express **per-lane data-dependent control flow** — a loop whose
trip count, and termination, differ per element of a tile. Today such code is written as a `tl.while`
over a tile-wide predicate, which the compiler lowers to a single `scf.while` whose `scf.condition` is a
**tile-wide `tt.reduce` to one `i1`**: the whole tile lock-steps to the *busiest* lane while finished
lanes are masked. On irregular, control-flow-bound workloads this is a large, measurable performance
loss versus the equivalent thread-/SIMT-model (CUDA) kernel. We propose a first-class **per-lane
sub-tile region** (`tt.scalar_region` / a `serial_range` loop form) in which each lane executes an
independent scalar instruction stream with its own loop trip and early exit, reconverging at the region
boundary — recovering the thread model's behavior *inside* a Triton kernel, while the regular part stays
tiled.

## Motivation

Tile DSLs assume regular, dense tensor work. On data-dependent control flow they collapse: finite-state
automata, sparse/graph frontier expansion, rejection-style sampling, and — in ML — **MoE top-k routing**
with ragged, imbalanced expert loads. We measured the gap precisely on a work-efficient NFA worklist and
a suite of eight oracle-gated tile-vs-thread witnesses (same per-lane source lowered to a Triton tile
kernel vs a CUDA thread kernel vs a CPU oracle; every number traces to a versioned CSV):

- The same idiomatic per-lane program lowered to the thread model runs **~4.2×** faster than the tile
  lowering on the NFA worklist (≤64 states), and **matches or exceeds hand-written CUDA**.
- A **regret law** across the witnesses: the tile loses on **scalar-control** irregularity (automata
  ~2×, rejection ~4×, MoE routing ~2.4×) — and, as a *correct negative*, the tile **wins** when the
  per-step payload is a dense vector (ragged attention head-dim: regret ~0.64×). The sign is set by
  per-step instruction efficiency, not "irregularity": scalar work makes the lock-step tile issue *more*
  instructions (the reduce-gated `while` + masked full-width lanes); a dense head-dim it vectorizes.
- This is exactly why Triton is the tool of choice for flash-attention yet collapses on automata/MoE
  routing: the missing capability is *per-lane scalar control*, not layout.

Crucially, this gap is **not closed** by the adjacent work: Gluon exposes per-thread *layout* (Linear
Layouts) but not the scalar element load a data-dependent loop needs; TLX / warp specialization
(incl. Tawa) specialize *warps* for different tasks, not *lanes* within a tile; and NVIDIA's CUDA Tile IR
backend for Triton is a tile-level path that likewise does not address per-lane data-dependent control.
The primitive proposed here is complementary to all of them.

## The IR-level diagnosis (why `tl.while` is lock-step today)

For a per-lane data-dependent loop, the TritonGPU IR is:

```mlir
scf.while (%acc, %j : tensor<BxT, #blocked>, tensor<BxT, #blocked>) {       // the whole tile is carried
  %p = arith.cmpi slt, %j, %trip : tensor<BxT, #blocked>                    // per-lane predicate
  %r = "tt.reduce"(%p) axis=0 -> i32                                        // reduced to ONE scalar
  %c = arith.cmpi sgt, %r, 0 : i32
  scf.condition(%c) ...                                                     // tile loops to busiest lane
} do { ^bb0(...): <body predicated by (%j < %trip)> ; scf.yield ... }
```

The lock-step is structural: `scf.condition` takes a **single `i1`**, so per-lane termination cannot be
expressed — a rewrite that gives `scf.condition` a per-lane `tensor<Bxi1>` is rejected by the MLIR
verifier. The carried tensors are already `sizePerThread=1` (one element/lane), so this is **not** a
layout problem that Gluon could fix; it is the loop construct. (A runnable probe demonstrates both the
detection of this signature inside `libtriton` and the verifier rejection of the per-lane-condition
rewrite.)

## Proposal: `tt.scalar_region` / `serial_range`

A region (surface syntax: a `tl.scalar_region()` context, or a `serial_range` loop form) in which the
marked code executes **per lane as an independent scalar instruction stream**:

- **Semantics.** Inside the region, a tile of shape `[B]` (`sizePerThread=1`, `threadsPerWarp=W`) is
  viewed as `B` independent scalar programs. Each lane has its own control flow: a `while`/`for` with a
  per-lane trip count and early `break`; scalar loads/stores at per-lane addresses; a register-resident
  per-lane scalar state. Lanes diverge and **retire independently** (Volta+ Independent Thread
  Scheduling). At the region exit there is an implicit **reconvergence barrier**, after which downstream
  tile ops see a coherent tile again. Cross-lane ops (`tt.reduce`, layout conversions) are **disallowed**
  inside the region (it is scalar-per-lane by construction).
- **Example.**
  ```python
  # outside: regular tiled setup (coalesced loads of per-lane start/len)
  with tl.scalar_region():
      acc = tl.zeros((), tl.int32)            # per-lane scalar
      for j in tl.serial_range(0, length):    # per-lane trip count; lanes retire independently
          e = tl.load(ptr + off + j)          # scalar load at a per-lane address
          acc += f(e)
  # outside: reconverged; `acc` is a normal [B] tile again
  ```
- **Lowering (sketch).** Detect the marked region; keep iter values in `sizePerThread=1` registers;
  lower the per-lane loop **below TritonGPU** (TritonGPU→LLVM/NVVM) to a thread-style loop with a per-lane
  branch (no `tt.reduce` gate), relying on ITS for independent lane progress; **disable the software
  pipeliner / multibuffering** inside the region (it assumes lock-step tiles); insert a `bar.warp.sync`
  (or block barrier) at the region exit to reconverge. The thread model is the *existence proof* that
  this lowering is sound and fast (CUDA achieves 1.0×, the high-level Warp DSL ~0.9× on the same source).
- **Cost-model selection.** A pass can auto-detect the lock-step signature (the `scf.while` over
  `#blocked` carried tensors gated by a `tt.reduce`) and route only those regions through the per-lane
  lowering, leaving everything else tiled — so the feature is opt-in *and* auto-applicable.

## Alternatives & prior art (and why they differ)

- **Gluon / Linear Layouts** — per-thread *layout*, not per-lane *control flow*; `gl.load` returns a
  layout-typed block, never a scalar, so the data-dependent loop cannot be lowered. Orthogonal.
- **TLX / warp specialization / Tawa (CGO'26)** — specialize *warps* for different tasks (dense
  producer/consumer pipelines); they do not give *lanes within a tile* independent scalar control. The
  proposed region composes with, but is distinct from, warp specialization.
- **NVIDIA cuTile / CUDA Tile IR** — a tile-level Virtual ISA + a Tile IR backend for Triton; tile-level,
  and does not address per-lane data-dependent control / SIMT fallback. The proposal targets exactly the
  irregular case tile IRs omit, and could inform a Tile IR `scalar_region` op too.
- **Manual rewrite to a fixed-trip masked loop** — the status quo; pays the masked full-width cost and,
  on scalar payloads, loses ~2–4×.
- **CPU whole-function / partial-CFG vectorization (Moll & Hack, PLDI'18)** — the *opposite* direction
  (vectorize divergent CFG into lanes); here we *de-vectorize* a marked region to recover per-lane MLP.

## Compatibility & scope

Opt-in and region-local: no change to existing kernels. The region is only valid where the tile is
`sizePerThread=1` along the lane axis; cross-lane ops are diagnosed as errors inside it. Initial scope:
a single per-lane `serial_range`/`while` with scalar state and scalar loads/stores — enough for automata,
SpMV/CSR, graph frontier, rejection sampling, and MoE routing. Nested/region-of-region and per-lane
shared-memory are out of scope for v1.

## Evaluation plan

Reuse the oracle-gated witness suite (correctness vs a CPU oracle, bit-exact or `allclose`) and report
tile-vs-thread regret + Nsight attribution (issue activity, threads-per-instruction, long-scoreboard
stall, occupancy) per witness, on ≥2 architectures (consumer + datacenter). Success = a region lowered
**by the compiler** that is oracle-correct and closes the scalar-control gap toward the thread model
(target: within ~1.3× of CUDA on the NFA worklist), with no regression on the regular/dense path.

## Evidence & reference work (what exists today)

- Eight oracle-gated tile-vs-thread witnesses + the regret law (incl. the ML sign-flip), every number
  CSV-traced; the out-of-band thread lowering of the *same* per-lane source closes the residual ~4.2×.
- A TritonGPU detection pass (`tritongpu-thread-region`) that compiles into `libtriton`, matches the
  lock-step signature, and tags candidate regions; a falsifiable probe that the per-lane-condition
  rewrite is rejected by the MLIR verifier (the structural wall).
- A REAL in-compiler rewrite in the same pass (opt-in): it hoists `reduce_max(trip)` once and replaces
  the per-iteration cross-lane reduce with a scalar counter (provably equivalent; preserves per-warp
  termination). Built into `libtriton`, oracle-correct end-to-end, **1.55x** on the lock-step kernel
  (155.6->100.4 us). This is a partial in-compiler win -- it trims the lock-step loop's reduce overhead
  but does NOT grant per-lane sub-warp retirement; that is exactly what the proposed primitive adds.
- Honest status: the *detection* + the structural-impossibility result + an *automatic selector* over an
  out-of-band thread lowering are built; the **in-compiler per-lane lowering proposed here is not yet
  upstreamed** — that is what this RFC is for.
