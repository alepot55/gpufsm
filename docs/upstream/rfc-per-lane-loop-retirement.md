# [RFC] Per-lane retirement for divergent while-loops (verified, opt-in NVIDIA pass)

## TL;DR

Triton lowers a data-dependent loop such as

```python
while tl.max((j < trip).to(tl.int32)) > 0:
    active = j < trip
    acc += tl.where(active, j, 0)
    j += 1
```

to a **warp-lock-step** loop: the latch warp-reduces the per-lane predicate
(`nvvm.redux.sync max`) and branches on the uniform result, so **every lane
iterates until the slowest lane in its warp finishes** and pays a
warp-collective reduction per iteration. On sm_70+ (independent thread
scheduling) that schedule is a codegen choice, not a hardware requirement.

This RFC proposes an **opt-in** NVIDIA-backend pass
(`nv-per-lane-loop-retirement`) that rewrites the latch to branch on the
**per-lane** predicate — each lane retires as soon as its own condition
fails — deletes the then-dead cross-lane reduction, and reconverges the
entering lanes (`activemask` captured at the preheader) with
`bar.warp.sync` at the loop exit.

The rewrite is guarded by a **static verifier that proves observational
equivalence** with the lock-step schedule before touching the loop; loops
it cannot prove safe are left untouched. Draft PR: <PR-LINK>. All numbers
below measured on RTX 4070 (sm_89), each run oracle-checked bit-exact
against a CPU reference.

## Why it matters (measurements)

On the canonical kernel above (2^20 elements, `BLOCK=32`, geometric trip
distribution, mean 16, clamp 256):

- **2.5–4.2x** end-to-end depending on trip distribution and clocks
  (166.7 -> 40.2 us on the canonical config), bit-exact output.
- Nsight: **39x fewer issued instructions** (36.1M -> 0.92M) — work becomes
  proportional to `sum(trip_i)` instead of `32 x max(trip per warp)`.
- The effect is *predictable*: across six controlled trip distributions the
  masked baseline follows `t = 50.3 + 1.08 * E[warp-max trip]` us with
  R^2 = 0.998, while the retired kernel collapses to a flat ~40 us floor.
  The law holds out-of-sample (four held-out distributions, mean error 5%).
  The cost driver is the *absolute straggler*, not the divergence ratio: a
  single straggler lane (1/32 lanes at trip 256, the rest at 2) costs
  **6.6x**.
- PTX/SASS confirm the mechanism: PTX `redux.sync` 1 -> 0 and an explicit
  `bar.warp.sync`; in SASS the hardware warp-collective `REDUX.MAX.S32` is
  eliminated (static instructions 48 -> 40) and ptxas realizes the
  reconvergence through its convergence barriers (`BSSY`/`BSYNC`).
- Real workloads with the same latch shape (power-law CSR SpMV row loop,
  MoE top-k routing) see 1.14–1.25x — modest because their cost is
  gather-bound, which is itself confirming: the pass recovers *control*
  overhead, and does no harm when control is not the bottleneck.

## Design

A single module pass over the LLVM dialect
(`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/PerLaneLoopRetirement.cpp`),
run inside `make_llir` after the first canonicalize/CSE, gated on
`capability >= 70`. Enablement is a **`CUDAOptions` field**
(`per_lane_loop_retirement: bool`, so it participates in the compilation
cache key and can be set per kernel launch), whose default comes from the
`TRITON_ENABLE_PERLANE_LOOP_RETIREMENT` env knob — the same pattern as
`ptx_options`.

Matched latch shape (with robust peeling of the extractvalue/insertvalue
wrappers the tile lowering packs per-lane scalars in):

```
%pred = llvm.icmp <cmp> ...                  // any per-lane comparison
%z    = zext/select-normalized %pred to i32
%r    = nvvm.redux.sync max|umax|or|add %z, C(-1)   // full-warp mask
%any  = llvm.icmp sgt|ne %r, C(0)
llvm.cond_br %any, ^body, ^exit
```

(over the {0,1} range of the normalized input, max/umax/or/add all compute
"any lane active" against 0).

## The equivalence argument (what the verifier proves)

Under the masked lock-step schedule an "inactive" lane still executes the
body with `active = false`; after the rewrite it does not execute at all.
The two schedules are observationally equivalent iff the iterations a lane
would have spent masked-off can neither change its own live state nor be
observed by anyone else. The verifier checks, structurally and
conservatively (refusing the loop otherwise):

1. **Body safety** — no operation requiring the retired lane's
   participation or with effects that would be lost: any NVVM op
   (collectives, barriers, nested redux latches), calls, atomics,
   unpredicated `llvm.store`, and inline asm that stores/synchronizes
   (`st.`/`atom`/`red.`/`bar.`/`membar`/`fence`). Predicated gathers
   (Triton's masked `tl.load`) are safe: a false-predicate lane performs no
   access under either schedule.
2. **Single exit** — the latch is the only way out, so every entering lane
   passes the reconvergence point; the preheader must branch
   unconditionally (what `scf.while` lowering produces), so the captured
   `activemask` names exactly the lanes that will reach the sync.
3. **Live-out freezing** — every loop-carried value used after the loop is
   *frozen* on lane-inactive iterations: updated only through
   `select(pred, x, old)` or masked-identity forms
   (`old + select(pred, x, 0)`, also `|`, `^`) — precisely how Triton's
   `tl.where(active, ...)` idioms lower. The analysis is
   projection-aware: it follows the per-lane scalars through the
   struct-typed block arguments (extractvalue/insertvalue) of the real
   lowering.
4. **Predicate monotonicity** — once a lane's predicate turns false it can
   never turn true again under continued lock-step execution (otherwise
   lock-step would resume a lane that retirement has already lost): every
   icmp operand is frozen/invariant, or a constant-step induction moving
   away from re-satisfying an ordered comparison (`j += c, c >= 0` against
   `slt`; the mirrored cases likewise; `eq`/`ne` only with frozen
   operands).

Condition 4 is easy to miss: live-out freezing alone is *not* sufficient,
because an unmasked induction (`j += 1`) keeps moving on a lock-step
inactive lane and could re-arm a non-monotone predicate. To our knowledge
this condition has not been articulated for this class of rewrites.

The verifier already earns its keep in the tests: it refuses a
`tl.sum`-style latch loop whose induction variable is stored after the
loop (under lock-step the stored value is the warp-max-coupled one), and
refuses the two-moving-operands predicate.

## Testing

- `test/Conversion/nv_per_lane_loop_retirement.mlir`: three positive cases
  (plain scalars; an `add`-reduction latch; the struct-carried shape the
  real lowering produces) and six negative cases (collective in body,
  unfrozen live-out, non-monotone predicate, min-redux, partial mask, side
  exit), FileCheck'd including the verifier remark.
- Full local lit suite passes.
- Standalone e2e bench (in the PR): compiles the kernel both ways via the
  launch option, asserts bit-exact output vs. an exact oracle and PTX
  `redux.sync` 1 -> 0.

## Limitations / open questions

- The monotonicity argument assumes the induction step does not overflow
  within the loop's lock-step lifetime (a program hitting that would
  iterate ~2^31 times).
- Latches whose per-lane condition is not an `icmp` (e.g. an `and` of two
  predicates) are refused; extending the monotonicity reasoning to boolean
  combinations is mechanical follow-up work.
- `vote.sync any`-shaped latches (should Triton ever emit them) are an easy
  additional match head.
- The win is an *instruction-issue* win, largest for control-bound loops
  with divergent trip counts; the measurements quantify both ends.

Feedback wanted on: the pipeline position (post-CSE in `make_llir`), the
option-vs-knob surface, and whether there is interest in a default-on path
once the verifier has soaked.
