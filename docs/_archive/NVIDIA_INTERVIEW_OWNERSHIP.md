# Ownership & Whiteboard Derivation — the tile-vs-thread regret and its cure

Purpose: this is the **own-every-line** companion to paper 2. It is written so the work can be
reconstructed cold — mechanism re-derived from first principles, the MLIR pass re-written from memory,
and the warp-scheduling / ITS / PTX–SASS reasoning defended at a whiteboard. Every number here traces to
a versioned CSV (cited inline); nothing is quoted from a slide.

If you can reproduce the four derivations in §3 on a whiteboard and rewrite the pass in §4 from the
recipe, you own this work. That is the bar.

---

## 1. The one-sentence claim (say this first)

> On a tile DSL (Triton), the cost of an irregular workload is not set by its arithmetic but by how the
> tile abstraction is forced to *issue instructions* across lanes that should have retired. The thread
> (SIMT) model can retire a lane the instant its data-dependent loop ends; the tile model cannot express
> that, so it keeps issuing masked full-width work. The sign and size of the resulting "abstraction
> regret" follow a single instruction-issue accounting rule.

That is the whole paper in three sentences: a *law* (when/why tile loses or wins), a *diagnosis* (the
exact IR reason it can't be fixed inside TritonGPU), and a *cure* (name + build the missing primitive).

---

## 2. The mechanism, derived (not asserted)

Two facts about the hardware:

1. A warp issues **one instruction for up to 32 lanes** per step. Lanes masked off still occupy the slot
   — they cost issue bandwidth, not just nothing.
2. Independent Thread Scheduling (ITS, since Volta) lets lanes within a warp follow **divergent program
   counters** and **retire independently**; the warp reconverges at an explicit `bar.warp.sync` (or the
   compiler's reconvergence point). A lane whose data-dependent loop has ended simply stops being
   scheduled.

Define, per workload, **threads-per-instruction (TIPI)** = average number of *active* lanes at each
issued instruction. A perfectly converged step has TIPI = 32; a maximally divergent step approaches 1.

- **Thread model:** when lanes diverge, the warp issues an instruction for the *subset still live*, then
  retires the rest. TIPI drops below 32 but **the retired lanes stop consuming issue slots**. Total
  issued instructions ≈ (work of the longest-running lane).
- **Tile model:** a Triton program is one SPMD instance over a `#blocked` tile. Divergence is expressed
  only by **masking** — every lane keeps issuing every instruction of the *union* of all lanes' control
  paths, predicated off where inactive. TIPI stays ≈ 32 but the **instruction count is the union**, not
  the longest lane.

So the regret is a ratio of issued instructions:

```
regret  ≈  (tile issued insns) / (thread issued insns)
        ≈  (union of all lanes' steps, full width)
           ----------------------------------------
           (longest lane's steps, narrowing as lanes retire)
```

**The sign flip — the part interviewers probe.** The numerator/denominator comparison happens *per step*.
What matters is the **instruction cost of one step of the divergent loop body**:

- If a step is **scalar control** (compare, increment a per-lane counter, chase one pointer — automata,
  MoE routing, rejection sampling), the tile must issue that scalar op at full tile width while the thread
  issues it only for live lanes. The tile issues **more** per step ⇒ **tile loses** (regret > 1).
- If a step is a **dense, vectorizable reduction over a contiguous dim** (the head-dim dot-product +
  online-softmax of attention), the tile lowers it to a few wide vector/MMA instructions, while the thread
  model issues the loop scalarly per lane. The tile issues **fewer** ⇒ **tile WINS** (regret < 1).

This single rule predicts the whole table (`paper2/data/landmark/regret_law.csv`):

| witness | regret | tile TIPI | thread TIPI | why (one line) |
|---|---|---|---|---|
| pointer_chase | 1.00 | 32.0 | 32.0 | fixed trip, no divergence — **negative control**, must be 1.00 |
| spmv_uniform | 1.94 | 32.0 | 32.0 | occupancy 50 vs 94%, not divergence — baseline gap |
| hashprobe | 1.40 | 32.0 | 3.65 | divergent probes; thread retires, gather dilutes the win |
| spmv_powerlaw | 2.17 | 32.0 | 32.0 | baseline + row-length divergence (cv 3.79) |
| automata_nfa | 1.96 | 32.0 | 30.34 | divergent active-set, **scalar control** → latency starvation |
| rejection | 4.00 | 32.0 | 17.12 | divergent trip counts (max/mean ≈ 5), masked pure compute |
| moe_powerlaw | 2.36 | 30.10 | 7.71 | ragged expert counts, **scalar per-step** → tile loses |
| attention_powerlaw | **0.64** | 32.0 | 3.45 | **dense vectorizable head-dim → tile WINS** |

The one sentence that ties it to the audience: *this is why Triton excels at flash-attention yet
collapses on automata* — same compiler, opposite sign, and the sign is set by whether the divergent step
is dense-vector or scalar-control.

> Honesty note you must volunteer (it is a strength, not a weakness): two early numbers were corrected
> under scrutiny. (a) A naïve "global-max bounded-for" A/B for the hoist measured **0.45×** — *slower* —
> because it deleted per-warp early termination; the correct per-warp-max hoist is **1.55×**. (b) An
> Nsight run without `--kernel-name` profiled a torch *setup* kernel and reported a bogus TIPI; fixed by
> always pinning `--kernel-name regex:<fn>`. Saying this unprompted demonstrates the measurement rigor an
> interviewer is actually testing for.

---

## 3. Four whiteboard derivations (be able to do each in ~2 minutes)

**(A) Why the negative control must be exactly 1.00.** Pointer-chase has a *fixed* trip count and no
data-dependent termination: every lane runs the same number of steps. No lane ever retires early, so the
thread model's "retire" advantage is null and TIPI = 32 on both sides. Predict regret = 1.00 *before*
running it; the CSV says 1.00. A law that didn't force this would be unfalsifiable — this row is what
makes the others mean something.

**(B) Bounding the rejection-sampling regret.** Trip counts are geometric; max/mean ≈ 5 across a warp.
Tile issues the loop body `max` times for all 32 lanes; thread issues it `~mean` times averaged, retiring
finished lanes. First-order regret ≈ max/mean ≈ 5; measured 4.00 (the gap is the fixed prologue/epilogue
that doesn't diverge). You should be able to say "≈5, slightly less because of the non-divergent
remainder" without the CSV in front of you.

**(C) The full-cure ceiling = 5.64×.** Take the lock-step automata-style kernel. Lower the *same source*
three ways (`paper2/data/landmark/f3_hoist_rtx4070.csv`):

| lowering | time | speedup | thread TIPI |
|---|---|---|---|
| tile baseline | 155.6 µs | 1.00× | 32.0 |
| reduce-hoist (built, in libtriton) | 100.4 µs | 1.55× | 32.0 |
| thread / per-lane retiring (nvcc) | 27.6 µs | **5.64×** | **11.65** |

Read it as: the *total* abstraction regret on this kernel is 5.64×. The reduce-hoist captures 1.55× of it
*without leaving tile IR* (it only removes the per-iteration cross-lane reduce). The residual 5.64/1.55 ≈
**3.6×** is **exactly** the per-lane sub-warp retirement — TIPI falling from 32 to 11.65 — and that is the
part TritonGPU structurally cannot express. Memorize: **1.55 in-IR, 3.6 below-IR, 5.6 total.**

**(D) Why 1.55× is provably correct, not a heuristic.** See §4 — the rewrite is an equivalence, and the
oracle `acc[i] = trip[i]·(trip[i]−1)/2` is bit-identical across all three modes (off / detect / hoist).

---

## 4. Rewrite the pass from this recipe (the part you must own line-for-line)

The pass is `tritongpu-thread-region` (`ThreadRegion.cpp`, in `~/m3full_build/triton-src` and mirrored at
`experiments/cure/triton_thread_region_pass/`). Two modes: `GPUFSM_THREAD_REGION=1` detects;
`=hoist` rewrites.

**What it matches (the lock-step signature).** An `scf.while` that (i) carries at least one `#blocked`
tensor iter-arg, and (ii) whose `scf.condition` is derived from a `tt.reduce` of a per-lane predicate
(`cmpi slt %j, %trip`). That is the IR fingerprint of "all lanes loop until the *last* lane is done" —
the masked, no-early-exit pattern.

**The reduce-hoist rewrite, in words (then write the C++):**

1. In the `before` region, find the `tt.reduce` feeding the condition, and the `arith.cmpi slt` feeding
   the reduce. Identify `%j` (a `before`-block argument — the per-lane counter) and `%trip`
   (loop-invariant: its defining op is *not* inside the while).
2. **Hoist** `%mt = reduce_max(%trip)` once, *before* the loop, by **cloning the matched `tt.reduce`** with
   an `IRMapping` that maps its input to `%trip` directly. (Clone, don't rebuild — you inherit the exact
   combine region and layout.)
3. Rebuild the `scf.while` with **one extra `i32` iter-arg** `js` (a scalar counter), initialized to 0.
4. New `before`: clone the old body, then replace the condition with the **scalar** `arith.cmpi slt %js,
   %mt`; forward the original condition args plus `js`.
5. New `after`: clone the old body, append `js' = js + 1`, yield the originals plus `js'`.
6. `replaceAllUsesWith(nw.getResults().take_front(oldNumResults))`; erase the old while.

**Why it is equivalent (one breath):** `%j` is a uniform splat (same value in every lane), so the *body*
is already correctly masked by `j < trip` per lane. The cross-lane `tt.reduce` only computed the *loop
bound* = `max(trip)`. `max(trip)` is loop-invariant, so computing it once up front and counting a scalar
to it visits **the same iteration count** and keeps **per-warp termination** — it only deletes the
redundant per-iteration reduction. Hence bit-identical output, 1.55× from removing the reduce.

**MLIR API gotchas you will hit (this LLVM treats deprecation as error):**
- `OpTy::create(builder, loc, ...)` — **not** `builder.create<OpTy>(...)`.
- `arith::ConstantOp::create(b, loc, b.getIntegerAttr(ty, 0))` — **not** `ConstantIntOp(int, Type)`.
- Region cloning is `IRMapping` + `builder.clone(op, mapping)`; map block args first via
  `addArgument`, then clone `without_terminator()`, then build the new terminator.

**Why you cannot finish the cure inside TritonGPU (the wall — say this precisely):** per-lane termination
needs a per-lane branch. `scf.condition` takes a **single `i1`**, not a per-lane vector of `i1`; and the
`#blocked` tensors are already `sizePerThread = 1`, so there is no sub-tile structure left to peel. The
only place a per-lane `cond_br` can be created is **below** TritonGPU, in the `make_llir` stage
(`add_scf_to_cf` → `add_to_llvmir` in `TritonNVIDIAGPUToLLVM`), where you can (a) extract the per-lane
scalar from the distributed layout, (b) emit a per-lane `cond_br` that ITS retires, (c) reconverge with
`bar.warp.sync`, and (d) opt the divergent loop out of the pipeliner. That coherent-tile-state-across-a-
divergent-loop problem is the genuine multi-week crux — it is *named and scoped*
(`docs/rfc/below-tritongpu-lowering.md`), deliberately not faked.

---

## 5. Warp-scheduling / ITS / PTX–SASS cheat-sheet (defend these cold)

- **Issue model:** SM sub-partition issues 1 instr/cycle from a selected warp; a stalled/masked lane does
  not free the slot for another lane. ⇒ masked full-width work *is* wasted issue bandwidth (the regret).
- **ITS (Volta+):** per-thread PC + call stack; lanes can be at different PCs; `@!p bra` lets a subset
  branch. Reconvergence is **not** automatic at every join — the compiler inserts it (`bar.warp.sync`
  / `BSSY`/`BSYNC` in SASS). This is *why* the thread model can retire lanes — and why the tile model,
  which never emits per-lane `bra`, cannot.
- **PTX vs SASS:** PTX is virtual ISA (predication shown as `@p`); SASS is the real schedule. In SASS
  you'd point to `BSSY`/`BSYNC` (reconvergence barriers) and per-lane predicate guards to *show* a lane
  retired. The tile lowering shows full-width predicated ops with no early `BRA` out of the loop.
- **`tl.max`/`tt.reduce`:** a cross-lane (intra-CTA) reduction → shuffle (`shfl.sync`) tree, not free; per
  *iteration* it is the cost the hoist removes.
- **Occupancy vs divergence — don't conflate.** spmv_uniform's 1.94× is an *occupancy* gap (50 vs 94%),
  TIPI = 32 on both sides — that is NOT regret in our sense. Only rows where **thread TIPI < 32** (hash
  3.65, rejection 17.12, automata 30.34, moe 7.71, attention 3.45) are paying the divergence/retirement
  mechanism. Being able to separate these two is the tell of someone who measured rather than hand-waved.

---

## 6. Questions to expect, and the crisp answer

- *"Isn't this just predication overhead?"* No — predication keeps a lane *issuing*; the point is the
  thread model lets it *stop issuing*. The cost is issued-instruction count, not predicate evaluation.
- *"Why not just bound the loop by the global max?"* That deletes per-warp early termination and is
  *slower* (0.45× — measured). The fix must keep per-warp `max`, which is the reduce-hoist (1.55×).
- *"Doesn't cuTile / CUDA Tile IR solve this?"* No. cuTile and its Tile IR (now being built as a Triton
  backend, CUDA 13.1) are tile-level and concede the irregular case to a hand-written SIMT fallback — so
  the per-lane gap persists in *both* TritonGPU and NVIDIA's Tile IR. The primitive is **complementary**
  to their roadmap, not subsumed.
- *"What would you build first at NVIDIA?"* The below-TritonGPU per-lane retirement op (§4 wall): the
  3.6× residual is sitting on the table, the hook-point is pinned, and the blocker (per-lane scalar
  extraction from a distributed layout + coherent tile state across a divergent loop) is exactly the kind
  of compiler problem the role exists to solve.

---

*Traceability:* every figure here comes from `paper2/data/landmark/regret_law.csv` and
`paper2/data/landmark/f3_hoist_rtx4070.csv`; the pass from `ThreadRegion.cpp`; the wall + hook-point from
`docs/rfc/below-tritongpu-lowering.md` and the RFC `docs/rfc/triton-per-lane-region.md`. No number in this
document is unsourced.
