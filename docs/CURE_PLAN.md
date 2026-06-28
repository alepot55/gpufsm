# Paper 2 — "From Diagnosis to Cure": closing the tile-SPMD abstraction regret on irregular automata

Research plan. Branch `claude/cure-ir-primitive`. Hardware: RTX 4070 (sm_89), Triton 3.5.1,
torch 2.9.1+cu128, CUDA toolkit 13.3 / driver 580. Started 2026-06-28.

This is a **distinct, non-overlapping** second paper to the HPEC "Two Faces of Abstraction
Regret" study (paper 1 = *diagnosis*). Paper 2 = *cure*: name the missing tile-DSL primitive
precisely, and demonstrate that supplying it **closes the measured regret**, on a fixed
algorithm. Different community (compilers/PL: CGO/CC/PACT, stretch PLDI/ASPLOS) → no dual
submission, no salami-slicing. Paper 1 is cited as the prior characterization.

## 1. The precise gap (what paper 1 left open)

Paper 1's strongest sub-result is *uncomfortable* and is the seed of paper 2:

> Triton **can already express** the work-efficient NFA worklist (via `libdevice.ffs` + a
> data-dependent `while`/`for` over loaded CSR bounds + a register-resident `int64` bitset),
> yet it still pays **~6.5× throughput regret vs hand-CUDA** on that exact kernel — about the
> same as the 6–8× on the naive full scan.

So the regret is **not** "Triton can't express the algorithm" (it can) and **not** "Triton
wasn't given a layout knob" (Gluon adds layout control and still can't even express it — the
control-flow wall is elsewhere). The regret survives *expressibility*. That means there is a
**residual primitive** the tile/SPMD model lacks that, if supplied, would close the gap. Paper
2's whole job is to (a) identify it causally and (b) prove the cure.

## 2. Hypotheses for the residual 6.5× (to be tested, not assumed)

The NFA worklist is **embarrassingly parallel across strings** and **entirely scalar within a
string** (one `int64` state word, scalar bit-twiddling, data-dependent loops). Candidate causes
of the tile-SPMD penalty, each independently measurable:

- **H1 — warp-uniform scalar waste (the prime suspect).** A Triton *program* is a CTA of
  `num_warps×32` threads. An all-scalar kernel body has no tile dimension, so the scalar work is
  executed **warp-uniformly**: ~1 of 32 lanes does useful work, the rest replicate it. CUDA maps
  **1 thread = 1 string**, so a block of 256 threads runs 256 independent automata. Predicted
  signature: Triton worklist has low *warp execution efficiency* / *achieved occupancy of useful
  work* and is issue/latency-bound, not memory-bound. The tile/SPMD model has **no "scalar
  program" (thread-per-program) construct** — this is the candidate missing primitive.
- **H2 — int64 emulation.** sm_89 emulates 64-bit integer ops with multiple 32-bit ops. Both
  Triton and CUDA pay this, but Triton's codegen may amplify it. Test: ≤32-state automata with an
  `int32` bitset (no emulation) — does the regret shrink?
- **H3 — codegen / register-residency.** Triton may spill the scalar working set or emit weaker
  scalar code than nvcc. Test: SASS/register inspection + Nsight register & spill counters.

The cure target is whichever of H1–H3 carries the regret (working hypothesis: H1 dominates).

## 3. The cure, in escalating rigor (each a falsifiable milestone)

**M0 — Reproduce the anchor (this session).** Self-contained micro-benchmark (Triton scalar
worklist vs hand-CUDA worklist via `torch.utils.cpp_extension.load_inline`) reproducing the
~6.5× on *this* machine. This is the number every later milestone must move. No gpufsm build
dependency. → `experiments/cure/m0_anchor.py`.

**M1 — Decompose the regret (H1/H2/H3).** Attribute the 6.5× to its causes with controlled
micro-kernels + Nsight (warp execution efficiency, occupancy, DRAM%, int op mix, spills). Output:
a causal breakdown ("X% warp-uniform waste, Y% int64, Z% codegen"). → `m1_decompose.py`.

**M2 — Cure prototype WITHOUT a compiler rebuild (the 2-week falsifiable milestone).** If H1
dominates, the missing primitive is *lane-level task parallelism* (pack P independent scalar
automata onto the P lanes of one Triton program, each lane an independent string, per-lane
divergent control flow emulated via a uniform max-trip loop + masking). Express it in Triton
*today* as a `[BLOCK]`-shaped tile of states. Two outcomes, both publishable:
  - **closes the gap** → the primitive is *latent*; the regret is a missing **front-end
    affordance** + a uniform-control-flow tax we quantify.
  - **masking overhead eats it** → per-lane independent control flow genuinely needs **compiler
    support** (thread-per-program lowering) → motivates M3, the real cure.
  Also test `tl.inline_asm_elementwise` (PTX escape hatch) to bound the H2/H3 contribution.
  → `m2_lane_packed.py`, `m2_inline_ptx.py`.

**M3 — The constructive compiler cure (the high-ceiling escalation).** Add the named primitive as
a first-class construct in the Triton MLIR stack (a "scalar/lane program" lowering, or a
`tl.scalar_program` / per-lane `tl.serial_range`), and show the regret closing with the *front
end unchanged from idiomatic Triton*. This is the landmark "we fixed it" contribution. Gated on
M2's outcome; high effort (build Triton from source, IR op + lowering). Fallback if M3 is too
costly within the timeline: the **breadth paper** (regret-is-paradigm across BFS/SpMV) is the
safe second-paper, already pre-scoped.

**M4 — Generalize + write up.** Show the cure transfers to the DFA gather kernel (paper 1's
second face) and to ≥1 more irregular pattern. Related work positioned in the compiler community
(Triton/MLIR, Mojo, JAX/Pallas, Hexcute, Tawa, Descend, cuTile). Artifact (one-command repro).

## 4. Success criteria (falsifiable, decided up front)

- **M0 pass:** reproduce ≥4× Triton/CUDA worklist regret on this machine (anchor exists).
- **M1 pass:** ≥1 hypothesis accounts for >half the regret with a measured Nsight signature.
- **M2 pass (strong):** a prototype recovers ≥2× of the regret (e.g. 6.5× → ≤3.3×). **Null
  result is also a result:** "lane-packing cannot recover it in current Triton" *is* the
  motivation for M3 and is reported honestly.
- **M3 pass:** idiomatic-Triton front end + new primitive lands within ~2× of CUDA.
- Every kernel validated bit-for-bit vs the `reference.py` oracle before any speed number counts.

## 5. Method discipline (non-negotiable, mirrors paper 1)

- Correctness gates speed: no throughput number is reported for a kernel that hasn't matched the
  oracle on the full suite. Median + CI95 over warmup+repeats (timing is non-gaussian).
- Skeptical-scientist rule: negative/partial results documented as knowledge, not buried.
- Every cited number traces to a versioned CSV under `paper2/data/`.
- Ruff/format/mypy/pytest CPU-only must stay green (CI parity, see CLAUDE.md §6).

## 6. Risk register

- **R1: M3 (Triton-from-source IR work) overruns the timeline.** Mitigation: M2 alone is a
  CGO/CC-grade "latent primitive" paper; M3 is upside. Breadth paper is the floor.
- **R2: H1 wrong — regret is int64/codegen not warp-waste.** Then the cure is inline-PTX/int32
  packing, a different but still constructive paper. M1 decides before we commit to M2/M3.
- **R3: lane-packed divergence makes correctness hard.** Mitigation: oracle gate + start at small
  state counts (≤32, int32) where it's tractable, then scale.
