# Paper 2 — Related Work, Novelty Positioning & De-risking

Synthesis of a 4-axis literature sweep (2026-06-28, parallel agents; every ID verified on
publisher/arXiv pages, uncertainties flagged). Companion to `docs/LITERATURE_REVIEW.md` (paper-1,
automata + abstraction-cost). This doc covers the axes specific to paper 2: GPU execution-model /
latency-hiding, tile-DSL design space, and the venue/threat strategy.

> UPDATE (M10): the cure is no longer just a design proposal — its lowering is IMPLEMENTED and
> measured (the same per-lane source → thread-model CUDA runs 4.2× the tile lowering, matches/exceeds
> hand-CUDA; `experiments/cure/m10_scalar_program.py`). This moves the contribution from
> "characterization + design" toward "characterization + diagnosis + demonstrated cure", which raises
> the strongest desk-reject risk's bar (the cure is shown to work, not just argued). The two
> distinctions below (layout≠control-flow; not-just-autotuning) still anchor the novelty.

## 1. Verdict: the niche is EMPTY, novelty holds — but on two sharp distinctions

Confirmed across all four sweeps: **no work studies the GPU execution-paradigm / DSL-expressibility
cost on irregular automata at a fixed algorithm.** All GPU automata engines are single-implementation
CUDA on an *algorithmic* axis; all 2024–26 tile-DSL papers target *regular* tensor work. Our survival
depends on holding two distinctions precisely (§4).

## 2. Mechanism grounding (the §4.4 intra-warp claim) — MUST cite

The residual is **MLP (memory-level parallelism) denied by the abstraction**, not occupancy.
- **Volkov, "Understanding Latency Hiding on GPUs", PhD thesis, UC Berkeley UCB/EECS-2016-143 (2016)**
  + the origin talk "Better Performance at Lower Occupancy" (GTC 2010). PRIMARY anchor: latency is
  hidden by in-flight independent requests (ILP/MLP), not only occupancy (TLP). Without this a
  reviewer dismisses our finding as "textbook occupancy."
- **Hong & Kim, "An Analytical Model … with MLP and TLP Awareness", ISCA 2009, 10.1145/1555754.1555775.**
  MWP/CWP formalism → Little's law currency for "overlapping independent loads hides latency."
- **Ding & Williams, "An Instruction Roofline Model for GPUs", PMBS@SC 2019 / Concurrency&Comp.P&E
  2022, 10.1002/cpe.6591.** Automata are integer/control-heavy → the *correct* roofline; lets us show
  Triton sits BELOW the issue ceiling at low GIPS despite *fewer* instructions = not instruction- nor
  bandwidth-bound.
- **NVIDIA Volta whitepaper** (Independent Thread Scheduling: per-thread PC, but still one
  instruction/cycle/warp) + **Fung et al., "Dynamic Warp Formation", MICRO 2007, 10.1109/MICRO.2007.12**
  (shared-PC lock-step is the divergence-cost baseline). Grounds "one SPMD instruction stream couples
  the lanes."

⚠️ **RIGOR CORRECTION (apply to DRAFT):** under SIMT, execution is still one-instruction-per-cycle-
per-warp; ITS gives independent per-thread *PCs*, not independent *issue*. The independence that
hides latency is in the **memory pipeline** (independent outstanding loads / MLP), NOT the issue
stage. Reword "32 independent threads that hide latency" → "independent in-flight loads (MLP) across
the warp's lanes." A microarchitecture reviewer will catch the old phrasing.

⚠️ **Also distinguish from memory divergence:** our lanes can be perfectly coalesced/uniform and still
stall together because a *data-dependent* (next-state) load serializes the single SPMD stream. State
explicitly that this is NOT branch/memory divergence (Fung/Meng/MeDiC) or a reviewer mislabels it.

## 3. Tile-DSL design space (the cure / §6 primitive) — cite & differentiate

| System | Venue/ID | Model | Relation |
|---|---|---|---|
| Triton (Tillet) | MAPL'19, 10.1145/3315508.3329973 | tile/SPMD | the critiqued baseline |
| **Gluon + Linear Layouts** | arXiv:2505.23819 + triton repo | tile + per-thread *layout* | **CLOSEST RISK** — per-thread data placement, NOT per-lane control flow |
| TLX | arXiv:2605.10905 | tile + warp-group MIMW | coarser (warp-group), async/tensor-core; complementary |
| Tawa | CGO'26, arXiv:2510.14719 | tile→auto warp-specialize | regular async dataflow; orthogonal |
| TileLang | arXiv:2504.17577 | tile + thread binding | thread *binding/layout* for regular kernels |
| Hexcute | arXiv:2504.16214 | tile + auto layout | decomposes the *layout* gap (our neighbor), dense GEMM |
| cuTile / CUDA Tile IR | NVIDIA CUDA 13.1 | tile (MLIR) vs SIMT | industrial tile direction; gap unsolved by vendor |
| Graphene | ASPLOS'23, 10.1145/3582016.3582018 | tile IR, data↔thread map | dense tensor mapping |
| Descend | PLDI'24, 10.1145/3656411 | thread-hierarchical | thread-level first-class (no tile drop-to-scalar) |
| NVIDIA Warp | (no paper) | **thread/SIMT** | **our existence proof** |
| Taichi | SIGGRAPH Asia'19, 10.1145/3355089.3356506 | data-oriented megakernel | irregular/sparse thread-model precedent |

## 4. The two distinctions our paper lives on (adversarial rebuttals)

**Risk A — "you didn't tune Triton" (autotuning counter-thesis: Ringlein et al., arXiv:2505.03780).**
The most likely killer at a compiler venue. Rebuttal (we already have the evidence):
1. The **Triton↔Gluon same-MLIR-stack control** (paper 1's probe): identical backend, only the
   expressivity lever changes → a surviving gap is not a tuning gap.
2. The **disclosed `num_warps` sweep (M2f)**: component A (~3.4×) is explicitly *attributed to launch
   config*, and the residual C is flat across the sweep. We literally separate "tuning" from
   "expressibility." Cite Ringlein and state: autotuning closes A and part of B; C is expressibility-
   bound, with a falsifiable probe.

**Risk B — "Gluon already does per-thread" (the desk-reject on the cure).**
Gluon's Linear-Layout per-thread ops = **data placement / per-thread arithmetic** (which lane holds
which element). It does NOT give **independent per-lane control flow**: per-lane data-dependent loop
bounds, per-lane `while`, scalar gather driven by per-lane state — `gl.load` still returns a
layout-tensor (no scalar load; paper-1 probe). **Layout control ≠ control-flow divergence.** Our
`scalar_program` lowers a marked region to a per-lane independent instruction stream.

**Risk C — "you reinvented SIMT/CUDA."** The contribution is not per-thread execution; it is (i) a
*bounded, marked, composable* region inside the tile model (regular part stays tile-fused) and (ii)
the capability→cost characterization naming exactly which primitive is missing and how much it gates.
Warp/CUDA are existence proofs, not the same artifact.

**Risk D (microarch) — Subwarp Interleaving (Damani et al., HPCA 2022, 10.1109/HPCA53966.2022.00065).**
Closest mechanism work: intra-warp stalls under low occupancy, fixed in *hardware*. We attribute the
*same stall* to the *DSL abstraction* — same SM is fast under CUDA, slow under Triton at equal
occupancy and fewer instructions. Software attribution, not a hardware fix.

## 5. Automata related work (new since paper-1 sweep) — cite & distinguish

- **MLIR-regex→DSA (Conficconi et al.), CGO 2025, 10.1145/3696443.3708916** — the ONLY "IR-for-automata"
  work and the genuine threat to "nobody studies abstraction for automata." Differentiate prominently:
  they lower regex to ONE fixed domain-specific accelerator; they do NOT compare GPU execution
  paradigms, do NOT study DSL expressibility, do NOT hold an algorithm fixed across programming models.
- **Hopps (Du, Emer, Sánchez), ASPLOS 2025, 10.1145/3676642.3736126** — sparsity-exploiting AP, but an
  ASIC accelerator (not GPU). Cite as sparsity frontier.
- Anchors reconfirmed: ngAP (ASPLOS'24, 10.1145/3617232.3624848 + TOCS 10.1145/3748646), HybridSA
  (OOPSLA'24, 10.1145/3689771 — DOI unconfirmed, verify), BitGen (MICRO'25, 10.1145/3725843.3756052),
  AsyncAP (POMACS'23, 10.1145/3579453), AutomataBLAS (TACO'25, 10.1145/3774656). STeP (ASPLOS'26) —
  unverified this pass, confirm before camera-ready.

## 6. Methodology + lineage citations (rigor shield)

Hoefler & Belli SC'15 (10.1145/2807591.2807644 — median+CI95); Roofline CACM'09; Instruction Roofline
(above); Pennycook PP-metric (arXiv:1611.07409 / FGCS'19 — frame as per-capability not per-hardware);
Halide PLDI'13; LMS "Abstraction without Regret" (CACM'12 — the term we invert); Nsight Compute
Profiling Guide (pin toolkit version). Precedents for "characterization+design, implementation as
future work": Jia et al. "Dissecting Volta" (arXiv:1804.06826), Volkov, Pennycook, Hoefler&Belli.

## 7. Venue strategy
- **Primary: CGO 2027 Round 2 (submit ~Sep 10, 2026), Standard Research Paper track** (NOT Tool Paper
  — that mandates an implementation we frame as future work). CGO explicitly welcomes "program
  characterization"; artifact optional but we submit it (oracle + CSVs + Nsight) for credibility.
- **Backstop: ACM TACO** (journal, rolling, tolerant of characterization+design; good CV value).
- IISWC is the best pure fit for the characterization half but 2026 round closed → target 2027.
- Stake priority on **arXiv now**; PMBS@SC as the methodology-friendly workshop for the term.

## 8. Prioritized expansions to make it bulletproof (→ feeds CURE_PROGRESS next actions)
1. [x] **MLP measured (M5, DONE) — confirmed, with one hypothesis corrected.** `m5_mlp_rtx4070.csv`:
   at matched occupancy WP2's `long_scoreboard` stall is **15.3× CUDA's** and it issues at **9.9% vs
   41%** — latency-bound on dependent loads, confirmed; issue-ratio 4.1× ≈ throughput-ratio 3.5×
   (Little's-law-consistent). CORRECTION: the predicted "Triton sustains FEWER in-flight requests"
   was WRONG — WP2 issues **26–84× MORE** memory requests (masked-lane gathers + int64) and still
   stalls. So the tile cost is *excess masked memory traffic + inability to hide dependent-load
   latency*, not a request-count deficit. (Phrase the mechanism via stall composition + issue rate,
   not request count.)
2. **Instruction Roofline placement** of both kernels (GIPS vs instruction intensity) — visual proof
   it's neither instruction- nor bandwidth-bound. Answers "did you just write a worse kernel?"
3. **Gluon prototype of `scalar_program`** (Linear Layout per-lane state + predicated ffs/while) — show
   it recovers a measurable fraction of CUDA/Warp; turns "we propose" into "we demonstrate realizable +
   show what plain Gluon can't." Directly neutralizes Risk B.
4. **Capability-generalization paragraph + table** (not new benchmarks): variable-length decode, ragged
   batching, BFS frontier all need the same per-lane escape — pre-empts "automata-only, narrow."
5. **Quantify the Little's-law crossover prediction** for the DFA/occupancy regime (we have the
   crossover; make it a predicted-vs-measured falsifier).
