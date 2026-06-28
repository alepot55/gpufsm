# Landmark plan — from "automata study" to "the irregular-workload gap in tile DSLs"

User mandate (2026-06-28): aim for **super-high tier** (ASPLOS/PLDI/MICRO/OOPSLA landmark), publication
certainty, and an NVIDIA-interview-grade calling card. Think much bigger; do not stop until excellent;
literature must be airtight. This doc is the new north star. Validated by a 4-agent verified literature
+ strategy sweep (2026-06-28) — see findings inline.

## The re-framing (the big idea)
NOT "Triton is slow on automata." The open problem the whole field is circling — and NVIDIA is betting
the platform on (cuTile / Tile IR) — is: **tile/SPMD GPU DSLs only handle regular dense tensors; on
irregular, data-dependent control flow they collapse, and no one has a principled, automatic fix.**

The landmark contribution = **principle + working compiler pass + predictive law**, generalized beyond
automata:
1. **The principle (general, falsifiable):** a DSL's *abstraction regret* on irregular GPU work is
   governed by the **execution paradigm (thread-SIMT vs tile-SPMD), not abstraction height** — proven
   by holding the MLIR stack fixed (Triton↔Gluon) and the language height fixed (Warp↔CUDA): the 2×2.
   Mechanism: lock-step tiles forfeit intra-warp memory-level parallelism on dependent-load-bound work.
2. **A working AUTOMATIC compiler transformation:** detect irregular data-dependent regions in a tile-DSL
   kernel and lower them to per-thread ("thread-mode") execution within the tile DSL — recovering the
   thread model's intra-warp latency hiding — leaving the regular part tiled. THIS IS THE MAKE-OR-BREAK
   (characterization alone = IISWC, not ASPLOS).
3. **A predictive "regret law":** regret tracks a measurable a-priori predictor, validated by a CORRECT
   NEGATIVE. ⚠️ REFINED by the hash-probe witness (P1, 2026-06-28): the predictor is NOT "dependent-load
   count" (FALSIFIED — hash-probe regret is flat ~1.4× while probe length grows 35×). The right
   predictor is the **tile's issue-activity deficit = how much per-element SCALAR CONTROL / control-flow
   divergence the lock-step tile must serialize**. Nsight: tile `tl.load` gather already gives full
   intra-warp MLP (32-wide), so hash-probe (clean gather, light control) keeps tile issue ≈ thread
   (48% vs 49%) → small regret; automata (heavy ffs/while/register recurrence) starves tile issue
   (9.9% vs 41%) → large regret. Two sub-mechanisms: (a) **latency starvation** (heavy scalar control →
   low issue), (b) **masked-lane waste** (divergent trip counts → tile does 32-wide gathers vs thread's
   active-only: thread_inst/inst 32 vs 3.65). SpMV (aligned gather, no divergence) → predict ~1× regret.
4. **Generality across a workload suite** + multi-GPU (≥2 arch; ideally H100/Blackwell).

## Novelty boundary (verified; defend precisely)
OPEN (claim): automatic *detection* + *region-granular* tile→thread *lowering* inside a tile DSL,
*cost-model-selected*, *MLP-driven*, on irregular workloads.
NOT open (do NOT claim): "first to notice tile is bad at irregular" (NVIDIA cuTile concedes it; manual
SIMT fallback); "first mixed thread/tile in one program" (Prism PLDI'26 typed perspectives — MANUAL;
TLX, Gluon, ISPC — manual); "first auto scalar-vs-vector region selection" (Partial-CFG-Linearization /
Region Vectorizer PLDI'18 — CPU, *opposite direction* = vectorize, ours = de-vectorize to recover MLP).
Closest prior art to differentiate hard, on page 1: **Prism (PLDI'26)**, **NVIDIA cuTile SIMT-fallback**,
**Tawa (CGO'26 — automatic in-Triton region restructuring but dense warp-specialization)**,
**Partial-CFG-Linearization (PLDI'18)**, **ISPC uniform/varying (InPar'12)**, **Dynamic Warp
Subdivision (ISCA'10 — our mechanism, in HW)**.

## Must-cite (verified IDs)
cuTile/Tile IR (NVIDIA, CUDA 13.1, 2025; github.com/NVIDIA/cuda-tile); Prism (PLDI'26, 10.1145/3808290,
arXiv:2511.11939); Tawa (CGO'26, arXiv:2510.14719); TLX (arXiv:2605.10905); Partial-CFG-Linearization
(Moll & Hack, PLDI'18, 10.1145/3192366.3192413); Whole-Function Vectorization (Karrenberg & Hack, CGO'11,
10.1109/CGO.2011.5764682); ISPC (Pharr & Mark, InPar'12, 10.1109/InPar.2012.6339601); Convergence &
Scalarization (Lee et al., CGO'13, 10.1109/CGO.2013.6494995); Dynamic Warp Subdivision (Meng et al.,
ISCA'10, 10.1145/1816038.1815992); Dynamic Warp Formation (Fung, MICRO'07, 10.1109/MICRO.2007.12); Volta
ITS whitepaper; Linear Layouts (arXiv:2505.23819); Hidet (ASPLOS'23, 10.1145/3575693.3575702); Graphene
(ASPLOS'23, 10.1145/3582016.3582018). Workloads: Gunrock (PPoPP'16, TOPC'17 10.1145/3108140), GAP
(arXiv:1508.03619), Rodinia (IISWC'09, 10.1109/IISWC.2009.5306797), SuiteSparse (TOMS'11), ngAP
(ASPLOS'24, 10.1145/3617232.3624848), HybridSA (OOPSLA'24, 10.1145/3689771).

## Integrity fixes required NOW (flagged by agents)
- **Gluon claim:** Gluon CAN express per-thread *layout/arithmetic* (Linear Layouts, issue #8580); it
  CANNOT express the *scalar element load* a data-dependent CSR loop needs. Reword every "Gluon can't do
  per-thread" → "Gluon exposes per-thread layout but not the scalar load for data-dependent control flow;
  and provides no automatic irregular-region lowering." (Our probe is still valid; the phrasing was too
  broad.)
- **Cost model:** our fit is predictive for CUDA (2.7% holdout) but NOT Triton (45%). Do NOT headline
  "predictive cost model." Rename to "mechanistic regret decomposition / attribution"; the regret LAW
  (§3, predictor-vs-measured across the suite) is the predictive claim, built fresh with enough points.
- Do not claim "first to show tile is wrong for irregular" — frame NVIDIA's fallback note as motivation.

## Execution program (phases)
- **P1 — GENERALITY (autonomous, START NOW; biggest upgrade I can do alone):** implement, in the
  tile-vs-thread framing with the M10 nvcc-lowering machinery + a Triton tile kernel + CPU oracle, the
  non-automata witnesses and build the regret law:
    - **hash / B-tree probe** (cleanest non-automata dependent-load witness),
    - **rejection sampling** (control-flow face, memory held constant),
    - **SpMV (CSR, variable nnz)** = the predicted NEGATIVE control (bandwidth-bound → low regret),
    - **BFS/SSSP worklist** (reviewer credibility; GAP/SuiteSparse inputs).
  Per workload: a-priori predictor (Nsight long-scoreboard% / MLP from the thread kernel) + measured
  tile-vs-thread regret; oracle-gate; CSVs in paper2/data/landmark/. Then the regret-law figure.
- **P2 — the REAL compiler pass (autonomous, hard):** retry the Triton-from-source build with the
  diagnosed cmake fix; if it builds, prototype a TritonGPU `thread_region` op + lowering (Approach B:
  force sizePerThread=1 per-lane region, disable pipelining, reconverge at exit via ITS) + an automatic
  cost-model selector. If the build stays infeasible, deliver the *automatic selector* over the M10
  lowering as a strong proxy + the IR design, and be honest.
- **P3 — multi-GPU (needs USER):** cloud A100/H100 cross-arch (the regret follows the paradigm column,
  arch-independent). Prep a one-command script; the user provides access.
- **Throughout:** keep correctness-gated, every number CSV-traced, Nsight mechanism, honest corrections.

## Venue calibration (verified bar)
Top venues want a WORKING artifact + a principle + a surprising general result + ≥2 GPUs + ≥5–10
workloads + AE artifact. ngAP won best paper with 20 apps + a real execution model + reproduced artifact.
Target: ASPLOS/PLDI/OOPSLA if P2 (the pass) lands; CGO if partial; IISWC/PACT as the honest floor if the
pass isn't real by deadline. NVIDIA signal: frame contribution (2) as a Triton-MLIR pass, go down to
PTX/SASS, name the missing IR primitive, discuss Hopper/Blackwell + warp-specialization interaction.
