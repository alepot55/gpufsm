# TACO paper — adversarial review findings + hardening status (2026-07-03)

Adversarial reviewer pass (skeptical TACO reviewer). Verdict: **Major revision**
(borderline reject as-was). The built pass + oracle-gating + Nsight + A100 are
above-bar substance; the issues below were the reject-risk. Status per item.

## Fixed tonight (integrity/honesty — the reject-risk items)
- **M1 [FIXED]** Flagship decomposition didn't multiply to its anchor. Abstract said
  "10x decomposes as 3.4x × B × 2", but 10.1x is at batch 4096 while 3.4x (component A)
  and Table 1's staged decomposition (=26x) are at batch 16384. Verified vs CSVs
  (m0_anchor=10.1x@4096; m2f_numwarps A=2.77x@4096, 3.44x@16384). Fix: abstract now
  states the regret is batch-dependent (10.1x@4096 → 26x@16384 as the occupancy-gated
  component B saturates); A quoted as a 2.8–3.7x range; no false product claim.
- **M2 [FIXED]** Headline "cure" 4.2x is transpile-to-CUDA (nvcc), and the generated
  kernel beats hand-CUDA ~2.15x (soft baseline). Abstract now frames 4.2x as an
  out-of-band existence proof; the in-compiler passes are the load-bearing contribution.
- **M4 [FIXED]** ML generality oversold (witnesses use one-element-per-lane mapping no
  production kernel uses). Abstract rescoped: the law holds on ML-shaped witnesses under
  that mapping; whether it survives production warp-cooperative mappings = future work.
- **M8 [FIXED]** Adjacent 0.51x (≈2x) vs 3.3x read as contradiction. Now labelled:
  0.51x is the median head-to-head across the ≤64-state sweep; 3.3x is the specific
  Nsight-profiled config (32 states, batch 16384).
- **De-anon [FIXED]** Removed the "proposed upstream to Triton + RFC + links redacted"
  fingerprint (anonymity is a hard TACO requirement); kept the verifier's technical
  content (monotonicity condition) phrased neutrally.

## Remaining (for the revise cycle — do before/at submission)
- **M5 [DONE — decomposition]** The flagship staged decomposition now runs on the A100
  (Modal, gpufsm CUDA ext built for sm_80). It reproduces to the SAME 26x total with matching
  stage factors: num_warps 3.7x on BOTH; lane-packing 2.3x(A100)/2.6x(4070); residual-to-thread
  3.1x/2.8x. The batch-4096 anchor rescales 10.1->5.6x (clock-sensitive register-resident regime;
  direction persists). Folded into Threats. Data: paper2/data/cross_arch/{m0_anchor,m2f_numwarps,
  m2_lane_packed,m2e_worklist_packed}_*a100*.csv. NOTE: 2 earlier Modal runs produced FAKE echoes
  (build failed: missing g++, then clang) — caught by a hardened capture guard, deleted, not
  committed. STILL OPEN under M5: the built CURE (4.15x pass) on A100 (needs from-source cure build).
- **M3 [PARTIAL]** In-compiler cure is 4.15x synthetic but 1.14–1.25x on real SpMV/MoE. Lead
  the compiler section with real-workload numbers; present 4.15x as a mechanism-isolating bound.
- **M6 [OPEN, may be intrinsic]** Clean result lives in ≤64-state register-resident toy regime;
  real automata (Brill 42661 states) run sub-Gbps/confounded. Show the built cure on some real
  automaton even if the residual is confounded.
- **M9 [OPEN]** Thin stats: no CI95/IQR on load-bearing numbers (4.15x, 15.3x, 39x are point
  estimates). Add dispersion, esp. on the decomposition factors and the anchor (3 seeds exist).
- **M7 / tone [OPEN]** Tone down over-naming ("straggler law", "the cure", 8 bolded contributions
  → consolidate to 3–4). State the paper1 delta crisply. Frame mechanism as localizing a known
  latency-hiding effect to the tile/thread boundary.
- **Abstract [OPEN]** Still a ~34-line firehose; tighten to ~200 words.

## Bottom line
The integrity holes that risked a reject are closed. Biggest remaining credibility lever = M5
(flagship results on a 2nd GPU). The rest is tightening + honest reframing, all revise-cycle work.


## CGO 2027 pivot (2026-07-08, after TACO desk-reject "too premature")
All remaining items CLOSED with recipe-reproducible data (pinned wheel: triton 3.8.0@81a46fa +
versioned perlane_retire_full.patch; built on Modal CPU, run on Modal A100 + local RTX 4070):
- **M5 [CLOSED]** The built cure now reproduces on the A100: lock-step 1.6-3.8x by distribution,
  same flat cured floor (~75us), SpMV/MoE ~1.0x, verifier declines the out-of-scope latch. Data:
  paper2/data/cross_arch/cure_wheel_a100.csv + cure_nvidia_a100-sxm4-40gb.log.
- **M9 [CLOSED]** Dispersion added: geometric-law cure median 2.46x IQR [2.46,2.50] n=5 seeds
  (cure_ci.py). Straggler law refit on the pinned recipe: t=32.4+1.09*E[warp-max], R2=0.997;
  floor 43.0+-1.6us; out-of-sample 1.5% mean / 2.1% max (was 5%/7.5%); single-straggler 7.2x.
- **M3 [CLOSED]** Honest real-workload framing: with the pinned recipe SpMV/MoE gains are ~1.0x
  (the old 1.14/1.25x did not reproduce); reframed as thesis-confirmation (recoverable regret
  scales with per-step control). The 4.15x headline replaced by the reproducible 2.3-6.7x range.
- **NEW (integrity)** The old local-build numbers (4.15x, 2.5-7.3x, SpMV 1.14x, MoE 1.25x,
  straggler fit 50.3+1.08) were NOT reproducible from the versioned recipe and were REPLACED
  in the paper by wheel-recipe numbers. Slope stability (1.08 vs 1.09) reported as build-stability.
- **NEW (soundness)** cure_rejection witness: the verifier correctly DECLINES the accumulate-OR
  early-exit latch on both GPUs (PTX redux stays 1) -> scope explicit, sound by default.
Paper: paper2/gpufsm_cgo.tex (sigplan, 10pp incl. refs vs 11pp text limit, 0 overfull).


## ASPLOS restructure + integrity audit (2026-08-21)

Rewrote `paper2/gpufsm_asplos.tex` for the September cycle. The presentation items below
were the point of the exercise; the integrity items were found on the way and matter more.

### The constraint that drove the restructure

The CFP runs a **rapid-review round that reads only the first two pages**. Those pages held
a 350-word abstract that spilled onto page 2, a bolded contribution list, and no figure.
They now hold the abstract, a teaser figure, the mechanism, the impossibility result, the
built cure with its numbers, the predictive law, and four contributions. M7 and the
"abstract firehose" item from the TACO pass are closed by this.

### Integrity items found (all fixed)

- **Cross-arch stage factors did not reproduce.** The paper quoted the ladder as matching
  across GPUs (4070 2.6x/2.8x vs A100 2.3x/3.1x). Those four numbers reproduce from no
  versioned CSV. `m2e` gives 3.0x/2.4x on the 4070 and 2.3x/3.0x on the A100, and the two
  halves used *different* lane-packed kernels (`wp2` on the 4070, `wp` on the A100), which
  is why they looked comparable and were not. Table 1 is now the same four kernels measured
  the same way on both devices. The surviving claim is weaker and true: total 26x vs 25x,
  launch stage identical, last two stages trade places.
- **"num_warps is 3.7x on both architectures"** held at batch 65536, not at the batch 16384
  the surrounding text specifies (3.44 and 3.05 there). Now given as the 2.8-3.7x range.
- **A sample-fragile statistic.** `corr(speedup, D) = 0.06` was computed over the six
  designed distributions; adding the four held-out ones raises it to **0.55**. Replaced by
  the comparison of fits, stable on both samples: R2 0.997/0.998 (straggler) against
  0.00/0.29 (divergence ratio). Conclusion unchanged, evidence no longer fragile.
- **The generality law contradicted its own floor.** It claimed regret is created by scalar
  control, then reported uniform-nnz SpMV at 1.94x with zero divergence. Now scoped to the
  regret that grows and that the cure recovers, with the occupancy floor as a separate
  component, present where the tile mapping costs occupancy and absent at matched occupancy
  (pointer-chase, 1.00x).
- **`t = 32.4 + 1.09x` has two parameters**, so "one-parameter model" was wrong, in three
  places and in the submission abstract. It is a single-*predictor* model.
- **Citation error**: `hopps2025` had Yufeng Du; the first author is **Xingran Du**
  (verified against the ACM DL record). Venue and year were right.
- **Out-of-sample max error** was quoted as 2.2%; the CSV says 2.1%.
- **NFA regret** quoted as 2.0x in the law section where the figure and CSV say 1.96x.
- **A format-compliance defect**: figures were generated 6in wide and included at
  `width=\columnwidth`, a 0.55x downscale putting axis labels near 5pt against the CFP's 8pt
  floor. `docs/SUBMISSION_ASPLOS.md` had passed this because it looked for `\resizebox`.
- **A de-anonymization risk**: the captured TTGIR embeds an absolute path containing the
  author's username. The new IR listing is stripped of every `loc(...)`.
- **A scope ambiguity I introduced**: reordering put the selector immediately after the
  in-compiler pass, so "routes the region to the thread lowering" read as if it routed to
  that pass. It routes to the out-of-band `nvcc` lowering. Now stated, with the seam called
  out explicitly.

### Substantive additions

- A listing of the matched TritonGPU IR, so the impossibility result is read off the code:
  the per-lane predicate exists, `tt.reduce` destroys it, `scf.condition` takes one `i1`.
- A **specification** of the proposed primitive, not just its name: lane-wise condition,
  per-lane live-out freezing, the body restriction it forces (no cross-lane ops, which is
  the body-safety condition the verifier already discharges), and the open composition
  question (`tt.dot`).
- **ML-Triton** (arXiv:2503.14985) added to related work. It is the strongest "isn't this
  already solved" risk in the tile-DSL space: it descends Triton's interface to the warp.
  Answered precisely, since a warp is still 32 lanes under one latch. DARM (CGO'22) added
  alongside the linearization work.
- The Warp control (0.9x of hand-CUDA) restored to the method section, where it does its
  real job of separating abstraction *height* from execution *paradigm*.

### A claim that had no CSV, measured and found false (21 Aug 2026)

The sweep of every numeric claim against its CSV turned up one with no CSV at all: *"divergence
adds on top of the floor, taking power-law SpMV from 2.2x at tile width 32 to 5.8x at 256."*
The 5.8x lived only in a docstring comment in `experiments/cure/landmarks/landmark_spmv.py:11`;
the committed CSV has the default `BLOCK=32` only. For a paper whose thesis is that every number
traces to a versioned CSV, that is the worst kind of gap.

Measured it on a Modal A100, oracle-gated in every row
(`paper2/data/landmark/spmv_width_nvidia_a100.csv`). **The claim is false.**

| tile width (num_warps) | uniform, CV=0 | power-law, CV=3.79 | increment |
|---|---|---|---|
| 32 (1) | 1.80x | 3.20x | +1.40 |
| 64 (2) | 5.64x | 3.94x | -1.70 |
| 128 (4) | 5.65x | 4.57x | -1.08 |
| 256 (8) | 5.73x | 5.60x | -0.13 |

By width 256 the matrix with **no** control divergence is the dearer of the two. What grows with
tile width is the lowering baseline, and part of it is the same `num_warps` artifact as the NFA
worklist, since `NUM_WARPS = BLOCK/32`. The paper now reports this as its fifth self-correction.

Trust check on the run: at width 32 it reproduces the committed cross-arch A100 figures
(uniform 1.78, power-law 3.20) to within noise, so the harness is measuring what it did before.

### Still open

- **CLOSED (21 Aug, measured).** The pass showing ~1.0x on real workloads was the largest
  reject risk. It is now a measured, controlled result rather than an argument. Built the
  pinned cure wheel on Modal and ran the lock-step kernel on the SAME power-law row lengths
  the SpMV witness uses, with an accumulate body instead of the DRAM gather (H100, both
  modes oracle-gated, PTX redux.sync 1->0):

  | distribution | body | speedup |
  |---|---|---|
  | power-law rows | DRAM gather (the SpMV witness) | ~1.0x |
  | power-law rows | cheap accumulate | **6.31x** |
  | uniform rows | cheap accumulate | 1.09x (control) |

  Only the denominator changes. `paper2/data/landmark/cure_real_dist_h100.csv`. The H100 is
  also a third architecture for the pass: geometric law 1.47x, against 2.46x on the 4070 and
  1.8x on the A100.

- The older, weaker form of the same point, kept because it is the reason the experiment was
  worth running: the power-law CSR matrix the SpMV witness runs on has a per-warp straggler of 150
  rows against a mean of 16 (D=9.65), a *larger* straggler than five of the six synthetic
  distributions the law was fitted on and a higher divergence ratio than any of them
  (`paper2/data/landmark/straggler_of_real_inputs.csv`, computed from the generator, no GPU).
  So the workload is not short of stragglers; the per-iteration DRAM gather dwarfs the tax.
  The scope condition is therefore "divergent trips **over a cheap body**", not
  "control-bound", and the paper now says so. What would still strengthen it is the pass
  firing on a real workload that meets both halves.
- Attempted the obvious candidate (the NFA worklist through the built pass) and **stopped**:
  the Modal volume is full, 13 Triton trees of 13 GB each, all at the same commit
  `c346e50c7b`, from the upstream PR work. Freeing space means deleting build trees another
  workstream may need, and the paper's own mechanism predicts a null there anyway, since the
  NFA residual is the latency channel and the pass removes the straggler channel. Left to
  the user.
- **CLOSED (21 Aug, measured).** The selector was not wired to the in-IR pass. It is now.
  The two halves had never coexisted in one Triton build because `ThreadRegion.cpp` carries a
  superseded second pass that `registration.patch` does not declare, so the file will not
  compile once the patch is applied. With that pass removed the wheel carries both, and the
  whole decision runs in one process: on an H200 the lock-step worklist is detected and
  routed for **1.90x** (69.8 -> 36.7 us, PTX redux.sync 1 -> 0), while a fixed-trip variant of
  the same body is not detected and keeps the tile path. Asking the detector for its in-IR
  rewrite mode changes neither decision nor timing.
  Data: `paper2/data/landmark/selector_incompiler_h200.csv`. Recipe and the three traps:
  `experiments/cure/triton_thread_region_pass/README.md`.
- `refs.bib` has bibtex warnings for missing page numbers and publishers on several
  entries. Cosmetic for review, worth fixing for camera-ready.

## Bibliography audit (21 Aug 2026) — seven errors, two of them severe

Every non-arXiv entry in `paper2/refs.bib` was checked against Crossref, Wiley, dblp and the
publishers' own pages. The edits landed in commit `34a2f40`, whose message is about the
beyond-64-state result and does not mention them; this is the record.

**Two DOIs resolved to entirely different papers.** Verified independently on Crossref:

| key | DOI in the paper | what it actually resolves to | corrected to |
|---|---|---|---|
| `fung2007` | `10.1109/MICRO.2007.12` | Dubach et al., *Microarchitectural Design Space Exploration* | `10.1109/MICRO.2007.30` |
| `subwarp2022` | `10.1109/HPCA53966.2022.00065` | Kao & Krishna, *MAGMA* | `10.1109/HPCA53966.2022.00090` |

Both are load-bearing citations: Fung is the divergence baseline we distinguish ourselves
from, Subwarp Interleaving is the hardware fix we distinguish ourselves from. A reader
following either DOI would have landed on an unrelated paper. Note that ACM DL serves the
Fung paper at the `.12` URL, so ACM's own record is self-inconsistent; Crossref and dblp
both say `.30`.

**Two author lists were wrong**, the same class as the `hopps2025` error found earlier:

- `mlirregex2025` had "Somaini, Filippo and Carloni, Luca". The authors are **Andrea
  Somaini** and **Filippo Carloni**: the first names were shifted by one and "Luca" was
  invented. Luca Carloni is a different and well-known researcher, so this misattributed
  the work to someone who did not write it.
- `bitgen2025` had "Chu, Hongyu"; the second author is **Xiaowen Chu**.

**One entry conflated two papers.** `instroofline2022` carried the title and authors of the
PMBS@SC'19 paper but the DOI of the CPE'22 journal article, which has a third author
(Awan). Aligned to the DOI.

**One title was truncated.** `hoefler2015` was missing "Twelve Ways to Tell the Masses When
Reporting Performance Results".

Left alone deliberately: `insum2026`'s author order, where ACM's deposited metadata
disagrees with the camera-ready and arXiv, and ours matches the camera-ready; and the
accent in `hopps2025`'s "Sánchez", where Crossref and dblp disagree.

Result: bibtex warnings 48 -> 0, no "et al." in the rendered bibliography, every field
sourced. The remaining `Overfull \vbox (1.888pt)` is on page 13, inside the references,
caused by the bibliography growing; it is sub-millimetre and vertical, and clearing it would
mean removing accurate metadata.

**Lesson, already recorded in `docs/memory/verify-citations-on-the-source.md`:** a DOI that
looks plausible is not a verified DOI. The only check that catches this is resolving it.
