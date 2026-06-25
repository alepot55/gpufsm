# Literature Review: GPU Automata Processing & DSL Abstraction Regret

Generated: 2026-06-25
Review type: **scoping** (exploratory, to set publication direction)
Search window: 2009–2026 (emphasis 2021–2026)
Databases / sources: arXiv, ACM DL, IEEE Xplore, USENIX, dblp, Semantic Scholar/Crossref,
official project docs & GitHub/Zenodo (via web search + fetch). All citations below were
verified for title/venue/year and a DOI/arXiv/URL unless explicitly flagged.

> Method note: produced via 5 parallel scoping searches (SOTA automata, emerging GPU DSLs,
> abstraction-cost/portability, measurement methodology, venues/timeline). This is a scoping
> review for direction-setting, **not** a systematic review — no exhaustive protocol/PRISMA.

---

## Research Question

For irregular finite-automata (NFA/FSM) processing on GPUs, **how much of the performance gap
between a high-level DSL (OpenAI Triton) and low-level CUDA is attributable specifically to the
*memory layout the DSL can express* ("abstraction regret"), rather than to algorithm or
scheduling — and is a quantified, ablation-based answer a publishable contribution given the
2024–2026 state of the art?** Secondary: would adding newer GPU languages turn the
Triton-vs-CUDA study into a stronger multi-DSL analysis?

---

## Inclusion / Exclusion Criteria

- **Include:** GPU automata/NFA/regex engines; GPU DSLs/kernel languages (2023–2026); performance
  portability & abstraction-cost studies; memory-layout-dominance evidence for irregular GPU
  workloads; GPU benchmarking methodology; relevant venues/CFPs.
- **Exclude:** pure FPGA/ASIC automata accelerators except as context; dense-tensor-only DSL work
  except as contrast; non-GPU regex except Hyperscan (CPU baseline).
- Preprints included but **labeled**; primary results separated from reviews/positions.

---

## Evidence Summary (verified anchors)

### A. SOTA GPU automata processing
| Work | Venue / Year | Memory & execution strategy | Key result | Verified ID |
|---|---|---|---|---|
| iNFAnt | SIGCOMM CCR 40(5), 2010 | NFA on GPU; symbol-indexed transition table in global mem; active-state bit-vector | First practical GPU **NFA** engine; the canonical baseline | 10.1145/1880153.1880157 |
| Hyperscan | USENIX NSDI 2019 | CPU SIMD regex; graph decomposition | Standard CPU SOTA baseline (8.7× Snort) | dblp WangHCPLHZ19 |
| ANMLZoo | IISWC 2016 | Benchmark suite (14 automata apps) | De-facto AP benchmark suite | 10.1109/IISWC.2016.7581271 |
| AutomataZoo | IISWC 2018 | Benchmark suite (24, architecture-neutral) | Modern, less-biased suite | hplp/AutomataZoo |
| **AsyncAP** | **POMACS 7(1) Art 27 / SIGMETRICS 2023** | Adds **input-symbol-level** parallelism (async start offsets); worklist | up to 58× avg (parallelism-limited), 2.4× (saturated) | 10.1145/3579453 |
| **ngAP** | **ASPLOS 2024 (Best Paper)** | **Non-blocking** multi-symbol processing + memoization + privatization (locality) | **7.9× avg, up to 901×** over prior GPU SOTA | 10.1145/3617232.3624848; TOCS 10.1145/3748646 |
| **HybridSA** | **OOPSLA 2024** | **Bit-parallel** NFA (shift-and) + heterogeneous CPU+GPU split | Large gains via bit-parallelism + reduced irregular traffic | 10.1145/3689771 |
| **BitGen** ("Interleaved Bitstream Execution…") | **MICRO 2025** | **Parabix bitstream** programs → CUDA; fused per-block loop | **19.5× vs GPU**, 1.7× vs CPU regex (geomean) | 10.1145/3725843.3756052 |
| AutomataBLAS ("Advancing Matrix Operations…") | ACM TACO 2025 | AP-as-**SpMV**, custom CTA-level kernel (memory-efficient) | Competitive matrix-algebra AP engine | 10.1145/3774656 (verify issue) |

### B. Abstraction-cost / performance-portability lineage
| Work | Venue / Year | Relevance | Verified ID |
|---|---|---|---|
| Pennycook et al., *A Metric for Performance Portability* | PMBS@SC 2016 | The canonical P(a,p,H) metric — must differentiate from | arXiv:1611.07409 (preprint); FGCS 2019 10.1016/j.future.2017.08.007 |
| *GPU Performance Portability needs Autotuning* | 2025 | **Counter-thesis**: gap is tuning, not abstraction — must rebut | arXiv:2505.03780 (preprint) |
| Halide | PLDI 2013 | "Abstraction decouples algorithm from schedule" root idea | 10.1145/2491956.2462176 |
| Triton (Tillet et al.) | MAPL 2019 | Triton ≈ 90–105% of CUDA on **regular** tensor ops — contrast | 10.1145/3315508.3329973 |
| SpMV format-selection (BestSF, Auto-SpMV) | TACO 2018; 2023 | Layout/format, not algorithm, dominates irregular GPU perf | 10.1145/3226228; arXiv:2302.05662 |
| Gunrock | PPoPP 2016 | "Data-centric abstraction" for irregular graph workloads | 10.1145/2851141.2851145 |

### C. Measurement methodology
| Source | Venue / Year | What it gives | Verified ID |
|---|---|---|---|
| Roofline | CACM 2009 | Prove memory-bound via arithmetic intensity | 10.1145/1498765.1498785 |
| Hierarchical GPU Roofline (NERSC) | CC:P&E 2020 | L1/L2/HBM roofline to attribute speedup to a memory level | 10.1002/cpe.5547; arXiv:2009.02449 |
| Hoefler & Belli, *Scientific Benchmarking…* | SC 2015 | Rigor: median + CI, not best-of-N/mean±std on non-Gaussian timings | 10.1145/2807591.2807644 |
| Nsight Compute Profiling Guide | NVIDIA (current) | Counter semantics (DRAM bytes, L2 hit, sectors/req, occupancy) | docs.nvidia.com/nsight-compute |
| ACM Artifact Review & Badging v1.1 | ACM | Available/Functional/Reusable/Reproduced badges | reviewers.acm.org |

---

## Thematic Synthesis

### 1. The frontier is *implicitly* memory-centric but no one isolates memory at constant algorithm
The 2024–2026 frontier — **ngAP** (non-blocking + memoization + privatization), **HybridSA**
(bit-parallel + heterogeneous), **BitGen** (bitstream fusion), **AutomataBLAS** (AP-as-SpMV) —
all win by *reducing or reorganizing irregular memory traffic*, yet each frames its contribution
as a new **algorithm/execution model** and bundles memory effects in. **No verified paper
presents a controlled factorial ablation that holds the algorithm fixed and toggles only:
byte→bit state representation, global→shared/L2-resident CSR, sync→async transfer,
single→multi-stream.** That clean ablation is genuinely unclaimed. *(High confidence.)*

### 2. "Abstraction regret" is novel terminology over a non-novel phenomenon
The literal phrase returns **zero** literature hits — a coinage opportunity. But the underlying
idea re-skins (a) Pennycook's portability-efficiency deficit, (b) Halide's "abstraction limits the
schedule space", and (c) decades of P3HPC "portable gets 50–80% of native CUDA". **To be
defensible the term must be operationalized**: a measured cost model decomposing the Triton↔CUDA
gap into memory-layout-attributable components, on an *irregular automata* workload, with the
autotuning counter-thesis (arXiv:2505.03780) explicitly rebutted. A named coinage *plus a
measurement instrument* is publishable; a coinage alone is not. *(High confidence.)*

### 3. The defensible novel core = irregular automata × DSL-constrained memory layout × quantified ablation/cost model
No one occupies this intersection: SpMV-format work has layout-dominance but no DSL-abstraction
axis; portability work has the abstraction axis but on regular dense kernels; automata-GPU work
optimizes layout but never frames it as a DSL-expressibility cost. **Plus: no GPU-automata paper
uses Triton at all** — a Triton automata engine + Triton-vs-CUDA quantification (kernel-time and
transfer-time separated) on the standard suites is new territory. *(High confidence.)*

### 4. Adding newer DSLs can turn a binary into a defensible "abstraction spectrum"
Best framing: a **2-D map — abstraction level (x) vs expressible memory layout / control (y)** —
showing automata throughput tracks the *y*-axis, not *x*. Recommended additions:
- **NVIDIA Warp** (top pick): Python JIT, **thread-based SIMT** → *can* express per-state
  branching, atomics, scatter/gather, bit ops. Same productivity tier as Triton but a different
  execution model → separates "Python abstraction" from "Triton's tile/SPMD model". Trendy/citable;
  no NFA-on-Warp study exists (novelty).
- **Gluon** (cheapest, most on-thesis): Triton's experimental lower-level frontend, same MLIR stack,
  exposes layouts/shared-mem/warp-specialization → measures regret *inside one toolchain* by
  relaxing only the memory-layout constraint. Caveat: still tile/SPMD (relaxes layout, not scalar
  control flow) — that limitation is itself a finding.
- **Mojo** (breadth): cross-vendor (NVIDIA+AMD), full control flow + bit-level memory → external
  validity. Caveat: compiler OSS only ~fall 2026 (reproducibility risk).
- **Traps — discuss, do not benchmark** (tensor/tile-only, can't express data-dependent automata
  without faking dense masked ops): **cuTile/Tile IR, CUTLASS CuTe DSL, ThunderKittens, JAX/Pallas,
  TileLang, Hidet/TVM**. Belong in related work as the 2025–26 "tile convergence" trend; CuTe DSL is
  worth a mention for its first-class layout algebra. *(Medium-high confidence; Warp/Gluon automata
  fit is architectural inference, not yet empirically shown.)*

---

## Gaps and Limitations

- **Open research gap (our opportunity):** constant-algorithm factorial memory ablation + an
  *operational* abstraction-regret cost model on irregular automata across multiple DSLs and ≥2 GPUs.
- **Positioning risk:** without rebutting autotuning (arXiv:2505.03780) and citing Pennycook/Halide/
  SpMV-format/Gunrock, reviewers collapse the work into "performance portability re-discovered".
- **Precision risk (terminology):** HybridSA (bit-parallel NFA) and BitGen (Parabix bitstream) are
  *bit-level regex* engines, **distinct** from our classic *1-bit-per-NFA-state packed bitmap*. Our
  bit-packing lever for general NFA is still largely unclaimed, but the paper must distinguish it
  explicitly or be conflated with existing bit-parallel work.
- **Scope limit:** scoping review, not systematic; a few citations need final bibliographic checks
  (see caveats). FPGA/in-memory AP (Cache Automaton MICRO-50'17, Grapefruit FCCM'20, CAMA) noted as
  context only.

### Corrections to project notes (CLAUDE.md §5) — verified
- **AsyncAP is SIGMETRICS/POMACS 2023, NOT HPCA.** (10.1145/3579453)
- **BitGen (MICRO'25) = "Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs"**,
  a Parabix-bitstream regex engine — not 1-bit NFA state packing.
- **Add HybridSA (OOPSLA'24, 10.1145/3689771)** — bit-parallel GPU NFA, the closest prior art to our
  bit thesis; must cite and differentiate.
- **Add AutomataBLAS (TACO'25, 10.1145/3774656)** — memory-efficient AP-as-SpMV; closest "memory-
  efficient automata on GPU" framing; must differentiate.

---

## Recommended Direction (publication strategy)

**Lead with characterization, treat the engine as conditional** (matches CLAUDE.md §5 A+C primary,
B conditional):
- **Contribution A (safe, novel):** memory-centric factorial ablation + predictive abstraction-regret
  cost model (bytes-moved/symbol → predicted throughput), validated across the ANMLZoo/AutomataZoo
  suite and ≥2 GPUs, with Nsight counters causally attributing each rung.
- **Contribution B (conditional, high-upside):** portable Triton (+ Warp/Gluon) automata engine that
  recovers most of the gap; strong *only* if it beats the trivial multi-stream baseline and lands
  within ~2–3× of ngAP/CUDA.
- **Contribution C:** reproducible artifact (Zenodo DOI, ACM AE badges), figures regenerated from
  versioned CSVs.
- **Multi-DSL extension:** add **Warp + Gluon** as the two probes that empirically separate
  abstraction level from expressible memory layout; **Mojo** for cross-vendor breadth.

**Venue & timeline (today 2026-06-25; IISWC/PACT/MICRO/ASPLOS-Spring deadlines have PASSED):**
1. **arXiv preprint now** — establish priority (cs.DC / cs.PF).
2. **PMBS @ SC26 — full paper Aug 5, 2026** — primary realistic target (peer-reviewed, IEEE Xplore,
   scope = perf characterization). The arXiv draft is ~80% of this.
3. **ASPLOS 2027 Fall — Sep 9, 2026** — the real conference anchor; the ~2-month runway matures B +
   full ablation + multi-GPU + profiling.
4. *Stretch, only if B is real by mid-July:* **HPCA 2027 (paper Jul 31)** or **PPoPP 2027 (Aug 3)** —
   too tight for one week; likely desk-reject if half-validated.
5. **IISWC 2027 (~May 2027)** — most natural topical home but ~11 months out; good fallback.

**One-week reality check:** a *polished arXiv preprint of A+C* is achievable and high-value; a
*workshop paper* in ~1.5–2 weeks (PMBS); a *full conference submission* in one week is **not**
realistic. NB: GPU backends are already validated and `bitpacked`/`multistream` already implemented
(session 2), so the ablation harness is partly built — the remaining gating work is the
**shared-CSR** and **sync→async** axes, **Nsight profiling**, **roofline**, **cost model**, and
**multi-GPU** runs.

**What makes it HIGH contribution vs marginal:** the cost model must be **predictive, not
descriptive** (forecast throughput, validate across automata + ≥2 GPUs); "abstraction regret" must
be a **first-class measured metric**; and B must reach ~2–3× of ngAP/CUDA. Marginal if it's just
"CUDA beats naive Triton 10–30×", or re-derives BitGen's packing / ngAP's async, or sells
multi-stream as a contribution.

---

## References (verified; preprints labeled)

1. Cascarano et al. *iNFAnt*. SIGCOMM CCR 40(5), 2010. 10.1145/1880153.1880157
2. Wang et al. *Hyperscan*. USENIX NSDI 2019.
3. Wadden et al. *ANMLZoo*. IISWC 2016. 10.1109/IISWC.2016.7581271
4. Wadden et al. *AutomataZoo*. IISWC 2018.
5. Liu, Pai, Jog. *AsyncAP*. POMACS 7(1) Art 27 / SIGMETRICS 2023. 10.1145/3579453
6. Ge, Zhang, Liu. *ngAP*. ASPLOS 2024 (Best Paper). 10.1145/3617232.3624848; TOCS 10.1145/3748646
7. Le Glaunec, Kong, Mamouras. *HybridSA*. OOPSLA 2024. 10.1145/3689771
8. Ge, Chu, Liu. *Interleaved Bitstream Execution (BitGen)*. MICRO 2025. 10.1145/3725843.3756052
9. *AutomataBLAS / Advancing Matrix Operations…*. ACM TACO 2025. 10.1145/3774656 (verify issue)
10. Pennycook, Sewall, Lee. *A Metric for Performance Portability*. PMBS@SC 2016. arXiv:1611.07409 (preprint); FGCS 2019 10.1016/j.future.2017.08.007
11. *GPU Performance Portability needs Autotuning*. 2025. arXiv:2505.03780 (preprint)
12. Ragan-Kelley et al. *Halide*. PLDI 2013. 10.1145/2491956.2462176
13. Tillet et al. *Triton*. MAPL 2019. 10.1145/3315508.3329973
14. *BestSF* (SpMV format selection). ACM TACO 2018. 10.1145/3226228
15. Wang et al. *Gunrock*. PPoPP 2016. 10.1145/2851141.2851145
16. Williams, Waterman, Patterson. *Roofline*. CACM 2009. 10.1145/1498765.1498785
17. Yang et al. *Hierarchical Roofline for GPUs*. CC:P&E 2020. 10.1002/cpe.5547; arXiv:2009.02449 (preprint)
18. Hoefler, Belli. *Scientific Benchmarking of Parallel Computing Systems*. SC 2015. 10.1145/2807591.2807644
19. NVIDIA. *Nsight Compute Kernel Profiling Guide* (current).
20. ACM. *Artifact Review and Badging v1.1* (current).

DSL/language sources (official, 2025–2026): Triton releases & Gluon docs (triton-lang.org);
NVIDIA Warp (github.com/NVIDIA/warp); Mojo/Modular (github.com/modular/modular; arXiv:2509.21039);
CUTLASS CuTe DSL & cuTile/Tile IR (developer.nvidia.com); ThunderKittens (HazyResearch, ICLR 2025);
JAX/Pallas (docs.jax.dev); TileLang (arXiv:2504.17577, preprint).

---

## Search Log

| Facet | Sources searched | Date | Notes |
|---|---|---|---|
| SOTA GPU automata | arXiv, ACM DL, dblp, USENIX, GitHub/Zenodo | 2026-06-25 | ngAP/AsyncAP/HybridSA/BitGen/AutomataBLAS verified; DFAGE = code baseline only; AsyncAP venue corrected |
| Emerging GPU DSLs | official docs, GitHub releases, arXiv | 2026-06-25 | Triton 3.5.1, Gluon, Warp, Mojo, CuTe DSL, cuTile, ThunderKittens, Pallas, TileLang |
| Abstraction-cost / portability | arXiv, ACM DL, P3HPC, Springer | 2026-06-25 | "abstraction regret" = 0 hits (novel term); Pennycook/Halide/SpMV/Gunrock anchors |
| Measurement methodology | CACM, Wiley, NVIDIA docs, ACM | 2026-06-25 | Roofline + hierarchical roofline + Hoefler&Belli + Nsight counters + ACM AE |
| Venues / timeline | official CFP pages, researchr, wikicfp | 2026-06-25 | IISWC/PACT/MICRO/ASPLOS-Spring PASSED; PMBS Aug 5, ASPLOS-Fall Sep 9, HPCA Jul 31, PPoPP Aug 3 |

### Confidence ledger
- **High:** memory-ablation gap is real & unclaimed; "abstraction regret" is novel terminology;
  ngAP/HybridSA/BitGen existence & relevance; SpMV layout-dominance; venue deadlines (PMBS, ASPLOS-Fall,
  HPCA); AsyncAP venue correction.
- **Medium:** Warp/Gluon/Mojo *automata* fit (architectural inference, no published NFA benchmark yet);
  AutomataBLAS exact issue/pages; PPoPP 2027 abstract round; Gluon's scalar-control-flow expressiveness.
- **Low / verify before camera-ready:** DFAGE/Cache Automaton/CAMA exact citations; P3HPC 2026 & GPGPU
  2027 & IISWC 2027 dates (CFPs not yet posted); HPCA 2027 conference-date discrepancy.
