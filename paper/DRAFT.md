# The Two Faces of Abstraction Regret: Control-Flow and Memory-Layout Limits of GPU DSLs on Irregular Automata

**Status:** working draft. **Canonical submission artifact = `paper/gpufsm.tex`** (IEEEtran,
builds to PDF); this file is the prose companion, kept in sync on framing. All numbers trace
to `paper/data/*.csv` (RTX 4070, sm_89); positioning/novelty in `docs/NOVELTY_POSITIONING.md`,
verified citations in `docs/LITERATURE_REVIEW.md`.

---

## Abstract

GPU domain-specific languages (DSLs) such as OpenAI Triton deliver near-CUDA performance at
far lower effort on *regular* tensor algebra. We ask what that abstraction costs on
*irregular* workloads and answer with a metric, **abstraction regret**: the performance a DSL
forecloses — algorithm held fixed — because it cannot express the memory layout or control
flow a workload needs. We decompose regret along these two capability axes and instantiate it
on finite automata across the paradigm axis CUDA and NVIDIA Warp (thread-SIMT) vs Triton and
its low-level Gluon frontend (tile-SPMD). Automata expose **two complementary faces**: an NFA
active-set traversal that is *control-flow bound*, and a DFA dense-table walk that is *memory
bound* (throughput halves as the table crosses L2). On both faces the regret is large for the
tile-SPMD DSLs and small for the thread-SIMT ones — Triton pays 5–12× vs CUDA across the two
faces while Warp, an equally high-level *Python* DSL, matches or beats hand CUDA on the NFA
(0.6–0.9×) and stays within ~2× on the DFA — so regret is set by the
execution **paradigm**, not by how high-level the DSL looks. We make the attribution
**falsifiable** with the Triton↔Gluon controlled pair (identical MLIR compiler stack; Gluon
only adds explicit layout/shared-memory control): Gluon *still* cannot express the kernel, so
the binding constraint is the paradigm, not tuning or layout. A two-parameter cost model
predicts the regret and names the missing IR primitives (scalar gather in a tile,
register-resident bitset, data-dependent loop). Along the way we build a portable
work-efficient automata engine (≈330×–10⁴× over a faithful full scan, 15–170 Gbps, validated
bit-for-bit against a CPU oracle on six real ANMLZoo automata up to 48k states).

## 1. Introduction

Irregular workloads — graph traversal, sparse algebra, automata — are dominated by data
layout and data-dependent control flow rather than arithmetic. GPU DSLs raise the
abstraction level by hiding exactly those concerns (thread indexing, memory placement,
control flow), which is why they excel on dense tensor kernels. This paper measures what
that hiding costs on NFA processing, a canonical irregular workload (deep-packet
inspection, regex, bioinformatics).

**Contributions.**
- **(A) Abstraction regret, operationalized.** A named framing decomposed along two
  capability axes (control-flow vs memory-layout) and instantiated on the *two faces* of
  automata (control-flow-bound NFA, memory-bound DFA); a predictive two-parameter cost model
  (§3) and a constant-algorithm factorial ablation (§5–6); and — the move that makes the
  attribution falsifiable — a controlled Triton↔Gluon pair (same MLIR stack, Gluon only *adds*
  layout control; runnable probe `scripts/gluon_probe.py`) plus a capability→cost table naming
  the missing IR primitive, attributing the gap to the *execution paradigm*, distinct from
  generic performance-portability efficiency (Pennycook et al.) and from autotuning.
- **(B) A work-efficient portable NFA engine.** An active-set/worklist bit-packed kernel
  (§4) that removes the O(n²) compute wall and reaches 15–170 Gbps, the regime where the
  memory axes become load-bearing.
- **(C) A reproducible artifact.** One registry-based framework, a CPU reference oracle,
  median+CI95 sweeps, and figures regenerated only from versioned CSVs.

## 2. Background

NFA simulation maintains an active-state set and, per input symbol, applies the transition
relation and an epsilon-closure. GPU engines since **iNFAnt** (Cascarano et al., SIGCOMM
CCR 2010) store transitions in symbol-indexed / CSR form and represent the active set as a
bit-vector. The modern frontier — **AsyncAP** (SIGMETRICS 2023), **ngAP** (ASPLOS 2024,
non-blocking + memoization + privatization), **HybridSA** (OOPSLA 2024, bit-parallel),
**BitGen** (MICRO 2025, Parabix bitstreams), **AutomataBLAS** (TACO 2025, AP-as-SpMV) — all
win by reorganizing or reducing irregular memory traffic, yet each frames its contribution
as a new algorithm and bundles the memory effects in. **Hyperscan** (NSDI 2019) is the CPU
baseline; **ANMLZoo** (IISWC 2016) and **AutomataZoo** (IISWC 2018) are the standard suites.
No prior GPU-automata work uses Triton, isolates the memory axes at constant algorithm, or
frames the DSL gap as an expressibility cost.

## 3. Abstraction regret and the cost model

We model per-symbol time as a roofline-style sum (`gpufsm.costmodel`):

```
time_per_symbol = traffic_bytes_per_symbol / eff_bandwidth      (memory)
                + num_states**2 * compute_s_per_state2           (compute)
```

The compute term is *quadratic* because the faithful kernel performs an O(n) transition
scan and an O(n²) epsilon-closure (n convergence passes × n states) per symbol. The two
constants are fitted per backend from measured throughput (`scripts/calibrate_costmodel.py`).
The ratio of the fitted `compute_s_per_state2` between two DSLs, at constant algorithm, *is*
the abstraction regret.

## 4. Implementation

A single registry (`@register(Backend, technique)`) maps a backend/technique to an executor.
The NFA is stored once in CSR (symbol + epsilon) and consumed identically by every backend,
so comparisons are apples-to-apples. Correctness is gated against a CPU reference oracle
(`reference.py`, latch-first-match) and its bit-packed executable spec (`bitmap.py`).

- **CUDA** (`backends/cuda/nfa_kernel.cu`): `dense` (int8/state), `bitpacked`
  (register-resident 64-bit words, templated on word count), `multistream` (one
  thread/string), `multistream_shared` (CSR cooperatively staged into shared memory — a
  layout Triton cannot express), `multistream_async` (pinned + streamed H2D/kernel/D2H
  overlap), and **`worklist`** (active-bit iteration via `__ffsll` + frontier eps-closure;
  O(active), no O(n²)). A `worklist_global` variant keeps the working set in global memory
  (dynamic word count), removing the 512-state register cap so the engine scales to
  ANMLZoo-sized automata (thousands of states); register residency costs it ~4–5× vs the
  capped register kernel — itself a memory-layout data point. A **`worklist_warp`** variant is
  *block-parallel*: one warp per string, the 32 lanes partition the state-words and scatter
  transitions via `atomicOr`, spreading one string's loads across the warp — **3–9× faster
  than the single-thread global kernel on real ANMLZoo automata at a GPU-saturating batch**
  (~12× on dense synthetic; the speedup is batch- and active-set-density-dependent, far larger
  at small batch where the single-thread kernel cannot fill the GPU;
  `paper/data/worklist_warp{,_batch}_rtx4070.csv`). A **`worklist_shared`** variant
  stages the working set in dynamic *shared* memory (≤1536 states) — the working-set-layout
  ablation of the work-efficient kernel; it only ties `worklist_warp` (0.99–1.10×,
  `paper/data/worklist_shared_rtx4070.csv`).
- **Triton**: `dense`, `bitpacked`, `multistream`. The tile/SPMD model forbids `return`
  inside loops (forcing a done-latch rewrite) and truncates integer literals to 32 bits
  (bit masks must be int64 scalars); it cannot place the CSR in shared memory.
- **Warp**: `multistream` (thread-SIMT Python; active set in a register uint64, ≤64 states).
- **Gluon** (Triton experimental low-level frontend): attempted; `gl.load` always returns a
  layout-typed tensor (no scalar load), so the data-dependent CSR inner loop is
  inexpressible (`docs/DSL_EXPRESSIVENESS.md`).

## 5. Methodology

Throughput is measured on a fixed batch (2048×256 B) as total input bits / per-run batch
kernel time, reported as **median + percentile-bootstrap 95% CI** over 9 runs (GPU timings
are non-Gaussian; Hoefler & Belli, SC15), with kernel and transfer time separated. Every
configuration is verified bit-identical to the oracle on the example suite and on randomized
fuzz/stress NFAs (up to 500 states). Data and environment (GPU, CUDA, Triton, Warp versions)
are captured in `paper/data/sweep_techniques.csv`. We additionally validate on **six real
ANMLZoo automata spanning six families** — Levenshtein (2787 states), Hamming (11349, 2.1M
transitions), Brill (42661, 4.4M), Fermi (40786, particle-physics regexes), RandomForest
(33223, 6.27M transitions — an ML decision-forest, our densest), and CoreRings (48005,
synthetic ring — our largest state count) — loaded from pinned public ANML via
`gpufsm.io.anml` with the `all-input`/`start-of-data` start semantics handled correctly. On
all six, the scalable `worklist_global` kernel matches the reference oracle bit-for-bit on
every input, confirming correctness on production-scale automata (up to 48k states, 6.3M
transitions), not only synthetic NFAs.

## 6. Results

**6.1 The faithful kernel is compute-bound; memory layout is irrelevant there.**
`multistream`, `multistream_shared` (modeled CSR traffic = 0), and `multistream_async` tie
to within the bootstrap CI at every size (Fig. `fig_memory_ablation`), and throughput scales
as 1/n² (Fig. `fig_throughput_vs_states`). The cost model fits this to <1% relative error at
the largest size (n=256: CUDA 0.3%, Triton 0.6%) with a negligible memory term; the larger
residual at small n is fixed launch overhead the pure-n² model omits. Nsight Compute counters confirm it at the hardware
level (`paper/data/nsight_rtx4070.csv`): the full-scan kernel sustains ~19% of peak SM
throughput but only **0.01%** of peak DRAM throughput — compute exceeds memory by ~3 orders
of magnitude — and `multistream_shared` shows identical SM%, DRAM% and occupancy, raising
only the L2 hit rate (79→93%) without moving runtime. Conclusion: the memory axes (byte→bit,
shared CSR, async) cannot help until the algorithm is work-efficient.

**6.2 The work-efficient kernel unlocks the regime.** The `worklist` kernel is ≈330×–10⁴×
faster than full-scan (the speedup grows with n: 332× at 32 states, ≈10⁴× at 500) and
reaches 15–170 Gbps (Fig. `fig_worklist_speedup`), moving the workload toward memory-bound
where the §4 memory techniques become load-bearing (future work confirms with Nsight once
counters are unblocked, `docs/PROFILING.md`).

**6.3 Abstraction regret is the execution paradigm.** On the same full-scan kernel, the
directly-measured throughput ratio vs CUDA is **Triton 6–8×, Warp 0.9×** (Warp is *faster*;
Fig. `fig_abstraction_regret`); the two-parameter cost-model fit corroborates this, isolating
the per-DSL compute constant at **Triton 10.1×, CUDA 1.0×, Warp 0.63×**. Two equally
high-level Python DSLs sit at opposite extremes:
Triton's tile/SPMD paradigm strains to express per-state data-dependent control flow (it can
only run as one unrolled program), Gluon cannot express it at all, while Warp's thread-SIMT
model expresses it naturally and its codegen beats hand-written CUDA. Regret tracks *what the
model forbids you to express*, not abstraction height.

**6.4 Expressibility ≠ efficiency: the regret persists on the work-efficient kernel.** A
sharper test: can Triton express the *work-efficient* worklist (the kernel that matters), and
at what cost? Unlike Gluon (no scalar load), Triton *can* — `libdevice.ffs` plus a
data-dependent `while` loop iterate the active bits — and it is validated against the oracle.
But it still pays **≈6.5× vs CUDA** on that kernel (CUDA worklist 164 Gbps vs Triton worklist
25 Gbps at ≤64 states) — essentially the same regret as the 6–8× on the full scan. So even when Triton expresses the right
algorithm, its tile/SPMD model imposes a large constant penalty on scalar, data-dependent
automata work — expressibility does not buy efficiency.

**6.5 The second face: DFA is memory-bound, and the regret persists.** The DFA dense-table
walk is the memory-bound dual of the NFA. A fine table-size sweep (Fig. `fig_dfa_memory_bound`,
`paper/data/dfa_regret_rtx4070.csv`) makes the signature explicit: CUDA throughput rises to a
peak *exactly* at the L2 capacity (345 Gbps at the 6 MB table) and then falls **2.4×** to a
DRAM-bound plateau (~150–175 Gbps) once the table far exceeds L2. Warp tracks the same shape at
about half (160→97 Gbps). **Triton, by contrast, is flat at 29–32 Gbps across the entire
1–100 MB range** — it never enters the memory-bound regime because its tile/SPMD codegen
bottlenecks the scalar gather first (DFA regret 5–12×, largest where CUDA peaks at L2). So on
*both* faces — control-flow-bound (NFA) and memory-bound (DFA) — the regret tracks the execution
**paradigm**, not the workload's bottleneck: it is an intrinsic property of the DSL, not of
where the kernel happens to be limited.

**6.6 Capability → cost.** The table below maps the capabilities each kernel needs to whether
a DSL expresses them and the resulting regret (✓ expressible, ◐ only as a strained single
program, ✗ inexpressible; regret = throughput vs CUDA on each face). The thread-SIMT DSLs
(CUDA, Warp) express scalar load, data-dependent loops, register-resident bitsets and explicit
shared-memory layout; the tile-SPMD DSLs cannot express a scalar element load at all (Gluon)
or only via a strained single program (Triton) — exactly what the regret measures.

| Capability (paradigm)        | CUDA (thread) | Warp (thread) | Triton (tile) | Gluon (tile) |
| ---------------------------- | :-----------: | :-----------: | :-----------: | :----------: |
| Scalar element load          | ✓ | ✓ | ◐ | ✗ |
| Data-dependent loop          | ✓ | ✓ | ◐ | ◐ |
| Register-resident bitset     | ✓ | ✓ | ✗ | ✗ |
| Explicit shared-mem layout   | ✓ | ✗ | ✗ | ✓ |
| **NFA regret (control-flow)** | 1× | 0.6–0.9× | 6–10× | n/a (✗) |
| **DFA regret (memory)**       | 1× | 1.5–2.2× | 5–12× | — |

## 7. Related work

**Naming.** We deliberately invert *"abstraction **without** regret"* (LMS, Rompf & Odersky):
staging removes call/dispatch overhead, but cannot manufacture a layout or control-flow
pattern the surface abstraction forbids — which is precisely the residual we measure.

**Closest quantitative prior — Hexcute** ablates the Triton↔CUDA gap into layout-synthesis vs
dataflow, but as a *compiler* on *dense tensor* kernels; we contribute a *metric* decomposed
by *capability* (control-flow vs memory), on *irregular* automata, with an expressibility (not
autotuning) framing and a falsifiable Triton↔Gluon control. **Tawa** (warp specialization) and
**Descend** argue *qualitatively* that a DSL's execution model forecloses patterns; we
quantify it cross-DSL.

**Performance portability** (Pennycook et al., PMBS 2016 / FGCS 2019) decomposes efficiency
across a *hardware set*; abstraction regret decomposes across the *expressible-capability*
axis at fixed hardware. The Halide (PLDI 2013) / TVM (OSDI 2018) lineage established that
abstraction constrains the schedule space; SpMV format-selection (BestSF, TACO 2018) and
Gunrock (PPoPP 2016) that layout dominates irregular GPU performance — we add the
DSL-expressibility axis. We rebut the autotuning counter-thesis (arXiv:2505.03780): our gap is
expressibility, not tuning — Gluon, the explicit-control sibling on the same compiler stack,
still cannot express the kernel at any tuning. SOTA automata engines (ngAP, HybridSA, BitGen,
AsyncAP, AutomataBLAS) are CUDA-only baselines our worklist engine must approach; our
contribution is the metric + cost model + the first multi-DSL expressibility study of an
irregular workload.

**Positioning vs SOTA (not a benchmark).** Each engine below reports a speedup over *its own*
baseline/hardware — not comparable in absolute Gbps — and each is a new *algorithm* on CUDA;
our axis is orthogonal (algorithm fixed, measuring DSL expressibility). Matching ngAP-class
absolute throughput is explicit future work.

| System (venue) | Mechanism | Reported (own basis) |
| --- | --- | --- |
| iNFAnt (CCR'10) | symbol-indexed CSR, bit-vector | first GPU NFA |
| AsyncAP (SIGMETRICS'23) | input-symbol async parallelism | 2.4–58× |
| ngAP (ASPLOS'24) | non-blocking + memoization | 7.9× avg |
| HybridSA (OOPSLA'24) | bit-parallel + CPU/GPU split | bit-parallel |
| BitGen (MICRO'25) | Parabix bitstream fusion | 19.5× vs GPU |
| **This work** | **DSL-regret metric, 4 DSLs** | **orthogonal** |

## 7b. Threats to validity

- **Construct:** most throughput points use random NFAs; mitigated by validating on three
  real ANMLZoo automata spanning six families (2.8k–48k states, up to 6.3M transitions) —
  GPU == reference bit-for-bit.
- **Internal:** every backend/technique is gated against the CPU oracle (latch-first-match)
  on examples + fuzz/stress; timings use median + bootstrap CI, warmup, kernel/transfer split.
- **Implementation-effort asymmetry** (key DSL-comparison risk): the Triton 6–15× gap might
  reflect a weaker Triton kernel, not an inherent limit. Against this: (i) kernels are
  structurally mirrored from one CSR spec; (ii) **Warp**, an equally high-level Python DSL
  written with comparable effort, matches/beats hand CUDA — so "high-level ⇒ slow" is not the
  cause; the tile/SPMD model is.
- **External:** single GPU (RTX 4070); cost-model constants are per-backend fits that may
  differ across architectures → cross-architecture generality is future work.

## 8. Limitations & future work

- **Generality across architectures.** One GPU so far (RTX 4070, sm_89, 6 MB L2). The
  *qualitative* claims should be architecture-independent (the capability table is a property
  of the DSL compilers, not the silicon; the Triton↔Gluon control holds at compile time), but
  two quantities are L2-/SM-count-dependent and are the planned camera-ready run on a second
  arch (e.g. A100/H100, ≥40 MB L2): (i) the DFA memory-bound knee, which should shift to a
  *larger* table size on a bigger L2 — a clean falsifiable prediction of the memory-bound
  reading; and (ii) the absolute regret factors (cost-model constants are fits that may
  rescale). The whole sweep regenerates from one command, so it is a re-run, not a
  re-implementation.
- The single-thread worklist under-utilizes the GPU on large automata; the **`worklist_warp`**
  block-parallel kernel (one warp/string, 32 lanes partitioning the state-words) addresses this
  — at a GPU-saturating batch (4096 strings): **3–9× on real ANMLZoo automata**, ~12× on dense
  synthetic; the speedup is batch- and density-dependent (up to ~180× at small batch where the
  single-thread kernel can't fill the GPU). The warp kernel reaches its throughput plateau at a
  far smaller batch, so it's the right choice for few-stream / low-latency use. We further
  tested shared-memory working-set privatization (**`worklist_shared`**): it only ties
  `worklist_warp` (0.99–1.10×) — once work-efficient, the working-set *layout* is no longer the
  bottleneck (mirroring the compute-bound `multistream_shared` result). So the remaining gap to
  SOTA absolute throughput (ngAP-class) is **algorithmic** (memoization / non-blocking
  multi-symbol), not memory residency — the path for (B) to land at MICRO/ASPLOS strength.
- Nsight Compute counters are admin-gated on the test host (`docs/PROFILING.md`); the
  compute-bound claim is established by controlled ablation instead, with counters as
  confirmatory follow-up.
- CUDA bit-packed/worklist are capped at ≤512 states (8×64-bit words); Warp at ≤64.

## 9. Conclusion

For irregular automata, the GPU-DSL promise breaks along a measurable axis we call
abstraction regret. It is governed by expressibility — of memory layout and, more sharply
here, of data-dependent control flow — not by abstraction height: a high-level thread-SIMT
DSL (Warp) matches CUDA while a high-level tile DSL (Triton) pays 6–15× and its lower-level
sibling (Gluon) cannot express the kernel at all. The memory-organization thesis holds, but
only once the algorithm is work-efficient enough to be memory-bound — which our worklist
engine achieves.

---

### Artifact

Framework, kernels, oracle, sweep/calibration scripts, versioned CSVs, and figure generator
are in this repository (branch `worklist`/PR #1). `pip install -e ".[dev,triton,warp]"`
(+ `GPUFSM_BUILD_CUDA=ON`); `pytest -m "not gpu"` for CPU, `pytest -m gpu` on a GPU box;
`python scripts/sweep_techniques.py` and `python paper/figures.py` regenerate the data/figures.
