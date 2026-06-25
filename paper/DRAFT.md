# Abstraction Regret: Memory-Layout and Execution-Paradigm Constraints in GPU DSLs for Irregular Automata Processing

**Status:** working draft (session-2 autonomous). All numbers are from this repo's
versioned data (`paper/data/*.csv`) on an RTX 4070 (sm_89); regenerate figures with
`python paper/figures.py`. Citations are verified in `docs/LITERATURE_REVIEW.md`.

---

## Abstract

GPU domain-specific languages (DSLs) such as OpenAI Triton promise near-CUDA performance
at far lower programming effort, a promise borne out on *regular* tensor algebra. We ask
whether it holds for an *irregular* workload — non-deterministic finite automata (NFA)
processing — and find that it does not, in an instructive way. We introduce **abstraction
regret**: the performance a DSL forecloses because its abstraction cannot express the
memory layout *or the control flow* a workload needs. We make it measurable with a simple
two-parameter cost model and a controlled, constant-algorithm ablation across four memory
axes (byte→bit working set, global→shared CSR, sync→async transfer, single→multi-stream)
and three DSLs (CUDA, Triton, NVIDIA Warp), plus a feasibility study of Triton's Gluon
frontend. Two findings stand out. First, on a faithful full-scan NFA kernel the bottleneck
is *compute*, not memory: staging the transition table into shared memory (zero global
traffic) leaves throughput unchanged, and throughput scales as 1/n² with state count — so
memory-layout techniques cannot help until the algorithm is made work-efficient. We then
build a work-efficient active-set kernel that is 250×–10⁴× faster and reaches 15–132 Gbps.
Second, the Triton↔CUDA gap is a per-DSL constant (≈15.7× on this kernel) that no
traffic/compute term explains, while Warp — an equally high-level *Python* DSL — matches or
beats hand-written CUDA (0.62×). Abstraction regret is therefore set by the execution
*paradigm* (tile/SPMD vs thread-SIMT), not by how high-level the DSL looks: Triton's tile
model strains to express data-dependent automata control flow, Gluon cannot express it at
all, and Warp's thread model expresses it for free.

## 1. Introduction

Irregular workloads — graph traversal, sparse algebra, automata — are dominated by data
layout and data-dependent control flow rather than arithmetic. GPU DSLs raise the
abstraction level by hiding exactly those concerns (thread indexing, memory placement,
control flow), which is why they excel on dense tensor kernels. This paper measures what
that hiding costs on NFA processing, a canonical irregular workload (deep-packet
inspection, regex, bioinformatics).

**Contributions.**
- **(A) Abstraction regret, operationalized.** A named framing plus a predictive
  two-parameter cost model (§3) and a constant-algorithm factorial ablation (§5–6) that
  attribute the DSL gap to *expressible memory layout and control flow*, distinguishing it
  from generic performance-portability efficiency (Pennycook et al.) and from autotuning.
- **(B) A work-efficient portable NFA engine.** An active-set/worklist bit-packed kernel
  (§4) that removes the O(n²) compute wall and reaches 15–132 Gbps, the regime where the
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
  O(active), no O(n²)).
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
are captured in `paper/data/sweep_techniques.csv`.

## 6. Results

**6.1 The faithful kernel is compute-bound; memory layout is irrelevant there.**
`multistream`, `multistream_shared` (modeled CSR traffic = 0), and `multistream_async` tie
to within the bootstrap CI at every size (Fig. `fig_memory_ablation`), and throughput scales
as 1/n² (Fig. `fig_throughput_vs_states`). The cost model fits this to <1% relative error at
large n with a negligible memory term. Conclusion: the memory axes (byte→bit, shared CSR,
async) cannot help until the algorithm is work-efficient.

**6.2 The work-efficient kernel unlocks the regime.** The `worklist` kernel is 250×–10⁴×
faster than full-scan (the speedup grows with n: 1148× at 64 states, 7147× at 500) and
reaches 15–132 Gbps (Fig. `fig_worklist_speedup`), moving the workload toward memory-bound
where the §4 memory techniques become load-bearing (future work confirms with Nsight once
counters are unblocked, `docs/PROFILING.md`).

**6.3 Abstraction regret is the execution paradigm.** On the same full-scan kernel, the
fitted per-DSL compute cost vs CUDA is **Triton 15.7×, CUDA 1.0×, Warp 0.62×**
(Fig. `fig_abstraction_regret`). Two equally high-level Python DSLs sit at opposite extremes:
Triton's tile/SPMD paradigm strains to express per-state data-dependent control flow (it can
only run as one unrolled program), Gluon cannot express it at all, while Warp's thread-SIMT
model expresses it naturally and its codegen beats hand-written CUDA. Regret tracks *what the
model forbids you to express*, not abstraction height.

## 7. Related work

Performance portability (Pennycook et al., PMBS 2016 / FGCS 2019) measures efficiency across
a *hardware set*; abstraction regret measures efficiency across an *abstraction axis at fixed
hardware*, attributed to expressibility. The Halide (PLDI 2013) / TVM (OSDI 2018) lineage
established that abstraction constrains the schedule space; we show the analogous constraint
on *control flow + memory layout* for irregular automata. SpMV format-selection (BestSF, TACO
2018) and Gunrock (PPoPP 2016) establish that layout dominates irregular GPU performance; we
add the DSL-expressibility axis. We must (and do) rebut the autotuning counter-thesis
(arXiv:2505.03780): our gap is layout/control-flow expressibility, not tuning — Triton cannot
express the shared-CSR layout or the work-efficient bit-scan at any tuning. SOTA automata
engines (ngAP, HybridSA, BitGen, AsyncAP, AutomataBLAS) are the baselines our worklist engine
must approach; our contribution is the framing + cost model + the multi-DSL expressibility
study, not the kernels alone.

## 8. Limitations & future work

- Single GPU architecture so far (RTX 4070); validate on a 2nd arch (generality) before
  camera-ready.
- The worklist kernel is one thread/string; a cooperative warp/block-parallel version
  (iNFAnt/ngAP-style) is needed to approach SOTA absolute throughput and to make the memory
  axes bite — this is the path for contribution (B) to land at MICRO/ASPLOS strength.
- Nsight Compute counters are admin-gated on the test host (`docs/PROFILING.md`); the
  compute-bound claim is established by controlled ablation instead, with counters as
  confirmatory follow-up.
- CUDA bit-packed/worklist are capped at ≤512 states (8×64-bit words); Warp at ≤64.

## 9. Conclusion

For irregular automata, the GPU-DSL promise breaks along a measurable axis we call
abstraction regret. It is governed by expressibility — of memory layout and, more sharply
here, of data-dependent control flow — not by abstraction height: a high-level thread-SIMT
DSL (Warp) matches CUDA while a high-level tile DSL (Triton) pays 15.7× and its lower-level
sibling (Gluon) cannot express the kernel at all. The memory-organization thesis holds, but
only once the algorithm is work-efficient enough to be memory-bound — which our worklist
engine achieves.

---

### Artifact

Framework, kernels, oracle, sweep/calibration scripts, versioned CSVs, and figure generator
are in this repository (branch `worklist`/PR #1). `pip install -e ".[dev,triton,warp]"`
(+ `GPUFSM_BUILD_CUDA=ON`); `pytest -m "not gpu"` for CPU, `pytest -m gpu` on a GPU box;
`python scripts/sweep_techniques.py` and `python paper/figures.py` regenerate the data/figures.
