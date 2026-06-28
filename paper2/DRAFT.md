# From Diagnosis to Cure: Decomposing the Tile-SPMD Abstraction Regret on Irregular Automata

Working draft (paper 2). Companion to the HPEC "Two Faces of Abstraction Regret" study (paper 1 =
diagnosis). All numbers trace to `paper2/data/*.csv`, RTX 4070 (sm_89), Triton 3.5.1, torch
2.9.1+cu128, CUDA 13.3/driver 580. Every kernel is validated bit-for-bit against the `reference.py`
NFA oracle / `simulate_dfa` DFA oracle before any throughput is reported. Timings are medians over
warmup+9 reps at GPU-saturating batch.

## Abstract (draft)

Tile-based GPU DSLs (Triton) trail hand-written CUDA by ~10× on irregular finite-automata workloads.
Paper 1 attributed this "abstraction regret" to the execution *paradigm* (tile/SPMD vs thread-SIMT)
rather than abstraction height. This paper turns the diagnosis into an *anatomy and a cure*: we
decompose the regret into three independently measured components and identify, to the instruction
level, the single missing primitive. On a work-efficient NFA worklist the 10× anchor decomposes as
**(A) a launch-configuration artifact (~3.4×, default `num_warps`), (B) a lane-packing-recoverable
component (warp-redundant scalar execution), and (C) an irreducible ~2× residual**. Component C is
*not* instruction count, occupancy, or memory bandwidth: at matched occupancy, matched warp count,
and fewer warp-instructions than CUDA, a per-lane Triton worklist is still 3.3× slower because it is
latency-bound (10% issue activity). The root cause is **intra-warp latency hiding**: a CUDA warp is
32 independent scalar threads that hide each other's memory latency, whereas a Triton warp is 32
lock-step lanes of one instruction stream and cannot. We show this residual is *regime-dependent*:
for the memory-bound DFA it closes (lane-packed Triton matches CUDA, 1.05×, once the table spills to
DRAM) because both paradigms then rely on cross-warp memory-level parallelism. We specify the missing
IR primitive (a `scalar_program` region lowering each lane to an independent instruction stream),
bound what it can buy, and give the thread-model existence proof.

## 1. Contributions

1. A **falsifiable decomposition** of the tile-SPMD automata regret into artifact + recoverable +
   irreducible, each measured and Nsight-attributed (§4).
2. Identification of the irreducible residual as **intra-warp latency hiding** — not redundancy,
   occupancy, instruction count, or bandwidth — via a per-lane Triton worklist that has fewer
   warp-instructions than CUDA yet is 3.3× slower at 10% issue activity (§4.4).
3. A demonstration that the residual is **regime-dependent** (control-flow/latency-bound NFA vs
   memory-bound DFA), unifying both with paper 1's two faces through a single latency mechanism (§5).
4. A **launch-configuration finding** (default `num_warps=4` inflates the worklist regret ~3.4×) that
   re-baselines prior worklist numbers and is a methodological caution for DSL benchmarking (§4.2).
5. The **missing-primitive design** + a bound on its payoff + the thread-model existence proof (§6).

## 2. Background

NFA simulation (work-efficient worklist: iterate the active state set via `ffs` over a bitmask) is
control-flow / latency-bound; DFA simulation (one dependent table gather `trans[cur*256+sym]` per
symbol) is memory-bound — paper 1's "two faces". CUDA/Warp are thread-SIMT (1 thread = 1 string);
Triton is tile/SPMD (a program operates on tiles). The kernels, oracle, and ablations are from the
`gpufsm` framework.

## 3. Method

Correctness gates speed (oracle bit-for-bit on every kernel/config first). Median+CI95, saturating
batch (lane-packing benefit is occupancy-gated, so we sweep batch). Nsight Compute confirms the
mechanism behind every throughput claim, not just the wall-clock. Negative/partial results and
corrected over-claims are reported (the analysis below corrected two of our own intermediate
hypotheses — see §4.3, §4.4).

## 4. The decomposition (NFA worklist)

### 4.1 The anchor
Work-efficient register-resident worklist, ≤64 states, batch 4096, 12 configs (16/32/48/64 states ×
3 seeds): `triton/worklist` 22 Gbps vs `cuda/worklist` 227 Gbps = **10.1× median** (9.4–12.3×).
[`m0_anchor_rtx4070.csv`]

### 4.2 Component A — launch-configuration artifact (~3.4×)
The Triton worklist defaults to `num_warps=4`: four warps per program redundantly process the *same*
one string. Sweeping `num_warps∈{1,2,4,8}` (scalar worklist, oracle-gated): `nw=1` vs `nw=4` =
**2.77×@4096 → 3.44×@16384 → 3.69×@65536** (each doubling ~halves throughput — added warps are pure
redundancy). [`m2f_numwarps_rtx4070.csv`] *This inflates prior worklist regret numbers (incl. paper
1's) and must be disclosed.*

### 4.3 Component B — warp redundancy, recoverable by lane-packing (occupancy-gated)
M1 (Nsight): the scalar Triton program executes its scalar work warp-uniformly — `thread_inst_per_
inst = 32.00` (Triton) vs 30.34 (CUDA), the same number with opposite meaning (CUDA's 32 lanes run 32
*different* strings, Triton's run the *same* one), yielding ~90× more warp-instructions.
[`m1_nsight_rtx4070.csv`] *Lane-packing* (pack 32 strings into the 32 lanes, exploiting the shared
CSR so inner-loop bounds stay scalar) removes this. On the dense scan, pure lane-packing (work held
equal) recovers **3.2×@4096 → 9.8×@16384 → 19.4×@65536** toward the ideal 32×; it is **occupancy-
gated** (small batch starves it). [`m2_batch_scaling_rtx4070.csv`] *Correction:* M1 framed the ~90×
warp-redundancy as the bottleneck; Nsight on the lane-packed kernel shows the redundancy is removed
~26× yet throughput moves only 3.8× — it was largely hidden by occupancy. [`m2_nsight_rtx4070.csv`]

### 4.4 Component C — the irreducible residual = intra-warp latency hiding (~2×)
A per-lane Triton worklist (each lane its own `ffs` + per-lane CSR gather, no cross-lane reduce, no
active-set union — the cleanest pure-Triton "scalar-lane program") beats the union variant 1.27× but
is still **0.51× of CUDA** (≈2× slower). [`m3_lite_rtx4070.csv`] Nsight is decisive: it has **fewer
warp-instructions than CUDA (4.66M vs 5.07M) and the same occupancy (22.5%)**, yet is 3.3× slower
because **issue activity is 10.1%** and warp-latency is 27 cyc/inst — *latency-bound*.
[`m3_lite_nsight_rtx4070.csv`] Raising occupancy does not help: a BLOCK sweep {32,64,128,256} only
*worsens* it (0.49×→0.31×) — bigger lock-step tiles add divergence (the `while max` runs to the
busiest of more lanes). [`m3_lite_b_occupancy_rtx4070.csv`] **Root cause:** a CUDA warp is 32
*independent* threads, so a stalled load on one lane is hidden by 31 others; a Triton warp is 32
lock-step lanes of one instruction stream and can hide latency only *across* warps (occupancy-bound).
The pure-Triton ceiling is ~0.49× CUDA; only per-lane independent instruction streams (the thread
model) close it.

## 5. Regime-dependence (DFA) — the unification

The DFA (memory-bound) decomposes the same way but the *residual closes in the DRAM regime*. Across
table sizes (cache→L2→DRAM), oracle-gated: scalar Triton DFA is **flat ~29 Gbps** for all sizes
(independent confirmation of paper 1's scalar-gather ceiling); lane-packing gives **~12×** over it;
and **PK/CU = 0.55–0.62× in cache but 1.05× at a 16 MB (>L2) table** — lane-packed Triton *matches*
CUDA. [`m4_dfa_rtx4070.csv`] Mechanism (honest: DRAM utilization only 14.6%, so memory-*latency* not
saturated bandwidth): at moderate (cache) latency CUDA's intra-warp hiding wins ~1.7×; at DRAM
latency both paradigms must hide latency via cross-warp memory-level parallelism and **converge**.
[`m4_dfa_nsight_rtx4070.csv`] So the NFA's fundamental ~2× residual and the DFA's closeable gap are
**one mechanism at different latency scales**, mapping onto paper 1's control-flow vs memory faces.

## 6. The cure: the missing primitive

**Design.** A `scalar_program` region (or per-lane `serial_range`) in the tile DSL that lowers the
marked code to a *per-thread independent instruction stream* (thread-SIMT) instead of a lock-step
tile, giving each lane independent control flow and intra-warp latency hiding. The diagnosis names
exactly what it must provide: independent per-lane `ffs`/`while`, per-lane data-dependent loop bounds,
register-resident per-lane bitset, scalar gather.
**Bound.** §4.4/§5 bound the payoff: it closes the latency-bound residual (up to ~2× on the NFA) and
nothing extra in the already-converged DRAM-DFA regime — a regime-dependent, falsifiable prediction.
**Existence proof.** The thread model already realizes it: CUDA achieves 1.0× and, in paper 1, the
high-level *Warp* DSL (thread-SIMT Python) reaches 0.9× — so the primitive is not hypothetical, it is
the paradigm the tile DSL lacks. A full Triton-MLIR implementation is a separate systems effort
(future work; feasibility assessed in `docs/CURE_PROGRESS.md`).

## 7. Threats to validity
- Single GPU (RTX 4070); the mechanism (intra-warp latency hiding, issue activity, the DRAM
  convergence) is architecture-general but absolute factors need an A100/H100 re-run.
- The lane-packed prototypes are single-word (≤64 states). Real ANMLZoo automata (Levenshtein 2787,
  Brill 42661, Fermi 40786) run oracle-valid but sub-Gbps (algorithmic, orthogonal to the regret);
  testing lane-packing on them needs a multi-word kernel (future strengthening).
- `num_warps` is a tuning knob; we disclose the sweep rather than report a single config (the
  "did you tune Triton?" de-risk).

## 8. Related work (to expand)
Paper 1 (diagnosis); Triton/MLIR; NVIDIA Warp (thread-SIMT existence proof); Hexcute/Tawa/Descend
(layout/dataflow decomposition on dense tensors — off the irregular axis); ngAP/HybridSA/BitGen
(CUDA-only automata, orthogonal axis). Gap: no constructive, instruction-level account of *why* a
tile DSL pays on irregular control flow, nor the regime-dependent cure.

## 9. Conclusion
The abstraction regret on irregular automata is not a monolith: it is a launch artifact, a recoverable
redundancy, and an irreducible intra-warp-latency-hiding residual whose size is set by the latency
regime. The cure is a named, bounded, existence-proven IR primitive. Building it in Triton-MLIR is the
landmark future step.
