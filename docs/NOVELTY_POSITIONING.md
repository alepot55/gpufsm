# Novelty & Positioning (verified 2026-06-26) — for the highest-tier submission

Goal: the strongest, most defensible contribution for a top venue, grounded in complete SOTA.
This is the anti-desk-reject map: what we claim, what we must distinguish, the clean gap.
Preprints labeled; verify each ID before camera-ready.

## The defensible novelty claim

> We introduce **abstraction regret**: a metric for the performance a GPU DSL forecloses
> because its abstraction cannot express the **memory layout** or **control flow** a workload
> needs — measured with the *algorithm held fixed*, and decomposed along those two capability
> axes. On irregular finite automata (NFA, control-flow-bound; DFA, memory-bound) across four
> systems on the paradigm axis **CUDA / Warp (thread-SIMT) vs Triton / Gluon (tile-SPMD)**, we
> show regret is governed by the DSL's **execution paradigm, not its abstraction height**, and
> we make the attribution **falsifiable** via the Triton↔Gluon controlled pair (same MLIR
> compiler stack; only the expressiveness lever changes).

Claim novelty ONLY for: (i) capability-axis decomposition (control flow vs layout); (ii) the
irregular-automata regime (NFA+DFA = two faces); (iii) the execution-paradigm-not-height
finding; (iv) the algorithm-fixed regret metric; (v) the Triton↔Gluon falsifiable control.
Do NOT claim "DSLs cost performance" or "Triton<CUDA on irregular code" — both known.

## Must-distinguish prior work (closest first)

| Work | Venue/Year | Relation | How we differ |
|---|---|---|---|
| Rompf & Odersky, "Abstraction without Regret" (LMS) | CACM 2012 | **Name inversion, opposite thesis** | They: staging removes call/dispatch overhead → no penalty. We: an *irreducible* regret bounded by what the surface abstraction can express (a layout/control-flow staging cannot manufacture). Cite explicitly. |
| **Hexcute** (Zhang et al.) | arXiv:2504.16214, 2025 (**preprint**) | **Closest quantitative**: ablates the gap into layout-synthesis vs dataflow/pipelining | They: a layout-synthesis *compiler* on *dense tensor* kernels (GEMM/Attn/MoE). We: a *metric* by *capability axis*, on *irregular automata*, with an expressibility (not autotuning) framing. Distinguish early and explicitly. |
| **Tawa** (Chen et al., incl. Grover) | CGO 2026 (arXiv:2510.14719) | Qualitative: tile/SIMT DSLs "preclude explicit warp roles" | They: qualitative, one pattern (warp specialization), tensor-core attention. We: quantify it, cross-DSL, irregular. |
| **Descend** (Köpcke et al.) | PLDI 2024 | Documents what GPU DSLs can't express (irregular/indirection) | Language-design framing, no regret metric. We quantify. |
| Pennycook, Sewall, Lee — perf-portability metric 𝓟 | PMBS@SC 2016; FGCS 2019 | The reference portability metric | Theirs: efficiency across a *hardware set* (architectural vs application). Ours: across the *expressible-capability* axis at fixed hardware. |
| ngAP (ASPLOS'24, Best Paper), AsyncAP (SIGMETRICS'23), BitGen (MICRO'25), HybridSA (OOPSLA'24), iNFAnt/DFAGE | 2010–2025 | Automata-on-GPU SOTA — our **workload & baselines**, all CUDA | None makes a DSL-expressibility/abstraction-regret argument; we are the first DSL-spectrum study of this workload. |
| KernelBench, TritonBench, ParEval | 2023–2025 | DSL/kernel benchmark suites | KernelBench/TritonBench = **dense only**; ParEval has irregular categories but **no Triton/Warp/Gluon** (CUDA/HIP/Kokkos, LLM-codegen). |

## The clean, claimable gap (verified by absence)

**No prior work benchmarks modern GPU kernel DSLs (Triton/Gluon vs CUDA vs Warp) head-to-head
on irregular finite-automata workloads.** Dense-tensor DSL benchmarks dominate; the one
irregular suite (ParEval) excludes these DSLs. We found no public NFA/DFA/regex implementation
on Triton or Warp. (State as "to our knowledge"; it is absence-of-evidence, not proof.)

## The top-venue move: causal + falsifiable (de-risk "you just didn't tune Triton")

The standard death of a characterization paper is "this is engineering, your Triton kernel was
under-tuned." We neutralize it structurally:
- **Triton↔Gluon controlled pair**: identical MLIR compiler stack; Gluon only *adds* explicit
  control of layout/shared-mem/warp-specialization. If the regret were tuning/layout, Gluon
  closes it. Our probe shows Gluon **cannot even express** the work-efficient automata kernel
  (no scalar load) → the binding constraint is the **tile/SPMD execution paradigm**, not tuning.
  This is a *falsifiable* attribution, not a vibe.
- **Capability → cost table + named missing primitive**: tie each regret number to a concrete,
  named missing capability (scalar-gather-in-tile, register-resident bitset, data-dependent
  loop bound, explicit shared-mem layout). This converts diagnosis into causal mechanism.

## Scope decision (autonomous, for impact × feasibility on 1 GPU, solo)

- **KEEP** automata as the deep anchor: NFA (control-flow face) + DFA (memory face) = "two faces",
  4-DSL paradigm axis, the memory ablation, the cost model, real ANMLZoo validation, the
  optimized worklist (170 Gbps) and worklist_global (42k states).
- **DO NOT** expand to a full BFS/SpMV taxonomy (research flags it as a feasibility trap for a
  solo/1-GPU team and dilutes the deep automata result).
- **Venue:** IISWC / PACT are the honest primary targets (reward characterization+metric, accept
  1–2 GPU studies); reach for CGO/ASPLOS/PLDI *iff* the falsifiable causal story (Triton↔Gluon +
  missing-primitive) lands convincingly. arXiv preprint first for priority.
- **Camera-ready hardening:** a 2nd GPU architecture (one cloud rental) + Nsight occupancy/L2/
  bandwidth (done for the key kernels) materially raise the ceiling.

## Title (working)
"The Two Faces of Abstraction Regret: How a GPU DSL's Execution Paradigm — not its Abstraction
Height — Bounds Irregular-Automata Performance."

## Confidence
HIGH: "abstraction regret" unused as our framework; Pennycook is hardware-axis; no DSL-vs-irregular
benchmark exists; LMS naming collision. MEDIUM (verify pre-submission): Hexcute final venue/numbers
(preprint); Tawa CGO'26 camera-ready; that no unpublished Gluon-vs-Triton irregular comparison exists.
