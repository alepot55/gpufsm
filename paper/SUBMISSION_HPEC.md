# IEEE HPEC 2026 — submission checklist & metadata

Target: **IEEE HPEC 2026** (30th Annual, fully **virtual**, 14–18 Sep 2026).
This is the fast, low-cost, IEEE-Xplore-indexed venue for the paper.

## Hard facts (verified 2026-06-27)
- **Paper deadline:** **7 July 2026, 23:59 AoE** (extended). Notification **19 Aug**, camera-ready **4 Sep**.
- **Submission system:** Microsoft CMT — https://cmt3.research.microsoft.com/HPEC2026/
- **Upload:** the **PDF** (`.pdf`/`.doc`/`.docx` accepted) — source is NOT required.
- **Length:** **6 pages, references and acknowledgments excluded.** Extra page = $200.
- **Template:** standard IEEE conference 2-column (we use IEEEtran `conference`).
- **Anonymity:** single-blind — papers are **not** anonymous (keep name + affiliation; arXiv preprint OK).
- **Proceedings:** accepted full papers submitted to **IEEE Xplore** → counts as an international conference publication.
- **Registration (presenting author must pay):** early (≤4 Sep) student IEEE/SIAM **$140**, non-member student **$180**. No APC.

## The file to submit
`paper/gpufsm_hpec.pdf` — built from `paper/gpufsm_hpec.tex` (separate from the 8-page full
version `gpufsm.tex` kept for a journal/SC extension). Verified: **body = 6 pages** (references
begin in the right column of p.6 and spill to p.7; refs are excluded from the limit), **0
overfull boxes**, no undefined refs. Rebuild: `pdflatex gpufsm_hpec.tex` (twice).

What was trimmed from the 8pp version to fit 6 body pages (content preserved):
- Removed 4 redundant figures (cost-model fit, throughput-vs-states, worklist-speedup,
  memory-ablation); kept the two centerpieces (abstraction-regret bar, DFA two-faces curve).
- Inlined the Nsight table and the SOTA-positioning table into prose.
- Compressed the cost-model section (kept the inline equation + holdout result, dropped the figure)
  and the kernel-limitations paragraph. The 2×2, capability→cost map, causal control, two faces,
  and A100 cross-arch result are all intact. All 35 references kept (they don't count).

## Plain-text abstract for the CMT "abstract" field (≤5000 chars; this is ~1.7k)

GPU domain-specific languages (DSLs) such as OpenAI Triton deliver near-CUDA performance at far
lower effort on regular tensor algebra. We ask what that abstraction costs on irregular workloads
and answer it with a metric we call abstraction regret: the performance a DSL forecloses, with the
algorithm held fixed, because it cannot express the memory layout or control flow a workload needs.
We decompose regret along these two capability axes and instantiate it on finite automata across
the paradigm axis CUDA and NVIDIA Warp (thread-SIMT) versus Triton and its low-level Gluon frontend
(tile-SPMD). Automata expose two complementary faces: an NFA active-set traversal that is
control-flow bound, and a DFA dense-table walk that is memory bound (its throughput halves as the
table crosses L2). On both faces the regret is large for the tile-SPMD DSLs and small for the
thread-SIMT ones: Triton pays 5-13x versus CUDA across the two faces, while Warp, an equally
high-level Python DSL, matches or beats hand CUDA on the NFA (0.6-0.9x) and pays only 1.4-2.3x on
the DFA. So regret is set by the execution paradigm, not by how high-level the DSL looks. We make
the attribution falsifiable with the Triton-Gluon controlled pair (identical MLIR compiler stack;
Gluon only adds explicit layout/shared-memory control): Gluon still cannot express the kernel, so
the binding constraint is the paradigm, not tuning or layout. A two-parameter cost model
corroborates the regret (predictive for the thread model; holdout 2.7%) and we name the missing IR
primitives (scalar gather in a tile, register-resident bitset, data-dependent loop). Along the way
we build a portable work-efficient automata engine (~330x-10^4x over a faithful full scan, 15-170
Gbps, validated bit-for-bit against a CPU oracle on six real ANMLZoo automata up to 48k states). We
confirm the centerpiece on a second GPU architecture (NVIDIA A100): the 2x2 regret pattern and the
architecture-independent tile-SPMD scalar ceiling both reproduce.

## Suggested CMT metadata
- **Title:** The Two Faces of Abstraction Regret: Control-Flow and Memory-Layout Limits of GPU DSLs on Irregular Automata
- **Authors:** Alessandro Potenza — Politecnico di Milano (alessandro1.potenza@mail.polimi.it)
- **Topics/keywords** (pick from CMT list): GPU computing; performance characterization;
  domain-specific languages / compilers; high-performance computing; benchmarking; automata / pattern matching.

## Steps
1. Create/login CMT account, open the HPEC2026 site, "Create new submission."
2. Paste title + the plain-text abstract above; add authors + keywords.
3. Upload `gpufsm_hpec.pdf`. Confirm it renders as 6 body pages + references.
4. Submit before **7 Jul 23:59 AoE**.
5. (Optional, single-blind allows it) post the full 8pp version to arXiv for priority/visibility.
6. On acceptance (19 Aug): prepare IEEE-Xplore-compliant camera-ready by 4 Sep; register ($140 student, early).
