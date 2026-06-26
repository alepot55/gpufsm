# arXiv submission — checklist & metadata

The paper is preprint-ready. The submission tarball is built by `bash paper/arxiv_build.sh`
(→ `paper/arxiv_gpufsm.tar.gz`): a self-contained source bundle (gpufsm.tex + IEEEtran.cls +
the 6 figures), inline bibliography (no .bib/.bbl needed), verified to compile clean-room in
8 pages, 0 overfull boxes, no undefined refs.

## What you do (requires your account; cannot be automated)
1. **arXiv account** at https://arxiv.org — first CS submission needs an **endorsement**
   (cs.DC). If you lack one, arXiv shows endorsers to request; a co-author/advisor with prior
   cs.DC submissions can endorse instantly.
2. **New submission → upload** `paper/arxiv_gpufsm.tar.gz`. arXiv runs pdflatex itself; check
   the rendered PDF matches `paper/gpufsm.pdf`.
3. Paste the metadata below.
4. **License:** CC BY 4.0 recommended (max reuse; standard for artifacts). arXiv non-exclusive
   is the minimum.
5. Submit. (Optional) hold for the 8 pm ET announcement cutoff for next-day listing.

## Metadata to paste

**Title:**
The Two Faces of Abstraction Regret: Control-Flow and Memory-Layout Limits of GPU DSLs on Irregular Automata

**Authors:** Alessandro Potenza

**Primary category:** cs.DC (Distributed, Parallel, and Cluster Computing)
**Cross-list:** cs.PL (Programming Languages), cs.PF (Performance)

**Comments:** 8 pages, 6 figures, 5 tables. Artifact (code, data, all figures regenerable
from versioned CSVs): https://github.com/alepot55/gpufsm

**Abstract (plain text):**

GPU domain-specific languages (DSLs) such as OpenAI Triton deliver near-CUDA performance at far
lower effort on regular tensor algebra. We ask what that abstraction costs on irregular
workloads and answer it with a metric we call abstraction regret: the performance a DSL
forecloses -- with the algorithm held fixed -- because it cannot express the memory layout or
control flow a workload needs. We decompose regret along these two capability axes and
instantiate it on finite automata across the paradigm axis CUDA and NVIDIA Warp (thread-SIMT)
versus Triton and its low-level Gluon frontend (tile-SPMD). Automata expose two complementary
faces: an NFA active-set traversal that is control-flow bound, and a DFA dense-table walk that
is memory bound (its throughput halves as the table crosses L2). On both faces the regret is
large for the tile-SPMD DSLs and small for the thread-SIMT ones -- Triton pays 5-13x versus
CUDA across the two faces, while Warp, an equally high-level Python DSL, matches or beats hand
CUDA on the NFA (0.6-0.9x) and pays only 1.4-2.3x on the DFA. So regret is set by the execution
paradigm, not by how high-level the DSL looks. We make the attribution falsifiable with the
Triton<->Gluon controlled pair (identical MLIR compiler stack; Gluon only adds explicit
layout/shared-memory control): Gluon still cannot express the kernel, so the binding constraint
is the paradigm, not tuning or layout. A two-parameter cost model corroborates the regret
(predictive for the thread model; holdout 2.7%) and we name the missing IR primitives (scalar
gather in a tile, register-resident bitset, data-dependent loop). Along the way we build a
portable work-efficient automata engine (~330x-10^4x over a faithful full scan, 15-170 Gbps,
validated bit-for-bit against a CPU oracle on six real ANMLZoo automata up to 48k states). We
confirm the centerpiece on a second GPU architecture (NVIDIA A100): the 2x2 regret pattern and
the architecture-independent tile-SPMD scalar ceiling both reproduce.

## After it lands
- Add the arXiv ID to `CITATION.cff` and `README.md`.
- Cut a tagged release and mint a **Zenodo DOI** for the artifact (see `docs/ARTIFACT_APPENDIX.md`).
- Make the GitHub repo public (currently private; see CLAUDE.md §2.4).
