# PPoPP 2027 — submission packet (paper 2, "the cure")

Everything below is ready. The only remaining work is a **USER action on HotCRP** (create account,
paste, upload, submit). Prepared 2026-07-25.

## Venue facts (verified 2026-07-25 from conf.researchr.org/track/PPoPP-2027/PPoPP-2027-papers)

| item | value |
|---|---|
| Site | https://ppopp27-summer.hotcrp.com/ |
| **Paper deadline** | **Mon 3 Aug 2026, AoE** |
| Separate abstract deadline | **NONE** — abstract is submitted with the paper |
| Page limit | 10 pages of **text and figures**, references **excluded** (no cap on refs) |
| Format | ACM `acmart`, **sigplan** option, 10 pt body, captions >=9 pt, figure/table fonts >=8 pt |
| Review | **Double-blind** — no author identity anywhere in the PDF |
| ORCID | Required to complete publishing |
| Artifact Evaluation | Optional, after acceptance (submission 9 Nov 2026) |
| Rebuttal | 6-8 Oct 2026 · Notification 26 Oct 2026 · Final paper 18 Dec 2026 |
| Resubmission | Allowed until the deadline; the last uploaded version is the one reviewed |

## Our compliance check (verified against the built PDF, 2026-07-25)

- **Page budget: 9 pp text+figures / 10 allowed.** Total PDF is 10 pp; references start on p.10.
  A `\clearpage` before the bibliography guarantees no figure lands after the references (before this,
  Figures 4 and 6 fell on p.9 *after* the reference list, which both read badly and muddied the count).
- **LaTeX: 0 overfull hbox, 0 undefined references, 0 errors.** 26 bibitems, 26 unique citations, all resolved.
- **Anonymity: clean.** PDF metadata carries no Author field; no occurrence of the author name,
  affiliation, `gpufsm`, or a personal GitHub URL anywhere in the text. The only `Anonymous` reference
  left is our own paper 1 (correct under double-blind). The two GitHub URLs present are citations to
  NVIDIA Warp and NVIDIA cuda-tile, which are legitimate third-party references.
- **Format:** acmart sigplan, as required.

## Abstract for the HotCRP form (plain ASCII, no unicode — paste as is)

Tile-based GPU DSLs (Triton) trail hand-written CUDA by ~10x on irregular finite-automata workloads.
Prior work attributed this "abstraction regret" to the execution paradigm (tile/SPMD vs. thread-SIMT)
rather than abstraction height, but left open why, mechanistically, and what would close it. We answer
both with an anatomy and a cure.

On a work-efficient NFA worklist the regret is batch-dependent (10.1x at batch 4096, 26x at a
GPU-saturating batch) and decomposes, staged, into a launch-configuration artifact (default num_warps;
2.8-3.7x), a lane-packing-recoverable component, and an irreducible ~2x residual. Nsight localizes the
residual to abstraction-denied intra-warp latency hiding: at matched occupancy, with fewer
warp-instructions than CUDA and below both roofline ceilings, the per-lane tile still spends 15.3x more
cycles in the dependent-load stall (9.9% vs. 41% issue activity). A CUDA warp overlaps its lanes'
independent in-flight loads; the lock-step tile serializes the dependent next-state load. This unifies
both faces of the regret: for the memory-bound DFA the residual closes (1.05x) once the table spills to
DRAM. The staged decomposition reproduces on an A100 (same 26x total; num_warps component 3.7x on both),
so the anatomy is a property of the paradigm, not one GPU.

We name the missing IR primitive (a per-lane sub-tile loop/exit that lowers each lane to an independent
instruction stream) and prove the in-tile-IR lowering is structurally blocked (scf.condition is a single
i1). Since the tile IR cannot express it, we build the cure below it: a TritonGPU->LLVM
per-lane-retirement pass wired into libtriton, oracle-correct and guarded by a soundness verifier.
Rebuilt from a pinned one-command recipe, it gives 2.3-6.7x on control-bound lock-step kernels on an RTX
4070, reproduces at 1.6-3.8x on an A100 from the same artifact, and, consistent with the law, yields no
gain on gather-bound SpMV/MoE (~1.0x). Finally, a generality law across eight oracle-gated workloads
shows the regret is created by per-step scalar control, not memory irregularity (pointer-chase negative
control 1.00x); on ML-shaped witnesses under a one-element-per-lane mapping it holds (MoE 2.36x; a
sign-flip on attention, 0.64x, where the tile wins). Every kernel is correctness-gated against a CPU
oracle; every number traces to a versioned CSV.

## Other HotCRP form fields

- **Title:** From Diagnosis to Cure: Decomposing the Tile-SPMD Abstraction Regret on Irregular Automata
- **Authors:** Alessandro Potenza, Politecnico di Milano, alessandro1.potenza@mail.polimi.it,
  ORCID 0009-0004-6106-139X. (HotCRP hides author identity from reviewers; fill it in truthfully.)
- **PC conflicts:** none to declare. Politecnico di Milano is the only affiliation.
- **Keywords:** GPU compilers, tile DSLs, Triton, SIMT, warp-level parallelism, irregular parallelism,
  per-lane control flow
- **File to upload:** `paper2/ppopp-submission-anonymous.pdf` (neutral filename on purpose — the repo
  name "gpufsm" would de-anonymize).

## Submission steps (user)

1. **The site is OPEN as of 2026-07-25** (it was closed on 08-Jul; the chairs flipped it open in the
   meantime, as predicted). Go to https://ppopp27-summer.hotcrp.com/ .
   > **Deadline caveat:** HotCRP displays `Monday Aug 3, 2026, 11:59:59 AM AoE` — **AM**, i.e. *noon*
   > AoE, not the usual end-of-day. Read twice on two separate fetches. Noon AoE on 3 Aug = 02:00 CEST
   > on 4 Aug; if it is really a chair typo for PM it would be 14:00 CEST on 4 Aug. **Do not gamble on
   > which:** submit by **1-2 Aug** and the ambiguity is moot.
2. Create an account with the polimi email.
3. New submission -> paste title, then the ASCII abstract above.
4. Add author (name, affiliation, email, ORCID). Mark PC conflicts (none).
5. Upload `paper2/ppopp-submission-anonymous.pdf`.
6. Submit. Re-uploading before the deadline is allowed and only the last version is reviewed.

**Recommended strategy: submit now, improve later.** Since the site is open and PPoPP reviews the last
version uploaded before the deadline, submitting the current PDF immediately costs nothing and removes
all deadline risk (site outages, the AM/PM ambiguity, account issues). Any further polishing is then a
re-upload, not a race.

## Integrity notes (what was checked before declaring this ready)

- **Every straggler-law number re-derived from the CSVs** by the new, versioned
  `experiments/cure/verify_straggler_law.py` (the fit was previously computed ad hoc and was not
  regenerable). The paper quotes sweep 2, the pinned wheel-recipe build: t = 32.4 + 1.09*E[warp-max] us,
  R^2 = 0.997, floor 43.0 +/- 1.6 us, corr(speedup, straggler) = 0.99 vs corr(speedup, D) = 0.06,
  single-straggler 7.19x, uniform 6.70x, geometric 2.26x. All reproduce exactly. Sweep 1 in the same
  CSV is the older local build (50.3 + 1.08) that the paper cites for build-stability; the two builds
  must never be pooled, and the script keeps them separate.
- **One number corrected:** the held-out max error is 2.15%, which the paper rounded *down* to 2.1%;
  now stated as 2.2%.
- **Five bibliography entries had placeholder authors** ("HybridSA authors", "AutomataBLAS authors",
  "Triton contributors", and two spurious "Anonymous") for third-party work. All replaced with the real
  author lists, verified via Crossref/arXiv. Only our own paper 1 stays Anonymous.
- **One factual error about cited work fixed:** the related-work text claimed Prism "lets a programmer
  hand-annotate typed thread/tile perspectives", but Prism's abstract states it works *without*
  hand-annotated perspective markers. Its title in our bib was also wrong (the paper is "Modular GPU
  Programming with Typed Perspectives"). Both corrected — the authors (MIT: Bansal, Amarasinghe,
  Ragan-Kelley) are plausible PPoPP reviewers. Tawa's title was likewise wrong and is fixed.
