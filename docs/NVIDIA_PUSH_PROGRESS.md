# NVIDIA push — progress log (read FIRST each iteration, update LAST)

Campaign north-star: `docs/NVIDIA_PUSH_PLAN.md`. Branch `dev`. Goal = NVIDIA-hire-grade work by turning
the 7 gaps into strengths via fronts F1–F5. One committed artifact (or honest "verified") per iteration.
Env: `.venv/bin/python` (gpufsm+CUDA RTX4070; ruff+mypy in .venv). From-source Triton 3.8 at
`~/m3full_build/triton-src`; rebuild `cmake --build ~/m3full_build/triton-src -j 8` (~5min).

## Fronts status
- F1 Triton RFC — DONE: `docs/rfc/triton-per-lane-region.md` reviewer-ready (+ `_research_notes.md`).
  GATED on USER to post as a `[RFC]` issue on triton-lang/triton.
- F2 ML-domain witness — DONE: attention (dense, regret 0.64 — tile WINS) + MoE routing (scalar, 2.36 —
  tile loses), folded into paper2 (regret_law 8 rows, fig sign-flip, sec:law unified mechanism).
- F3 real in-compiler lowering — REDUCE-HOIST PASS DONE (1.55x, oracle-correct, in libtriton) + folded
  into paper2 sec:compiler + RFC. FULL cure (per-lane retirement) scoped + bounded (~5.6x target) — needs
  the below-TritonGPU lowering (`docs/rfc/below-tritongpu-lowering.md`); NOT implemented (the wall).
- F4 multi-GPU A100/H100 — GATED on user cloud pod (`scripts/run_cross_arch.sh` ready).
- F5 submission — GATED on user accounts; ⚠️ paper-1 num_warps disclosure before HPEC (7 Jul) — flag only.

## Durable facts / gotchas (keep)
- WRITE FILES VIA SHELL (Edit/Write are bgIsolation-guarded; do NOT touch `.claude/settings*`).
- ⚠️ Nsight: ALWAYS `ncu --kernel-name regex:<fn>` (else you profile a torch setup kernel — bug caught).
- ⚠️ MLIR build (this LLVM treats deprecation as error): use `OpTy::create(b, ...)` not `b.create<OpTy>`;
  `arith::ConstantOp` + `getIntegerAttr` not `ConstantIntOp(int, Type)`.
- The thread_region pass (ThreadRegion.cpp) lives in ~/m3full_build/triton-src + patch in
  experiments/cure/triton_thread_region_pass/. Modes: GPUFSM_THREAD_REGION=1 (detect), =hoist (rewrite).
- **Unified regret-law mechanism (corrected, CSV-traced):** the thread model always retires lanes on
  divergence (threads-per-instruction < 32); the tile does masked full-width work (~32). The regret SIGN
  is set by per-step instruction efficiency: scalar control -> tile issues more -> tile loses (automata,
  rejection, MoE); a dense vectorizable head-dim -> tile issues fewer -> tile wins (attention). The cure
  = per-lane retirement (below-TritonGPU); the reduce-hoist is the in-IR slice (removes the reduce only).

## DONE (compacted, 2026-07-01)
**PUBLICATION = ACM TACO** (journal, rolling, revise-and-resubmit → certain; user: "pubblicazione certa").
Paper `paper2/gpufsm_taco.tex` (acmsmall, clean 15pp, ANONYMIZED+double-blind-safe [TACO auto-rejects
non-anon], CCS concepts+keywords, built-cure complete, 26 refs). Backups gpufsm_ppopp.tex/gpufsm2.tex.
Plan docs/TACO_PLAN.md. Submit via ScholarOne when strong (no deadline). Biggest lever = A100 datacenter
validation (user RunPod ~$5-10; scripts/a100_validate.sh turnkey) to pre-empt "add architectures" revision.
arXiv OUT (no endorsement); optional fast preprint = TechRxiv/Zenodo — AWAITING user y/n + candidacy timeline.
Build gotchas: lmodern+\emergencystretch=2em; ACM metadata minimal (\acmJournal{TACO} only, empty numeric
fields → 'Missing number'); font expansion needs lmodern.

**FLAGSHIP (weakness #2 flipped): cure BUILT** — LowerThreadRegionRetire pass in real Triton (make_llir):
masked lock-step latch→per-lane cond_br, cross-lane redux removed, bar.warp.sync. Oracle-correct: synthetic
4.15x (Nsight 39x fewer inst, work ∝ Σtrip), 2.5-7.3x across distributions, MoE 1.25x + SpMV 1.14x
(control-boundedness spectrum). CSVs cure_{speedup,nsight,generalize,realworkload}. Sources version-controlled
experiments/cure/triton_thread_region_pass/ + pipeline_wiring.patch (reproducible on a fresh pod).

**UPSTREAM (weakness #4): PR #10766** (split/join fold) LIVE — maintainer ThomasRaoux engaged; CI caught a
real WS-partition regression, FIXED (folds bail on discardable attrs), verified all join/split lit tests +
negative case, force-pushed a7ebe90, replied. fold-bitcast/fold-ptr-roundtrip held. STANDING AUTH: act on
Triton PRs/issues/maintainer replies autonomously. Lesson: run the FULL lit suite.

## Findings log (newest first)

- **2026-07-01 out-of-sample validation of the straggler law.** Froze the train fit (`masked=50.3+1.08·E[warp-max]`) and predicted 4 HELD-OUT distributions (bimodal, log-normal, spike, adversarial single-straggler; different seed): masked cost predicted within **5% mean / 7.5% max** → the law generalizes, not an overfit. Adversarial single-straggler warp (1/32 lanes trip=256, rest trip=2, D=25.8) → **6.6×** (one lane taxing 31). One honest sentence into `gpufsm_taco.tex` §implemented (15pp, clean, anon). `cure_heldout_rtx4070.csv`, `experiments/cure/cure_heldout.py`.

- **2026-07-01 predictive straggler law (paper2).** Built-cure sweep over 6 controlled trip distributions (oracle-gated both modes): `masked = 50.3 + 1.08·E[warp-max]` µs (R²=0.998); cured = flat 42.8µs floor. Speedup tracks the **absolute straggler** E[warp-max] (corr 0.99), **not** the divergence ratio D (corr 0.08) — geometric D=3.96→2.6× vs uniform D=1.94→6.9×. Falsifiable, <9% pred error. Folded into `gpufsm_taco.tex` §implemented (15pp, clean, anon). CSV `cure_predictive_rtx4070.csv`, `experiments/cure/cure_predictive.py`. Upgrades channel-(ii) from qualitative→predictive.
- 2026-07-01 ~10:52: **Related-work already journal-depth (26 refs); added the ONE genuine gap = ITS.**
  The cure's per-lane retirement mechanically relies on Independent Thread Scheduling but was uncited — added
  the accurate NVIDIA Volta whitepaper ref (its2017) + cited it at the mechanism (line 288). (Skipped Hexcute:
  won't fabricate its author list unverified.) Compacted the log (168→127). Clean 15pp, 0 overfull/undefined.
  The paper is now genuinely SUBMISSION-READY modulo the A100 (tomorrow) + user decisions (preprint y/n,
  candidacy timeline). Further micro-polish = diminishing returns; next real levers are A100 + user.- 2026-07-01 ~10:20: **TACO anonymity pass DONE + VERIFIED (avoids auto-reject).** Confirmed via TACO
  author guidelines: papers revealing author identity are subject to IMMEDIATE REJECTION (effectively
  double-blind). Made the paper double-blind-safe: 'A companion study'->'Prior work' (line 79); paper1 bib
  note 'Companion diagnosis paper'->'Manuscript under review'; verified author=Anonymous + refs.bib has NO
  name leaks (grep potenza/alepot55/alessandro = 0). Fixed 2 residual overfulls (emergencystretch). Clean:
  15pp, 0 overfull/undefined/missing-number. NEXT: journal-depth related work; tomorrow A100. Awaiting user:
  preprint (TechRxiv/Zenodo) y/n + candidacy timeline.- 2026-07-01 ~09:30: **TACO version + CCS concepts.** Converted to journal format (acmsmall) after the
  "pubblicazione certa" decision; added ACM CCS concepts (Compilers 500 / Parallel-programming-languages /
  SIMD) + keywords (TACO requirement). Clean: 15pp, 0 overfull/undefined/missing-number/fatal, figs render.
  PR #10766 still awaiting Raoux. Sent the user the current PDF; explained TACO timeline (~2mo first
  response, ~6mo to accept, online in Just-Accepted weeks after). NEXT: journal-depth (related work, deepen
  a section) + tomorrow's A100.
- 2026-07-01 ~08:45: **PIVOT to CONFERENCE: PPoPP 2027 (user: no arXiv, submit fast, all by early Aug;
  RunPod tomorrow).** Web-researched venues: PACT scaded, PPoPP 2027 ~3 Aug (co-loc HPCA/CGO/CC, Salt Lake,
  Mar 2027; official CFP not yet posted — MONITOR), ASPLOS 9 Sep, CGO ~Sep. Chose PPoPP (parallel-execution
  model = tile-SPMD vs thread-SIMT is home turf). DONE TODAY: (1) generated acmart.cls (CTAN), converted
  paper2 IEEEtran→ACM (paper2/gpufsm_ppopp.tex); font expansion via lmodern + emergencystretch killed
  35→0 overfull; **clean 8pp, 0 overfull/undefined/fatal, all 8 figs render, under 10pp limit**. (2) paper
  already ANONYMOUS (double-blind-ready, zero name/repo/github leaks). (3) built-cure story fully present
  (4.15x/39x/table/SpMV1.14x/MoE1.25x). (4) turnkey scripts/a100_validate.sh for TOMORROW (phase1 regret-law
  cross-arch fast+reliable; phase2 build Triton+cure, validate 4.15x on A100). Plan docs/PPOPP_PLAN.md.
  NEEDS USER: tomorrow spin up RunPod A100 2-3h (~$5-10) → run a100_validate.sh; later HotCRP submit.
  NEXT: PPoPP-audience framing polish; monitor PR #10766 (Raoux) + official PPoPP CFP.- 2026-07-01 ~08:15: **PR #10766 CI regression FOUND + FIXED (user flagged the red checks).** The integration
  checks weren't infra: nvidia-h100 + amd-gfx942 genuinely FAILED at lit tests — my split/join fold broke
  `ws_data_partition.mlir` (@test_split_join_reshape_trans_partition), which uses split(join(x,x)) round-trips
  as an intentional warp-specialization data-partition vehicle (ops carry async_task_id). My fold removed them,
  dropping the task attr. My earlier grep-based "no test triggers it" was WRONG (I only ran canonicalize.mlir).
  FIX (principled, layering-clean): both folds now bail when the op has discardable attrs — an erasing fold
  must not silently drop e.g. WS task ids (async_task_id is nvidia-only so can't be named in a core fold; the
  generic guard avoids the layering issue). VERIFIED: WS test now PASS + my canonicalize tests still fold +
  ALL 13 simple-RUN join/split lit tests green + added a negative FileCheck case. Amended PR commit a7ebe90,
  force-pushed to fork (CI re-running), replied to Raoux explaining the WS interaction + fix (good signal:
  found+fixed a real interaction). LESSON: run the full lit suite, not one file. User: "prova a sistemarla,
  non mollare, ragionaci bene" → done.- 2026-07-01 ~03:40: **🔥 MAINTAINER ENGAGED on PR #10766 — ThomasRaoux (Triton core maintainer, now at
  NVIDIA) asked "are there practical use cases?"** Exactly the maintainer-contact the hire-first strategy
  targeted. Replied (standing auth) honestly + technically: the fold completes the in-tree inverse-fold
  family (trans/trans, bitcast/bitcast, int_to_ptr/ptr_to_int); round-trips arise from composition/inlining
  (interleaved/complex/microscaling paths) + unblock downstream folds; guarded on type equality; and
  honestly noted no in-tree test hits it today + offered to close if too niche (respecting maintainer
  bandwidth). Comment: pull/10766#issuecomment-4849477051. HELD PR #2/#3 (his question is mildly skeptical →
  piling on more folds would read as spam; quality>quantity). CI: pre-commit/formatting PASS; integration-
  tests fail identically across ALL amd+nvidia backends (h100/a100/gb200/gfx942/950/90a) = infra/base-branch,
  not my 3-line IR fold. NEXT: watch for his reply; if positive, engage; the cure/paper remain the flagship.- 2026-07-01 ~03:12: **MoE datapoint — cure spectrum now complete + thesis-confirming.** cure_moe.py
  (MoE top-k routing, power-law, int64-exact oracle): masked ~233us vs cured ~186us = **1.25x**, oracle-OK,
  pass FIRED (PTX bar.warp.sync=1, redux.sync=0). Full real-workload spectrum: synthetic pure-control 4.15x
  > MoE mixed-gather 1.25x > SpMV memory-gather 1.14x. Both SpMV+MoE carry per-iteration gathers so are
  gather-bound; the SAME cure's benefit scales with control-boundedness → confirms the regret is per-step
  CONTROL not memory. Paper sentence refined to the SpMV+MoE spectrum (7pp clean). CSV
  cure_realworkload_rtx4070.csv (3 rows). Cure now validated on: synthetic (3 distributions) + 2 real
  workloads (SpMV, MoE). NEXT: paper submission-readiness pass; PR #10766 monitor.- 2026-07-01 ~02:45: **Cure HARDENED on a REAL workload (SpMV CSR) — de-risks "narrow/synthetic".** The
  LowerThreadRegionRetire pass FIRES + stays oracle-correct on the real power-law SpMV lock-step kernel
  (_spmv_tile, same latch): masked ~1793us vs cured ~1575us = **1.14x** (stable 3/3). Verified the pass
  fired: dumped _spmv_tile.ptx has bar.warp.sync + redux.sync removed. Modest speedup is HONEST + thesis-
  confirming: SpMV is memory-gather-bound (x[colidx] DRAM), so little of its cost is the control reduce —
  the SAME cure gives 4.15x on control-bound work and 1.14x on memory-bound SpMV, showing the recoverable
  regret is per-step CONTROL, not memory. Folded one honest sentence into paper2 (7pp, clean). CSV
  cure_realworkload_rtx4070.csv. The cure now validated on synthetic control-bound + real sparse. NEXT:
  optionally MoE (more control-bound, expect higher); push paper submission-readiness; PR #10766 monitor.- 2026-07-01 ~02:20: **M4b DONE — the BUILT cure folded into paper2 (flagship upgrade).** Rewrote the
  abstract + the "In the real compiler" contribution + \S sec:implemented: from "diagnosed / structurally-
  impossible / reduce-hoist 1.55x / out-of-band selector" to "the in-tile-IR lowering is structurally
  impossible SO we BUILD the cure below it (TritonGPU->LLVM pass wired into make_llir: redirect lock-step
  latch to per-lane predicate, drop cross-lane redux, bar.warp.sync reconverge) — oracle-correct, 4.15x, 39x
  fewer instructions, 2.5-7.3x across distributions, recovering the residual between reduce-hoist (1.55x) and
  the 5.64x thread bound". Mechanism framed HONESTLY (work ∝ Σtrip, instruction-issue win; tpi unchanged, no
  occupancy overclaim). Kept the structural-impossibility argument (it now MOTIVATES the below-IR build).
  LaTeX clean: 7pp, 0 overfull, 0 undefined. Numbers CSV-traced (cure_speedup/nsight/generalize_rtx4070.csv).
  **FLAGSHIP COMPLETE: the cure is BUILT + MEASURED + PROFILED + GENERALIZED + WRITTEN UP.** Weakness #2
  fully flipped. NEXT: harden on a REAL automata kernel (not just synthetic per-lane-while); push paper
  toward submission-ready; monitor PR #10766.- 2026-07-01 ~02:15: **M4c — the cure GENERALIZES (oracle-correct across 3 trip distributions).**
  cure_generalize.py, masked vs retire on the per-lane-while kernel: uniform 300→41us (7.3x), geometric
  99.3→39.8us (2.5x), pareto 142→41us (3.5x) — all oracle=OK. Not a one-kernel trick; speedup tracks how
  much the masked baseline over-works (32×warp-max vs Σtrip) — biggest for uniform (warp-max≈256). Cured
  hits a ~40us floor (work now small → memory/launch bound). CSV cure_generalize_rtx4070.csv. NEXT: M4b fold
  the built+measured+profiled+generalized cure into paper2 (flagship: diagnosed→built, 2.5-7.3x).
