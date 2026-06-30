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

## DONE (compacted, 2026-06-30)
Git consolidated to main+dev. Campaign plan + env set up. F2 (2 ML witnesses + sign-flip in paper),
F1 (RFC + research notes), F3 reduce-hoist measurement + PASS in libtriton (1.55x). Earlier 2026-06-30
(pre-pivot): gap#7 ownership doc `docs/NVIDIA_INTERVIEW_OWNERSHIP.md`; paper2 related-work engages cuTile/
Tile-IR (complementary framing); abstract→8 workloads + sign-flip; F3 full-cure SCOPED+BOUNDED = 5.64x
(=1.55x in-IR reduce-hoist + ~3.6x below-TritonGPU per-lane retirement; hook-point make_llir pinned,
`docs/rfc/below-tritongpu-lowering.md`). ⚠️ LICM dead-end: -triton-licm already hoists loop-invariant
`tt.reduce` (Pure) → reduce-hoist is NOT a mainstream PR (paper artifact only). All committed on dev.

## Findings log (newest first)
- 2026-06-30 ~23:15: **🎉 FIRST UPSTREAM PR LIVE — triton-lang/triton#10766 (split/join inverse fold).**
  User chose "open the strongest one first". Installed clang-format 19.1.6 → confirmed the branch is
  format-clean (no pre-commit churn). Forked triton-lang/triton→alepot55/triton, pushed fold-split-join,
  opened PR #10766 (OPEN, MERGEABLE, +56/3 files, author alepot55, title "[TRITON] Fold split(join(a,b)) ->
  (a,b) and join(split(x)) -> x"). CI not yet triggered (Triton gates CI for first-time contributors until a
  maintainer approves). This DIRECTLY closes weakness #4 (zero upstream contribution) — a real maintainer-
  reviewable PR in NVIDIA's co-maintained codebase. fold-bitcast@b68445d + fold-ptr-roundtrip@0541b42 HELD
  in reserve (open after a maintainer engages, per the chosen strategy). Design issue still optional/pending.
  NEXT (loop): monitor PR #10766 each wake (gh pr view comments + checks); address maintainer feedback /
  CI failures (fix in branch → rebuild → re-verify FileCheck → push fork); if positively engaged, follow
  with PR #2/#3.
- 2026-06-30 ~19:45: **CI regression-safety of the 3 PRs VERIFIED (content grep, no rebuild needed).**
  Checked the whole `test/` tree for the exact round-trip patterns each fold matches: ZERO tests have
  consecutive `tt.bitcast`; NO test has a `split(join)`/`join(split)` round-trip (pipeline tests carry a
  standalone `tt.join` only, split=0); the only `ptr_to_int(int_to_ptr)` round-trip is our new positive
  case (`ops.mlir` uses INDEPENDENT casts + no `-canonicalize`). ⇒ none of the 3 folds can fire on any
  existing test → no CI regression on the touched ops. Added a "Regression safety (verified)" note to each
  PR doc. This de-risks the #1 PR-bounce cause without the 3 expensive rebuilds. Still awaiting USER push.
- 2026-06-30 ~18:50: **Log compacted 131→62. Easy fast-relink fold space EXHAUSTED (honest).** Final
  triton-opt sweep: reshape(splat) and convert_layout(same-layout) ALREADY fold; expand_dims(expand_dims)
  survives but is niche AND needs a .td rebuild — not worth it. No new high-value mergeable fold this wake.
  3 clean verified PRs + the design issue is a stronger signal than a pile of trivial folds; deliberately
  NOT manufacturing marginal PRs. Real bottleneck = USER pushing the 3 ready branches / posting the issue.
- 2026-06-30: **ARM 1 — THREE mergeable Triton PRs built+verified, awaiting USER push.** Method each: real
  gap reproduced with triton-opt FIRST → implemented → FileCheck green on full test/Triton/canonicalize.mlir
  → isolated to a clean branch off upstream c05aa65 + patch/PR-doc in docs/upstream/. Reviewer @lezcano
  (template merged #10734/#9971):
  (1) `fold-split-join`@b5c33a4 — tt.join/tt.split mutual inverses but fold-less; added JoinOp/SplitOp folds
      guarded on exact type equality. split(join(a,b))->(a,b), join(split(x))->x. (3 files/56 ins)
  (2) `fold-bitcast`@b68445d — extended BitcastOp::fold to collapse nested bitcasts (round-trip->x,
      A->B->C chain->single A->C). (2 files/34 ins)
  (3) `fold-ptr-roundtrip`@0541b42 — hasCanonicalizer on TT_PtrToIntOp + mirror of existing
      CanonicalizeIntToPtrOfPtrToInt, completing the inverse pair: ptr_to_int(int_to_ptr(x))->x. (3 files/39 ins)
  ⚠️ I CANNOT push to triton-lang — USER pushes the 3 branches + opens PRs + shares links → I handle review.
- 2026-06-30 ~15:10: **DIRECTION = HIRE-FIRST** (user "più in alto, più grande"): 4-agent research + Triton
  recon → maximize NVIDIA signal via upstream PRs (plan docs/upstream/STRATEGY.md). Design issue draft
  docs/upstream/triton-issue-irregular-control.md ALSO pending user post (earns a maintainer thread w/o a
  niche-merge ask). Reframe = "characterized+partially closed a regret class the CUDA Tile-IR backend
  exhibits" (NVIDIA team Jie Xin/Jonathan Bentz), not "add a primitive". Cure-RFC primitive de-prioritized.

## Verified fold dead-ends (do NOT re-investigate)
reduce-hoist→no LICM PR; broadcast(broadcast/splat), expand_dims(splat), reshape(reshape/same),
trans(trans), addptr(addptr/zero), int_to_ptr(ptr_to_int), bitcast(bitcast) [now shipped], reshape(splat), convert_layout(same) ALREADY fold.
trans(splat)->splat is a REAL gap but needs a canonicalizer + .td rebuild (niche, low ROI).
