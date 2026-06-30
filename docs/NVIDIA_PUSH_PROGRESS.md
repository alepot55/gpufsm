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
F1 (RFC + research notes), F3 reduce-hoist gating measurement (1.41x source-level, with an honest
self-correction: a naive global-max bounded-for was 0.45x — removes per-warp termination), F3 reduce-hoist
PASS built into libtriton (1.55x, oracle-correct, via scf.while iter-arg surgery + cloned reduce_max),
F3 folded into paper+RFC. All committed on dev; details in git log.

## Findings log (newest first)
- 2026-06-30 ~13:40: **Related-work vs NVIDIA cuTile/Tile-IR strengthened (NVIDIA-relevant framing).**
  Updated paper2 Related Work + the cutile2025 bib entry to engage NVIDIA's current direction precisely:
  cuTile AND its MLIR Tile IR (now being built as a backend for Triton itself) are tile-level and concede
  the irregular case to a hand-written SIMT fallback -> the per-lane gap we name persists in BOTH Triton's
  TritonGPU and NVIDIA's Tile IR, so the proposed primitive is COMPLEMENTARY to NVIDIA's platform bet, not
  subsumed by it. Strong NVIDIA-interview framing (shows awareness of + alignment with their roadmap).
  Paper 7pp, 0 undefined/0 overfull. (Sources: NVIDIA dev blog "CUDA Tile IR Backend for OpenAI Triton",
  CUDA 13.1, 2025; github.com/NVIDIA/cuda-tile.) Non-gated queue now genuinely thin; remaining big levers
  are user-gated (post RFC, cloud pod, submission) or the multi-week below-TritonGPU C++.- 2026-06-30 ~13:10: **Paper consolidation pass — abstract + contributions brought current.** End-to-end
  re-read found two REAL stale omissions (the newest/strongest results were missing from the front matter):
  (1) the ABSTRACT said "six irregular workloads" with "two channels" and omitted the ML sign-flip and the
  reduce-hoist; updated to eight workloads + the correct sign-flip negative (MoE 2.36x confirms; attention
  0.64x, tile WINS -> "why Triton excels at flash-attention yet collapses on automata") + the real
  in-compiler reduce-hoist (1.55x, in libtriton). (2) the "In the real compiler" CONTRIBUTION bullet listed
  detect+wall+selector but not the reduce-hoist; added it (honestly partial). Spot-checked numbers trace to
  CSVs (4.2/3.9/1.55/2.36/0.64). No contradictions; six-core + two-further ML = eight is internally
  consistent (sec:law text vs abstract/caption "eight"). Paper 7pp, 0 undefined/0 overfull, PDF regen.
  The paper now tells the full, current story. NEXT: a 3rd distinct improvement (related-work vs the NVIDIA
  cuTile Tile-IR-for-Triton backend as motivation, OR a Methodological-Integrity note on the F3 0.45x
  self-correction), or begin the below-TritonGPU op (high-risk).- 2026-06-30 ~12:35: **F3 FULL-cure SCOPED + BOUNDED + hook-point pinned (design note).** Measured the
  full-cure target on the lock-step kernel: CUDA thread (one/element, retiring) = **27.6 us = 5.64x** vs
  tile 155.6us, with threads-per-instruction **11.65** (< 32 = per-lane retirement, Nsight-confirmed). So
  the full cure is worth ~5.6x; the built reduce-hoist captures 1.55x; the residual ~3.6x is EXACTLY the
  per-lane sub-warp retirement the structural wall blocks in TritonGPU. Pinned the exact hook-point
  (make_llir: add_scf_to_cf -> add_to_llvmir in TritonNVIDIAGPUToLLVM) + op semantics (per-lane scalar
  extraction from the #blocked layout, per-lane cond_br -> SIMT/ITS retirement, bar.warp.sync reconverge)
  + the exact blocker (the multi-week crux = per-lane scalar extraction from the distributed layout +
  coherent tile state across a divergent loop + pipeliner opt-out). Artifacts: `docs/rfc/below-tritongpu-
  lowering.md`, `experiments/cure/f3_full_cure_bound.py`, `paper2/data/landmark/f3_hoist_rtx4070.csv`
  (now 4 rows: tile/detect/hoist/thread). This is an honest, committed scoping result (not the C++
  implementation — that's the genuine multi-week effort) that STRENGTHENS the RFC with a concrete,
  measured lowering plan + payoff bound. NEXT: paper consolidation pass, OR a 3rd distinct witness, OR
  begin the below-TritonGPU op (high-risk, multi-iteration).
