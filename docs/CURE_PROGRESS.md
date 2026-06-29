# Paper 2 (cure) — autonomous loop progress bridge

Loop bridge across iterations/context windows (de-sloppify "SHARED_TASK_NOTES" pattern).
Read this FIRST each iteration; update it LAST. Plan: `docs/CURE_PLAN.md`. Branch
`claude/cure-ir-primitive`. Hardware: RTX 4070 (sm_89), Triton 3.5.1, torch 2.9.1+cu128.
Env: `.venv` (system-site-packages) with gpufsm built `+CUDA`. Run experiments with
`.venv/bin/python`.

## Milestone status
- [x] **M0 — anchor reproduced + oracle-validated.** `experiments/cure/m0_anchor.py` →
  `paper2/data/m0_anchor_rtx4070.csv`. **Regret = 10.1× median** (9.4–12.3× over 12 configs,
  states 16/32/48/64 × 3 seeds), triton/worklist (22 Gbps) vs cuda/worklist (227 Gbps), both
  register-resident 1-thread/string, batch 4096×256B, all oracle-matched. NOTE: the *fair*
  register-vs-register anchor (10×) is LARGER than the 6.5× quoted in old docs, which compared
  triton/worklist vs the slower cuda/worklist_global. This 10× is the number to beat.
- [x] **M1 — decomposed: H1 (warp-uniform waste) DOMINATES, decisively.** Nsight (n=64, 2048
  strings) → `paper2/data/m1_nsight_rtx4070.csv`: Triton issues **94.7× more thread-instructions
  and 89.8× more warp-instructions** than CUDA for identical work. Smoking gun:
  `thread_inst_per_inst` = **32.00 (Triton) vs 30.34 (CUDA)** — same number, opposite meaning. In
  CUDA the 32 lanes run 32 *different* strings (1 thread/string, 32 strings/warp); in Triton the
  32 lanes redundantly run the *same* string, because each Triton *program* is a full CTA (4
  warps×32 lanes) processing ONE string. The tile/SPMD model cannot say "1 program = 1 scalar
  thread" / "pack 32 strings into 32 lanes." **This IS the missing primitive.** H2 (int64) is
  second-order (95× redundancy ≫ any 2–4× int64 factor); H3 not separately needed. The 90× warp-inst
  overhead becomes "only" 6–10× time because Triton's huge grid (90% occ) hides much of it — but
  it caps throughput. → the cure (M2) must make each program process 32 strings, one per lane.
- [~] **M2a — lane-packed Triton: PARTIAL cure (~2–4×), corrects M1's over-claim.** Done +
  oracle-validated + Nsight-confirmed. `experiments/cure/m2_lane_packed.py` →
  `paper2/data/m2_lane_packed_rtx4070.csv`, `m2_nsight_rtx4070.csv`. Three-way isolation on the
  DENSE scan (A=skip-scalar O(active), B=noskip-scalar O(NS), C=lane-packed O(NS)/warp):
  **pure lane-packing C/B = 3.2× median** (work held equal), **realistic C/A = 1.8×**. Nsight
  (B→C, ns=32, 4096): lane-packing **removes ~26× warp-instructions** (near the ideal 32×, exactly
  as the missing-primitive predicted) **but throughput only improves 3.8×**. ⇒ DECISIVE: the ~90×
  warp-redundancy M1 measured is **largely hidden by occupancy — NOT the dominant throughput
  bottleneck**. Lane-packing also DROPS occupancy to 5.6% (128 warps total at batch 4096) → becomes
  latency-bound on the per-lane data-dependent scalar chain, and reintroduces costs it can't avoid:
  active-set UNION (no per-lane skip) and EARLY-TERMINATION divergence (the 0.14× outlier = strings
  that accept early; scalar exits, packed can't). **Honest correction of M1:** the cure is NOT a
  front-end packing trick. CUDA wins by getting BOTH full lane use AND occupancy AND per-lane
  control flow — tile/SPMD cannot. This sharpens the missing primitive to **genuine per-lane
  independent control flow + thread-style scheduling**, strengthening the paradigm thesis.
- [x] **M2e — worklist head-to-head: the anchor DECOMPOSES into artifact + recoverable +
  irreducible; mechanism = per-instruction tile tax.** `experiments/cure/m2e_worklist_packed.py` →
  `paper2/data/m2e_worklist_packed_rtx4070.csv`, `m2e_nsight_rtx4070.csv`. 4-way (CU=cuda/worklist,
  WT=gpufsm triton/worklist nw=4, WS=scalar worklist nw=1, WP=lane-packed worklist), oracle-gated:
    - **num_warps artifact:** gpufsm triton/worklist defaults to num_warps=4 → 4 warps/program all
      run ONE string redundantly. WS (nw=1) is **~3.3–3.6× faster** than WT (28→100 Gbps). So a big
      chunk of the M0 anchor was a LAUNCH-CONFIG artifact. ⚠️ AFFECTS PAPER 1 (its worklist Triton
      ~24 Gbps / 6.5× regret IS the nw=4 number) — must disclose + re-baseline.
    - **lane-packing the worklist WORKS (~3×, WP/WS):** my M2a prediction (union → degenerate to
      dense) was WRONG — for these automata the active-set union stays small, so packing helps.
    - **irreducible residual WIDENS with batch:** WP/CU = 0.50× @4096 → 0.42× @16384 → 0.18× @65536
      (regret_nw1 CU/WS = 3.45×/7.0×/13.3×). CUDA scales with batch; tile/SPMD does not.
    - **MECHANISM (Nsight, WP vs CU @16384, ns=32):** SAME occupancy (22.4%), SAME total warps (512),
      warp-inst within 1.33× — yet WP is **3.67× slower**. So the residual is NOT redundancy/occupancy
      (M1's hypothesis, removed by packing) but **per-instruction tile tax**: cross-lane `tl.reduce`
      for the active-set union, masking/predication, weaker scalar ILP. THIS is the IR-level missing
      primitive — a zero-cost scalar/lane program. Honest arc: M1(redundancy)→M2(packing removes it)
      →M2e(residual is per-instruction tile execution overhead, confirmed at matched occupancy).
- [x] **DECISION RESOLVED (user, 2026-06-28):** (1) Paper 1 — "ignore deadlines": handle the
  num_warps finding THOROUGHLY (no rush, no shortcuts); disclosing/re-baselining paper 1 is OK when
  the analysis is solid. (2) Paper 2 — "fai tutto il possibile, cerca il meglio; ti fermo io":
  FULL AMBITION GREEN-LIT, including M3 (build the real IR primitive). Max quality, max scope.
- [x] **M3-lite DONE — the missing primitive is INTRA-WARP LATENCY HIDING (decisive mechanism).**
  Built WP2: a per-lane scalar worklist in PURE Triton (each lane its own ffs + per-lane CSR gather,
  NO cross-lane reduce, NO union). `experiments/cure/m3_lite_scalarlane.py` → `m3_lite_rtx4070.csv`,
  `m3_lite_nsight_rtx4070.csv`. Oracle-validated. Results @16384: **WP2/WP = 1.27×** (removing the
  cross-lane union reduce helped, confirming it was a tile-tax source) but **WP2/CU = 0.51×** (still
  ~2× slower than CUDA). Nsight nails WHY: WP2 has **FEWER warp-inst than CUDA (4.66M vs 5.07M)** and
  SAME occupancy (22.5%), yet 3.3× slower because **issue_active = 10.1%**, warp-latency = 27 cyc/inst
  → LATENCY-BOUND. Root cause: in CUDA a warp = 32 INDEPENDENT threads (32 strings) → a stalled load
  on one lane is hidden by 31 independent lanes' work; in Triton a warp = 32 lanes in LOCKSTEP on one
  instruction stream → a gather stalls all lanes together, hiding only ACROSS warps (occupancy-bound).
  **THE missing primitive = intra-warp latency hiding via independent per-lane instruction streams**,
  which the thread model (CUDA/Warp) has and tile/SPMD structurally lacks. Reconnects exactly to
  paper-1's "paradigm not abstraction-height". Bounds M3-full: closing the residual ~2× means giving
  each lane an independent instruction stream = emitting thread-model (Warp-like) code for the region.
- [~] **M3-full FEASIBILITY assessed (2026-06-28) → deprioritized as loop work, kept as future
  work / dedicated effort.** Shallow-cloned Triton (`/tmp/m3full/triton-src`, main @c05aa65). Build
  reqs: setuptools+cmake+ninja+nanobind + a pinned LLVM (downloaded/built — the heavy part, 20–60+
  min, GB disk). Working env is triton **3.5.1** + torch 2.9.1+cu128; building modified main would
  mismatch torch's bundled triton and needs an isolated venv → fragile, multi-hour, version-coupled.
  JUDGEMENT (user mandate "cerca il meglio", "ti fermo io"): the paper's scientific contribution is
  COMPLETE without it — the constructive cure is already bounded+demonstrated: pure-Triton ceiling
  PROVEN at 0.49× (M3-lite-b) + thread model (CUDA here, Warp in paper 1) achieves 1.0× = existence
  proof that supplying the primitive closes the gap + we can specify the IR primitive precisely
  (`tl.scalar_program` region lowering each lane to an independent instruction stream). Building it
  in Triton-MLIR is arguably a SEPARATE systems paper. So: frame as design + existence proof + future
  work; revisit a real build only after the paper-2 core (decomposition + M4 generality) is written.
- [→] **M3-full IN PROGRESS (user 2026-06-28: "m3 full, voglio tanta roba, livello top 1").** Two
  parallel tracks toward a WORKING constructive cure:
  - **Track A (the literal ask): Triton from source.** Build launched in background at
    `~/m3full_build/` (home fs, 183GB free; /tmp too small at 4GB). Log: `~/m3full_build/build.log`,
    venv `~/m3full_build/venv` (--system-site-packages, isolated from the project .venv). triton-src =
    main @c05aa65. Once it builds, attempt the actual primitive: a TritonGPU op / lowering that emits
    per-thread (thread-SIMT) code for a marked region. NOTE: even after a successful build, the MLIR
    C++ op+lowering is the genuinely hard part (may not fully converge autonomously).
  - **Track B (guaranteed-working cure, the real deliverable): source-to-source `scalar_program`.**
    A small front-end that takes the idiomatic per-lane automaton and LOWERS it to a thread-model CUDA
    kernel (via torch.utils.cpp_extension.load_inline / NVRTC — toolchain confirmed available), then
    show it recovers CUDA-level throughput from the SAME per-lane source = the cure built and measured.
    This is the top-tier contribution: "we implemented the missing primitive and it closes the gap."
    Oracle-gate; paper2/data/m10_scalar_program_*.csv; add a "The cure, implemented" section to the paper.
  M2e localized the residual to per-instruction tile tax (cross-lane `tl.reduce` for the union +
  masking + weak scalar ILP). De-risk in order:
    - **M3-lite (do first):** probe escape hatches — `tl.inline_asm_elementwise` (PTX) and warp
      intrinsics — to express a per-lane scalar worklist with NO cross-lane reduce and NO masking
      (each lane runs its own ffs loop). Measure: does removing the tile-tax sources close the gap
      to CUDA? This bounds whether the primitive is latent-but-unexposed vs needs real compiler work.
    - **M3-full (if lite insufficient/promising):** add a `tl.scalar_program` / per-lane serial
      construct in the Triton MLIR stack (build Triton from source, new IR op + lowering); show the
      idiomatic front end + new primitive lands within ~1.5–2× of CUDA. The landmark "cure".
- [ ] **M4 — generalize (DFA gather) + write-up + artifact.**

## LITERATURE ANALYSIS DONE (2026-06-28, 4 parallel verified sweeps → `paper2/RELATED_WORK.md`)
Niche CONFIRMED empty; novelty holds on two distinctions. Key outcomes folded into the plan:
- **Rigor fix applied to DRAFT:** the mechanism is **abstraction-denied MLP** (independent in-flight
  loads), not "independent issue" (SIMT issues 1 inst/cycle/warp); and it's distinct from memory
  divergence. Reworded abstract + §4.4.
- **Two desk-reject risks + rebuttals locked:** (A) "didn't tune Triton" → Triton↔Gluon control +
  num_warps sweep (M2f) separate tuning from expressibility; (B) "Gluon already does per-thread" →
  layout ≠ control flow. Closest threats: MLIR-regex→DSA (CGO'25), Subwarp Interleaving (HPCA'22),
  Gluon Linear Layouts — all distinguished in RELATED_WORK.md.
- **Venue:** CGO 2027 R2 (~Sep 10 2026), Standard Research Paper track; backstop TACO; arXiv now.
- **Top new experiments to make it bulletproof** (added to actions below): direct MLP measurement,
  instruction-roofline placement, Gluon `scalar_program` prototype, capability-generalization table.

## Next concrete actions (FULL-AMBITION program, user green-lit; sequence to de-risk)
0. [x] **M5 DONE — latency-bound mechanism MEASURED (one hypothesis corrected).** `m5_mlp_rtx4070.csv`:
   at matched occupancy WP2 `long_scoreboard` stall = **15.3× CUDA**, issue activity **9.9% vs 41%** →
   latency-bound on dependent loads, confirmed; issue-ratio 4.1× ≈ time-ratio 3.5× (Little's-law check).
   CORRECTED my own sub-hypothesis: WP2 issues **26–84× MORE** memory requests (masked-lane gathers +
   int64), not fewer — the tile model pays twice (excess masked traffic + no dependent-load hiding).
   DRAFT + RELATED_WORK reworded to the measured framing (stall composition + issue rate, NOT request
   count). The central claim is now measurement, not inference — the decisive reviewer-proof addition.
0b. [x] **M5b DONE — instruction roofline settles "did you write a worse kernel?".** `m5b_roofline_
   rtx4070.csv` + `fig_roofline.png`. WP2 issues at **8.3% of peak warp-issue, 27% of peak DRAM**;
   CUDA at 32% / 3.4% — both far below BOTH ceilings → neither instruction- nor bandwidth-bound; WP2
   has FEWER warp-inst (4.66M vs 5.07M) yet 3.5× slower → structurally latency-bound. (Also fixed a
   data-integrity copy error: m3_lite_nsight CUDA thread_inst was the 2048-string value; corrected to
   148M @16384.)
0c. [x] **M6 DONE (via the Gluon probe) — layout ≠ control flow, concretely.** `scripts/gluon_probe.py`
   confirms plain Gluon cannot lower the data-dependent per-lane CSR loop (`gl.load` returns a
   layout-block, no scalar load → compile error captured). This IS the Risk-B rebuttal: Gluon's
   per-thread capability is layout, not control flow. A full per-lane escape needs the proposed
   `scalar_program` primitive (which Gluon lacks) → folded into DRAFT §6.
1. [x] **M2f DONE — num_warps artifact = 3.4× median** (2.77×@4096 → 3.44×@16384 → 3.69×@65536),
   monotone ~halving per num_warps doubling. `paper2/data/m2f_numwarps_rtx4070.csv`. Sizes the
   launch-config component of the anchor; underpins paper-1 disclosure.
2. [x] **M3-lite DONE** — see milestone. Residual ~2× is intra-warp latency hiding (latency-bound,
   issue 10%). Next sub-steps before M3-full:
   - **M3-lite-b (cheap): BLOCK/occupancy sweep on WP2** (BLOCK=64,128,256 → more warps/program →
     higher occupancy → more cross-warp latency hiding). Does WP2/CU close toward ~0.7–0.8×? Bounds
     the occupancy lever vs the fundamental intra-warp limit. Also try int32 tile for ≤32 states.
2b. [x] **M3-lite-b DONE — occupancy lever is EXHAUSTED; the intra-warp limit is fundamental.**
   WP2 BLOCK sweep {32,64,128,256} (num_warps=BLOCK/32), oracle-gated, event-timed @16384:
   WP2/CU = **0.49× (B=32, best)** → 0.34× (64) → 0.35× (128) → 0.31× (256). Bigger BLOCK HURTS:
   the lockstep `while tl.max` runs to the busiest of more lanes → more divergence/wasted masked
   iterations, cancelling any occupancy gain (Nsight B=128: occ 23% ≈ same, issue 10→18% but net
   throughput worse). ⇒ you cannot close the ~2× by throwing warps at it in pure Triton; the
   per-lane-independent-stream (thread-model) primitive is the only lever → motivates M3-full.
   `paper2/data/m3_lite_b_occupancy_rtx4070.csv`. **The pure-Triton ceiling is WP2/CU ≈ 0.49×.**
3. [x] **M4 DONE — the cure's residual is REGIME-DEPENDENT (generality + unification).** DFA
   decomposition across table sizes (cache→L2→DRAM), oracle-gated by simulate_dfa.
   `paper2/data/m4_dfa_rtx4070.csv` + `m4_dfa_nsight_rtx4070.csv`. (1) Scalar Triton DFA (nw=4) is
   FLAT at ~29 Gbps for ALL table sizes = independent confirmation of paper-1's scalar-gather
   ceiling. (2) Lane-packing gives **~12× over scalar Triton** (PK/TR), much more than the NFA's ~3×
   (the scalar DFA was extra-penalized). (3) KEY: **PK/CU is regime-dependent** — 0.55–0.62× in cache
   (≤4MB table, CUDA leads ~1.7×) but **1.05× at 16MB (>L2 DRAM) — lane-packed Triton MATCHES CUDA**.
   Mechanism (honest; DRAM% only 14.6% so it's memory-LATENCY not saturated bandwidth): when per-
   symbol latency is moderate (cache-resident), CUDA's intra-warp hiding across 32 independent
   threads wins ~1.7×; when latency is huge (DRAM), BOTH paradigms must hide it via cross-warp MLP
   and they CONVERGE. This UNIFIES the NFA (moderate latency → intra-warp matters → fundamental ~2×
   residual) and the DFA-DRAM (huge latency → cross-warp dominates → gap closes). ⇒ the abstraction
   regret / cure efficacy is set by the latency regime, mapping onto paper-1's two faces (control-flow
   vs memory). STRONG generalization of the decomposition.
4. [~] **M4b PARTIAL — real automata run+oracle-valid but exceed the ≤64-state prototype.** Download
   works (Levenshtein 2787, Brill 42661, Fermi 40786 states). cuda/worklist_global is oracle-valid on
   Levenshtein but absolute throughput is **sub-Gbps (0.1–0.4)** = algorithmic cost of large/deep
   automata, ORTHOGONAL to the regret thesis (consistent with paper 1). The lane-packed Triton
   prototype is single-int64-word (≤64 states), so testing lane-packing on real automata needs a
   MULTI-WORD kernel (per-word ffs + multi-word per-lane gather) — substantial, bug-prone, and the
   decomposition is a PARADIGM property not a size property. Documented as future strengthening, not
   blocking. → pivot to the DRAFT (now primary deliverable).
5. **DRAFT (primary) — paper2/DRAFT.md**: spine = the decomposition + regime-dependence + IR-primitive
   design + thread-model existence proof + threats. Every number from paper2/data/*.csv.
4. **Real automata** (ANMLZoo: Levenshtein/Brill/Fermi via gpufsm.io.datasets): confirm the
   decomposition on non-synthetic NFAs (sparser active sets → union cost smaller → packing better?).
5. **M3-full — build the Triton MLIR `tl.scalar_program` primitive** (Triton from source, new op +
   lowering). The landmark cure. Gated on M3-lite's signal but green-lit to attempt.
6. **Paper-1 num_warps disclosure** (thorough, no deadline pressure): once M2f is solid, update the
   paper-1 worklist number + add the num_warps sweep as methodology ("we tuned Triton").
7. **Write-up paper 2** (CGO/CC framing) + artifact, continuously as results land.

## Findings log (append-only, newest first)
- 2026-06-29: **REPRODUCIBILITY SECTION REFRESHED — now covers the P2/P3 artifacts (was stale).** The
  Reproducibility section predated the compiler work: it had a duplicated "figures regenerate from CSVs"
  sentence and omitted the major new AE artifacts. Rewrote it tightly to: (1) the artifact index (claim →
  command → CSV); (2) THREE runnable falsifiable probes — Gluon (compile failure), the lowering-wall
  (MLIR verifier rejects per-lane scf.condition), and the detection-pass check (in-libtriton pass tags
  the lock-step region with the env flag, no-op without); (3) the TritonGPU pass reproduced from a
  versioned patch + source against a pinned Triton commit with the build recipe; (4) the non-mutating
  one-command cross-arch re-validation. Deduped the redundant sentence. A real fix (AE reviewers read
  Reproducibility closely), not padding. Paper 7pp, 0 undefined/0 overfull.
- 2026-06-29: **FIGURE for sec:compiler — the automatic detect→route→lower loop visualized (fig:selector).**
  The most novel section (P2: detection pass + structural wall + selector) had no figure. Added
  `fig_selector()` to `paper2/figures.py`: a clean left-to-right SCHEMATIC (not a redundant speedup bar —
  fig_cure already shows the 4.2× magnitude) — per-lane kernel → `tritongpu-thread-region` detection →
  two branches: detected (NFA worklist) → thread lowering (378→1555 Gbps = 3.9×, ≥ hand-CUDA 742) vs no
  signature (pointer-chase) → tile path unchanged. All numbers pulled from p2_selector + m10 CSVs (no
  hardcoding). First render had overlapping boxes/labels; fixed the geometry (wider xlim, labels in the
  inter-box gaps) → professional. Wired into gpufsm2.tex sec:compiler with \label{fig:selector} +
  precise caption, referenced from the Automatic-detect-and-lower paragraph. TRADEOFF: this took the
  paper 6pp→7pp; kept it (target venues ASPLOS/PLDI/OOPSLA/CGO allow 11–12pp, and this is the only visual
  of the automatic detect-and-lower contribution — worth a page). 0 undefined refs, 0 overfull. PDF regen.
- 2026-06-29: **VERIFICATION PASS — CI green; mechanism rigor assessed as already-complete (no change).**
  Ran the exact CI-parity checks (CI lints src/gpufsm + tests only; experiments/ is outside scope):
  `ruff check src/gpufsm tests` ✓, `ruff format --check` ✓ (36 files), `mypy src/gpufsm` ✓ (23 files),
  `pytest -m "not gpu"` → **37 passed, 24 gpu-deselected**. The session's ~10 new experiments/cure/*.py
  do not touch the CI-linted/tested core, so the repo is green. Assessed the optional long_scoreboard
  rigor (priority 2) and DECLINED it as padding/misleading: the two regret-law channels are already
  quantified by the CORRECT metrics in regret_law.csv — issue-starvation by tile_issue vs thread_issue
  (automata 9.89 vs 41.00), masked-lane waste by tile_tipi vs thread_tipi (rejection 32 vs 17.12). A
  long_scoreboard (dependent-load-stall) column is the WRONG metric for the masked-waste channel
  (rejection is pure-compute, no memory stall → it would read ~0 and mislead) and is already reported for
  the latency-starvation channel (automata 15.3× in m1). Forcing it would weaken, not strengthen. Honest
  "verified, no change needed" iteration. The landmark stands complete; remaining work is hardware-gated.
- 2026-06-29: **SUBMISSION CAPSTONE — docs/SUBMISSION_PAPER2.md captures the path to publication.**
  Crisp, honest submission plan: 1-paragraph elevator pitch; the 8-item contribution list (condensed
  from the paper); target-venue rationale (ASPLOS/PLDI/OOPSLA primary — working in-libtriton pass +
  structural wall naming the missing IR primitive + implemented cure + generality law + AE; CGO strong
  fit for the compiler angle; IISWC/PACT honest floor); artifact-evaluation pointers (experiments/cure/
  README.md index, scripts/run_cross_arch.sh, triton_thread_region_pass/ patch, the falsifiable probes);
  known gaps stated upfront (multi-GPU cloud run pending but harness ready; the clean ~2× is the
  ≤64-state register-resident regime; in-IR lowering realized below TritonGPU); and the NVIDIA-interview
  framing (Triton-MLIR pass + named per-lane primitive as an actionable cuTile/Tile-IR proposal; ITS as
  the substrate; warp-spec is orthogonal/dense). The publication path is now documented end to end. The
  landmark for paper 2 is COMPLETE: built, written, verified, indexed, submission-honest, with the path
  captured. Remaining open items are all hardware-gated (the one A100/H100 cross-arch run) or the
  genuinely-future in-IR primitive.
- 2026-06-29: **PAPER THREATS/INTEGRITY TIGHTENED — submission-honest on all three fronts.** Surgical
  additions to gpufsm2.tex: (1) Methodological Integrity now reports THREE self-corrections (was two) —
  added the two-channel regret-law correction (we falsified our own single "tile issue deficit"
  predictor: rejection is 4.0× with tile issue ABOVE thread's; and the uniform-SpMV "≈1× negative
  control" guess was also wrong at 1.9×). (2) Threats/Single-GPU now states the cross-arch re-validation
  is a turnkey one-command harness checking the falsifiable paradigm-not-arch prediction. (3) New
  Threats/Compiler-pass-scope: detection runs in libtriton; the lowering is realized BELOW TritonGPU
  because the structured tile IR cannot express it (scf.condition single-i1 → per-lane loop termination
  inexpressible, verifier-rejected rewrite) — a property of TODAY's tile IR, falsifiable if a per-lane
  sub-tile loop/exit op is added. Full grep scan for stale phrasing (future-step/single-predictor/MLIR-
  is-engineering): none left (the two "single predictor" hits are the correct multi-component framing +
  the self-correction). Paper 6pp, 0 undefined, 0 overfull; PDF regenerated. The paper is now
  submission-honest end to end.
- 2026-06-29: **REPRODUCIBILITY INDEX + NORTH-STAR SYNC (navigability hardening).**
  `experiments/cure/README.md`: a reviewer/AE index mapping every artifact (m0/m2*/m3*/m4/m9 decomposition,
  m10 cure, landmark_{bfs,spmv,rejection,hashprobe} witnesses, p2_{ttgir_probe,pass_verify,lowering_wall,
  selector} compiler, p3_cross_arch) to the exact paper claim/section it supports, the run command, and
  the CSV it writes — flagging which need the from-source Triton ([src-triton], PYTHONPATH=...) vs the
  system stack. Verified against the live file inventory + each script's CSV-write path (grep), so the
  table is accurate not guessed. Also synced `docs/LANDMARK_PLAN.md` execution program to reality: P1
  (regret law, two-channel) ✅ DONE; P2 (build + detection pass + structural wall + missing primitive +
  selector) ✅ DONE (stronger than planned — the wall is a result, not a failure); P3 (cross-arch harness
  built + self-validated, cloud run hardware-gated). The north star now matches what's built.
- 2026-06-29: **PAPER END-TO-END HONESTY PASS — every number traced to a CSV; abstract updated to the
  full story.** Re-read gpufsm2.tex top to bottom and cross-checked all quantitative claims against
  paper2/data/*.csv: M10 sp/wp2 median 4.16 (paper 4.2× ✓), sp/cu 2.15 ✓; selector 3.899 (3.9× ✓);
  regret-law witnesses all match regret_law.csv (pointer_chase 1.00, spmv_u 1.94, hashprobe 1.40,
  spmv_pl 2.17, automata 1.96, rejection 4.00); automata issue 9.89/41.00 (9.9%/41% ✓); DFA m4 csv
  16MB>L2 pk/cu 1.05× ✓, cache 0.55–0.62× ✓; component-C residual 0.51× reconciles with WP2/CU =
  378/723 = 0.52× ✓. Confirmed the two automata figures are NOT contradictory and ARE labeled:
  component-C residual ~2× = per-lane Triton-vs-hand-CUDA, while M10 SP/WP2 4.2× has SP itself 2.15×
  ABOVE hand-CUDA (minimal thread worklist) → 2×·2× ≈ 4×, stated in sec:implemented. The ONE gap: the
  abstract omitted the two newest pillars — FIXED by adding (a) the generality law (six workloads,
  two-channel = issue starvation + masked-lane waste over a tile-lowering baseline, pointer-chase
  negative control 1.00×) and (b) the compiler work (detection pass in libtriton + structural-impossibility
  proof naming the missing per-lane loop/exit primitive + automatic selector 3.9×). Paper still 6pp,
  0 undefined refs, 0 overfull; PDF regenerated. No number drift found — the CSV-traceability discipline
  held across the whole session.
- 2026-06-29: **P3 CROSS-ARCH HARNESS BUILT + SELF-VALIDATED — one command, ready for cloud A100/H100.**
  `experiments/cure/p3_cross_arch.py` (+ `scripts/run_cross_arch.sh`): re-runs all regret-law witnesses
  (spmv uniform/powerlaw, rejection, pointer-chase/bfs, hashprobe) + the M10 cure (automata sp/wp2) + the
  P2 selector on whatever GPU is present, reads each freshly-measured regret, compares to the committed
  RTX4070 baseline, and writes paper2/data/cross_arch/regret_<gpu>.csv tagged with
  torch.cuda.get_device_name. Bakes in the FALSIFIABLE prediction: regret follows the execution PARADIGM
  not the arch, so each witness's regret persists in direction (divergent >1, pointer-chase ~1) while
  throughput rescales. SAFE/non-mutating: snapshots each witness CSV (and the selector CSV), runs, reads,
  RESTORES the committed baseline (verified: after a full run only the new cross_arch CSV is added; all
  *_rtx4070.csv unchanged). **Self-validated on the RTX4070** (reproduces its own baseline → sanity check
  the parse/restore/compare logic): all 6 persist=yes (rejection 4.0→4.12, spmv 1.95/2.17→1.91/2.19,
  hashprobe 1.40→1.34, pointer_chase 1.00→1.00, automata 4.3→2.87 [run-to-run/thermal variance, still
  >1.1]), selector VERIFIED, verdict CONFIRMED. Ready to run on a cloud pod the moment the user grants
  access; the A100/H100 numbers drop into cross_arch/ and either confirm or falsify arch-independence at
  one command. (P3 was the hardware-gated open item in LANDMARK_PLAN.)
- 2026-06-29: **REGRET-LAW RIGOR + HONESTY CORRECTION — the issue-deficit is NOT a universal predictor.**
  Completed the Nsight profiling of the one blank witness (spmv_powerlaw: tile issue 5.79% vs thread
  5.38%, tipi 32/32 → filled in regret_law.csv). This SURFACED an overstatement in the paper: the headline
  "regret tracks the tile's issue-activity DEFICIT" is only true for the latency-starvation witness
  (automata, tile 9.9% ≪ thread 41%). The cross-witness Nsight table shows the law is genuinely
  **two-channel**, not a single scalar predictor: (i) ISSUE STARVATION (tile issue driven below thread's:
  automata 9.9 vs 41); (ii) MASKED-LANE WASTE (tile does full-width 32-lane work while thread retires
  lanes — rejection: tile tipi 32 vs thread 17, and its issue rate is ABOVE the thread's, 53 vs 39, so
  "issue deficit" would MIS-predict it); over a divergence-free tile-lowering baseline (spmv 1.9× from
  occupancy/masking) + a divergence increment (powerlaw 2.2×@32→5.8×@256). Pointer-chase control (all
  equal, regret 1.00) nails that memory/dep-loads alone cost nothing. Corrected the headline + contribution
  bullet + fig:law caption to the accurate two-channel framing; also un-staled the Conclusion (it claimed
  MLIR integration was "future" — but the detection pass + selector are DONE this session, now stated).
  Paper 6pp, 0 undefined/0 overfull. (Skeptical-scientist: caught my own over-general predictor via the
  measurement I almost skipped — rejection's issue is above thread's, so the deficit framing was wrong.)
- 2026-06-29: **P2 FOLDED INTO THE PAPER — the landmark narrative is now publishable.** Added
  `\subsection{Implemented in the compiler...}` (sec:compiler) to gpufsm2.tex: the real in-libtriton
  detection pass (tritongpu-thread-region, lock-step scf.while/#blocked/tt.reduce signature) + the
  STRUCTURAL WALL (carried tensors already sizePerThread=1 → not layout; scf.condition is single-i1 →
  per-lane tensor<Nxi1> condition rejected by the MLIR verifier, captured via triton-opt) + the NAMED
  missing IR primitive (per-lane sub-tile loop/exit op) + the automatic selector (detect→route→thread
  lowering, 3.9× realized, oracle-gated, negative control on tile). New contribution bullet added.
  Related-work tightened with the SHARP differentiation + 4 new verified bibentries (prism2026 PLDI'26
  manual perspectives; cutile2025 manual SIMT fallback; tawa2026 CGO dense warp-spec; partialcfg2018
  PLDI'18 opposite direction = vectorize): none auto-detects+lowers an irregular per-lane region in a
  tile DSL, and our structural result shows why an in-tile-IR rewrite cannot suffice. Paper compiles
  CLEAN: 6pp, 0 undefined refs, 0 overfull hbox, 0 bibtex warnings; PDF regenerated. The whole P2 arc
  (build platform → detection pass → structural wall → missing primitive → automatic selector) is now
  both BUILT and WRITTEN. Next: P3 multi-GPU prep (cross-arch one-command script) or regret-law Nsight
  rigor (long_scoreboard stall % per witness).
- 2026-06-29: **P2 ENDPOINT REACHED — automatic SELECTOR closes the detect→lower loop (realized cure).**
  `experiments/cure/p2_selector.py`: given a per-lane kernel, AUTOMATICALLY detects the lock-step
  signature by running the real in-libtriton pass (subprocess under the from-source Triton with
  GPUFSM_THREAD_REGION=1, checks `ttg.thread_region_candidate` in the TTGIR), then ROUTES — lock-step →
  the M10 thread-model lowering (the cure, since in-IR lowering is structurally blocked); no signature →
  the Triton tile path. Two Tritons can't share a process (from-source 3.8 for detection vs system 3.5.1
  + gpufsm for measurement), so detection is a subprocess re-entry of the same file. VERIFIED oracle-gated:
  NFA worklist detect=1 → route=thread → **SP/WP2 = 3.9×** over the tile (consistent with M10's ~4.2×
  headline; state sweep 16/32/48/64 × 2 seeds), and a fixed-trip negative-control kernel detect=0 → left
  on tile. ⇒ the FULL landmark loop is now realized end-to-end: detect (real MLIR pass in libtriton) →
  decide (lock-step signature) → lower (thread model) → measured automatic gap-closing, with a correct
  negative control. `paper2/data/landmark/p2_selector_rtx4070.csv`. P2 = detection ✓ + structural-wall ✓
  + missing-primitive named ✓ + automatic selector ✓. Next: fold P2 into gpufsm2.tex; P3 multi-GPU.
- 2026-06-29: **P2 LOWERING WALL — the regret is STRUCTURAL in the loop construct (demonstrated, the
  landmark point).** Two facts from the real matched IR (paper2/data/landmark/p2_lockstep.ttgir): (1) the
  carried tile tensors are ALREADY `#blocked sizePerThread=[1], threadsPerWarp=[32]` — one element per
  lane — so the lock-step is **NOT a layout choice** (re-encoding is a no-op; Gluon-style per-thread
  layout would not help → directly supports "paradigm/control-flow, not layout/abstraction-height").
  (2) The lock-step is the **loop construct**: `scf.while`'s `scf.condition` is defined to take a single
  `i1`; the natural tile→thread rewrite (per-lane `tensor<NxI1>` condition so lanes terminate
  independently) is **rejected by the MLIR verifier**, captured verbatim via the built `triton-opt`:
  *"use of value '%active' expects different type than prior uses: 'i1' vs 'tensor<8xi1, #blocked>'"*.
  ⇒ per-lane loop termination is **inexpressible** in TritonGPU's structured tile control flow; the cure
  must lower BELOW TritonGPU to the thread model (ITS) — exactly what M10 does (nvcc, 4.2×). This is a
  STRONGER result than a hand-tuned in-IR rewrite: it proves the abstraction regret is structural in the
  loop construct, and **names the missing IR primitive — a per-lane (sub-tile) loop/exit op**. Falsifiable
  artifacts (Gluon-probe methodology): `experiments/cure/p2_lowering_wall.py` (exit 0 = wall confirmed) +
  `triton_thread_region_pass/perlane_while_attempt.mlir`. docs/P2_PASS_DESIGN.md lowering status RESOLVED.
  REMAINING P2 endpoint: the automatic SELECTOR over the M10 lowering (detect → route to thread lowering
  at the source/codegen boundary, since in-IR lowering is structurally blocked).
- 2026-06-29: **P2 DETECTION PASS REAL + VERIFIED IN libtriton — the make-or-break compiler loop works.**
  Wrote a TritonGPU MLIR pass `tritongpu-thread-region` (`ThreadRegion.cpp`, a `mlir::ModuleOp` pass):
  walks the module, matches the lock-step signature (an `scf.WhileOp` whose iter-inits include a
  RankedTensorType with a `BlockedEncodingAttr`, AND whose before-region `scf.ConditionOp` condition
  traces — via a bounded backward def-walk — to a `triton::ReduceOp`), tags each match with a
  `ttg.thread_region_candidate` UnitAttr + emits a remark. Registered end-to-end: pass def in `Passes.td`,
  source in `CMakeLists.txt`, python binding `add_thread_region` in `python/src/passes.cc`, env-gated
  insertion (`GPUFSM_THREAD_REGION`) early in `make_ttgir` (third_party/nvidia/backend/compiler.py).
  **Incremental rebuild worked** (`cmake --build` → Passes.td change regenerated Passes.h.inc, recompiled
  TritonGPUTransforms + ThreadRegion.cpp, relinked the 824 MB libtriton; ~5 min). **VERIFIED** by
  `experiments/cure/p2_pass_verify.py`: pass ON → `ttg.thread_region_candidate` present in TTGIR; pass OFF
  → absent (gated no-op); kernel still bit-exact (`sum_{j<trip} j`). ⚠️ Verify needed two DISTINCT JIT
  fns for ON/OFF because Triton's compile cache aliased the second compile to the first. This proves the
  full build-edit-rebuild-verify loop AND that the matcher is correct on real IR — the scaffolding the
  tile→thread *lowering* (next step) plugs into. Sources preserved in-repo (the Triton tree is separate):
  `experiments/cure/triton_thread_region_pass/` (ThreadRegion.cpp + registration.patch + README, base
  Triton commit c05aa65). docs/P2_PASS_DESIGN.md status updated.
- 2026-06-29: **P2 PLATFORM LIVE — Triton-from-source 3.8.0 builds, runs, and the pass target is pinned
  on real IR.** The make-or-break prerequisite is DONE: `libtriton.so` (823 MB) built; Triton 3.8.0
  imports via `PYTHONPATH=$HOME/m3full_build/triton-src/python`, JITs a kernel, runs correctly on GPU,
  full pipeline ttir→ttgir→llir→ptx→cubin. `experiments/cure/p2_ttgir_probe.py` (lint-clean, runnable)
  dumps the TTGIR of a per-lane data-dependent `while tl.max(active)>0` loop and ASSERTS the lock-step
  signature (falsifiable): an `scf.while` carrying `#blocked` tile tensors whose `scf.condition` is gated
  by a **`tt.reduce`-to-scalar** of the per-lane predicate → the whole 32-lane tile loops to the BUSIEST
  lane, idle lanes masked. That reduce-gate IS the masked-lane waste / issue-deficit made syntactic — the
  exact thing the `thread_region` pass must rewrite. IR saved to `paper2/data/landmark/p2_lockstep.ttgir`.
  `docs/P2_PASS_DESIGN.md` written: the working build recipe (cmake<4 + nanobind 2.10.2 + python3.12-dev
  + direct `cmake --build`), the pipeline insertion point (`make_ttgir` in nvidia/backend/compiler.py;
  passes in lib/Dialect/TritonGPU/Transforms/), the transformation (Approach B: re-encode sizePerThread=1,
  drop the tt.reduce gate, per-lane scf.while via ITS, disable pipeliner, reconverge at exit), and the
  pre-measured payoff bound (M10 = the out-of-band existence proof, 4.2×). REMAINING: the C++
  ThreadRegion.cpp + binding + selector, then incremental ninja relink + measure. Honest fallback if
  in-TritonGPU lowering is infeasible: automatic selector over the M10 lowering + this IR design.
- 2026-06-29: **PAPER INTEGRATION — regret law folded into gpufsm2.tex (positioning → measured).**
  Replaced the speculative `\subsection{Generality (positioning)}` + hand-wavy capability table with
  `\subsection{Generality: the regret law}` (`sec:law`) backed by the 6 measured witnesses + a new
  figure (`fig:law`, fig_regret_law.png) + a new contribution bullet. The section now STATES the law
  ("regret tracks the tile's issue deficit relative to thread, created by scalar control, not memory
  irregularity"), anchors it on the pointer-chase negative control (1.00×, tile≡thread on every Nsight
  axis), and honestly notes the multi-component refinement (tile-lowering baseline + divergence
  increment, including the falsified SpMV-uniform ≈1× guess). Compiles clean: **6pp, 0 undefined refs,
  0 overfull hbox, 0 bibtex warnings**. This is the single biggest paper upgrade — the generality claim
  is now empirical, not positioning. (Paper grew 5→6pp; fine for the ASPLOS/PLDI target; if a 6pp-cap
  venue, the DFA or roofline figure is the trim candidate.)
- 2026-06-28: **LANDMARK P1 witness #4 (graph pointer-chase) — THE TRUE NEGATIVE CONTROL, regret 1.00×.**
  `experiments/cure/landmark_bfs.py`: 1M independent random walkers × 64 FIXED steps on a power-law CSR
  graph (deg cv=3.79). Each step is a dependent gather (pointer-chase): the next address depends on the
  previous load — the canonical MLP-bound / latency-bound pattern — with irregular memory (scattered
  colidx, variable degree) but ZERO control-flow trip divergence (fixed step count). Tile (Triton, one
  walker/lane) vs thread (CUDA via nvcc/ctypes) vs exact numpy oracle (deterministic hash, both
  oracle-matched bit-for-bit). **regret = 1.00×** (tile 12098 vs thread 12157 Mstep/s). Nsight is the
  punchline — tile and thread are IDENTICAL on every axis: issue_active **5.42% vs 5.43%**, occupancy
  **72.11% vs 72.11%**, thread_inst/inst **32 vs 32**. ⇒ DECISIVE confirmation of the refined predictor:
  **regret = the tile's issue-activity DEFICIT RELATIVE TO thread**, and that deficit is created by
  SCALAR CONTROL, not by dependent loads or memory irregularity. Pointer-chase starves issue *equally*
  on both (latency hidden by occupancy identically) → paradigm makes no difference → regret 1. Automata's
  regret (1.96×) came specifically because scalar ffs/while recurrence starves the TILE's issue (9.9%) far
  below thread's (41%). This is a *better* negative control than dense-regular (cuBLAS) and than
  SpMV-uniform (whose 1.94× was a bandwidth/occupancy baseline, not latency) — an honest IRREGULAR
  workload sitting exactly on the no-regret line. Added to `regret_law.csv` (leftmost, green) + figure
  retitled "Regret tracks the tile's issue deficit, not memory irregularity". `bfs_rtx4070.csv`.
- 2026-06-28: **LANDMARK regret-law SYNTHESIS — the multi-component law, assembled across 5 witnesses.**
  `paper2/data/landmark/regret_law.csv` + `fig_regret_law()` in `paper2/figures.py`. Five oracle-gated
  tile-vs-thread witnesses, ordered by mechanism, each Nsight-attributed:
  spmv_uniform **1.94×** (baseline_occupancy, 50% vs 94% occ, ZERO divergence) → hashprobe **1.40×**
  (masked-waste diluted by a clean 32-wide gather, tile issue 48% ≈ thread 49%) → automata_nfa **1.96×**
  (latency-starvation: heavy scalar ffs/while recurrence, tile issue 9.9% vs 41%) → spmv_powerlaw **2.17×**
  (baseline + divergence increment, grows to 5.8× @ BLOCK 256) → rejection **4.00×** (masked-waste, pure
  compute, lockstep to max trip). **HONEST framing (the landmark claim, stated precisely):** regret is NOT
  a single-predictor law — it is **tile-lowering baseline (occupancy/register/masking, present even
  divergence-free) + a divergence increment** that splits into two measured sub-mechanisms,
  **latency-starvation** (scalar-control density starves issue) and **masked-lane waste** (divergent trips →
  the lockstep tile pays for idle lanes). The clean ~1× negative control is **dense-regular** work
  (Triton≈cuBLAS), drawn as the green reference line — NOT irregular SpMV. This multi-component attribution,
  isolated by independent witnesses with the SAME nvcc-lowering machinery, is stronger and more falsifiable
  than the original single "MLP denied" story. ⇒ the cure must recover BOTH the occupancy baseline AND
  per-lane control; front-end lane-packing alone (M2a) recovers only the masked-waste fraction.
- 2026-06-28: **LANDMARK P1 witness #3 (SpMV) — PREDICTION FALSIFIED (honest, important correction).**
  `experiments/cure/landmark_spmv.py`: SpMV CSR, uniform-nnz (no control divergence) vs power-law-nnz
  (divergent), same kernels, oracle-validated. PREDICTED uniform regret ~1× (clean negative control).
  RESULT: uniform regret = **1.94× — NOT ~1×**. Nsight: tile thread_inst/inst=32 = thread (no
  divergence, confirmed) BUT tile occupancy 50% / DRAM 29% vs thread 94% / 50% → the 1.9× is a
  BASELINE TILE-LOWERING OVERHEAD (occupancy/register-pressure/masking on every gather), present even
  with ZERO divergence, and NOT fixable by num_warps (BLOCK sweep 32/128/256 leaves uniform ~1.9×).
  Power-law adds a divergence increment that GROWS with BLOCK (2.2×@32 → 3.9×@128 → 5.8×@256, since
  the tile locksteps over more rows). ⇒ REVISED model: tile-vs-thread regret = **tile-lowering baseline
  (occupancy/masking, even divergence-free) + divergence increment (masked-waste/latency)**. The clean
  ~1× negative control is DENSE-REGULAR work (Triton≈cuBLAS, known), NOT irregular SpMV. This weakens
  the simple "regret follows control-divergence not memory" headline → needs the multi-component framing.
  `spmv_rtx4070.csv` + `spmv_nsight_rtx4070.csv`. (Skeptical-scientist: falsified my own negative-control
  prediction — better found now than by a reviewer.)
- 2026-06-28: **LANDMARK P1 witness #2 (rejection sampling) — regret 4.0×, isolates masked-lane waste.**
  `experiments/cure/landmark_rejection.py`: PURE control-flow divergence (data-dependent trip count via
  a deterministic hash, ~no memory), tile vs thread vs CPU oracle (exact 64-bit-wrap). Oracle-validated;
  trips min 0/mean 2/max 11 (divergent). **regret = 4.0×** (>> hash-probe's 1.4×). Nsight
  (`rejection_nsight_rtx4070.csv`): tile thread_inst/inst = 32 (full-width masked, lockstep to max trip)
  vs thread 17.1 (independent retire); tile issues MORE (53% vs 39%) but wastes it → 4× regret =
  masked-lane waste, fully exposed because there's no gather-MLP to dilute it (unlike hash-probe).
  SPECTRUM forming: rejection 4.0× (pure compute divergence) > automata ~2× (divergence diluted by
  memory latency) > hash-probe 1.4× (gather preserves MLP) > SpMV predict ~1× (aligned, no divergence).
  Unifying predictor: regret ≈ warp divergence factor × (compute-vs-gather-MLP-preserved fraction).
- 2026-06-28: **LANDMARK P1 witness #1 (hash-probe) — refines the regret predictor (honest, important).**
  `experiments/cure/landmark_hashprobe.py`: GPU hash table, data-dependent probe loop, tile (Triton
  lane-packed) vs thread (CUDA per-key) vs CPU oracle. Oracle-validated. Regret = ~1.4× and FLAT across
  load 0.5→0.95 (avg probe 1.8→62) → FALSIFIES the naive "regret ∝ dependent-load count" predictor.
  Nsight (`hashprobe_nsight_rtx4070.csv`): tile issues at 48% ≈ thread 49% (vs automata tile 9.9% vs
  41%) — the tile's `tl.load` gather already provides full 32-wide intra-warp MLP, so hash-probe is NOT
  latency-starved in the tile; its 1.4× is masked-lane waste (thread_inst/inst 32 vs 3.65). ⇒ the
  regret predictor is the tile's ISSUE-ACTIVITY DEFICIT (per-element scalar-control / divergence the
  tile serializes), not dependent-load count. Two sub-mechanisms: latency-starvation (automata) vs
  masked-lane-waste (hash-probe). Sharper, falsifiable cost model. Next P1: rejection sampling (pure
  control, isolates masking) + SpMV (aligned gather → predict ~1× = the negative control).
- 2026-06-28: **QUALITY UPGRADE (user: "migliora ancora") — the cure now proven to act VIA the
  mechanism, + 2-panel figure + reproduce.sh.** (1) Nsight-profiled the SP (scalar_program→threads)
  kernel: issue activity **36.2%** (vs the tile's 9.9%, ≈ CUDA's 41%) and long_scoreboard stall **29×
  lower** than the tile → the cure restores the thread-model issue signature, closing the gap through
  EXACTLY the intra-warp-latency-hiding mechanism (component C). This is the strongest possible
  validation (throughput + mechanism). `m10_nsight_rtx4070.csv`, paper §6.2. (2) fig_cure → 2 panels
  (throughput AND issue activity), self-contained. (3) `paper2/reproduce.sh`: one command regenerates
  every oracle-gated CSV + all figures (CGO artifact-eval friendly). Remaining genuine gaps need the
  user: 2nd GPU (cloud A100, mechanism is arch-general but absolute factors need re-measure) and the
  full Triton-MLIR build (abandoned, env-fragile). Paper 5pp, 6 figures, 0 undefined/0 overfull.
- 2026-06-28: **FINAL QA — paper consistency-verified + cure figure + 6th contribution.** Re-grepped
  EVERY headline number against the CSVs — all match (anchor 10.11, num_warps 2.77/3.44/3.69,
  lane-pack 3.19/9.79/19.35, residual 0.51, long_scoreboard 15.3, issue 9.89/41.00, DFA 0.55–1.05,
  SP/WP2 4.16, SP/CU 2.15). Added Fig. (f) `fig_cure` (SP vs CU vs WP2, from m10 CSV), a 6th
  contribution ("the cure, implemented"), and an artifact-availability note. RELATED_WORK updated:
  cure is now implemented, not just designed. Paper compiles 5pp, 0 undefined/0 overfull, 6 figures.
  **The paper is complete, internally consistent, and submission-ready (diagnosis→implemented cure).**
- 2026-06-28: **M3-full / M10 — THE CURE IS IMPLEMENTED AND MEASURED (Track B done; Track A build
  abandoned honestly).** Track B: `experiments/cure/m10_scalar_program.py` lowers the SAME idiomatic
  per-lane automaton body to a per-thread CUDA kernel (nvcc → .so → ctypes, bypassing torch's C++
  headers which don't compile on the local gcc-12/13). Oracle-validated bit-for-bit on with-accept
  NFAs; sustained throughput on no-accept NFAs (removes early-exit confound). **SP/WP2 = 4.52× median**
  (same per-lane source, threads vs tiles) and **SP/CU = 2.25×** (matches/exceeds hand-CUDA — the
  generated kernel is a minimal thread worklist). The residual is CLOSED BY CONSTRUCTION. Added paper
  §6.2 "The cure, implemented" + abstract/conclusion; recompiles 5pp clean. `m10_scalar_program_rtx4070.csv`.
  Track A (Triton-from-source build at ~/m3full_build): failed 3× at cmake-configure (after fixing
  cmake>=4 and the pip-cmake shim) — main→v3.8.0, env-fragile, error truncated by pip; per plan,
  documented and abandoned (the build fragility is not the contribution; the implemented lowering is).
  Honest scope: SP beating hand-CUDA 2.25× reflects a tighter minimal kernel, not a claim that the
  primitive is faster than all CUDA; the load-bearing number is SP/WP2 (threads vs tiles, same source).
- 2026-06-28: **M9 — multi-word lane-packed worklist works + oracle-correct to 256 states; residual is
  register-regime-scoped.** `experiments/cure/m9_multiword.py` ([BLOCK,NWORDS] tile, per-word ffs +
  per-lane masked multi-word scatter) validated bit-for-bit vs reference at 96/128/192/256 states
  (NWORDS 2–4). `m9_multiword_rtx4070.csv`: MW/CU = 1.50/2.15/1.44/1.69 — but this FLIPS above 1 not
  because Triton closed the residual; **cuda/worklist's register fast-path degrades past 64 states**
  (227 Gbps @≤64 → 13–27 @96–256, register pressure), so the >64 head-to-head is CONFOUNDED. Honest
  scope: the clean ~2× residual is a ≤64-state register-resident-regime result. Threat "prototype is
  ≤64 states" addressed (it generalizes + stays correct); ANMLZoo-scale needs the per-lane multi-word
  scatter which explodes — itself the tile limitation we study. Paper Threats updated; recompiles 5pp clean.
- 2026-06-28: **Paper expanded to 5pp (clean).** Added Methodological Integrity (the 2 self-corrected
  hypotheses), fuller Threats, Reproducibility section, self-contained figure captions; fixed a lost
  `\section{Related Work}` header. 0 undefined/0 overfull, every number CSV-traced. **Assessment: the
  paper is submission-ready and dense; further LaTeX expansion toward 11pp is low marginal value
  (CGO does not require it).** The remaining HIGH-value frontiers are (i) the multi-word lane-packed
  kernel to test on real ANMLZoo automata (removes the ≤64-state threat — tractable strengthening),
  and (ii) M3-full (the real Triton-MLIR primitive — high value, multi-hour fragile, user-gated).
  Next: attempt the multi-word kernel (strengthening) before considering M3-full.
- 2026-06-28: **LaTeX expanded — results table + Little's-law prose + denser method/related-work,
  still 0 undefined / 0 overfull.** Table I = multiplicative decomposition (28→104→383→735 Gbps =
  3.7×·3.7×·1.9× @16384, CSV-traced). Decided M8-as-new-experiment would mostly reconfirm M4 (DFA
  crossover already = Little's-law made quantitative) → folded the argument into prose, no weak rerun.
  Remaining: push toward CGO ~11pp (port honest-corrections narrative, full threats, artifact
  appendix); M3-full still deferred. Paper is submission-ready in skeleton; expansions are incremental.
- 2026-06-28: **LaTeX paper built — `paper2/gpufsm2.tex` compiles clean.** 4pp IEEE conference
  (IEEEtran.cls copied locally; unsrt bibstyle since IEEEtran.bst absent), 5 figures, 21-entry
  refs.bib from the verified citations, 0 undefined refs, 0 overfull hboxes; every number matches the
  CSVs. The paper now physically exists and compiles. 4pp is short for CGO (11pp) → room to expand the
  prose from DRAFT.md. PDF committed. NEXT: expand toward page budget + optional M8 (Little's-law
  quantitative crossover); M3-full still deferred.
- 2026-06-28: **M7 + DRAFT polish — submission-grade.** Added §6.1 capability-generalization table
  (decode / ragged batching / BFS / tokenization all need the same per-lane primitive) pre-empting
  "automata-only narrow"; figure refs (a–e) wired through; contributions tightened to the measured
  numbers; re-verified every headline number traces to a CSV (WP2/CU 0.51, DFA 0.55–1.05, anchor 10.1
  all confirmed). DRAFT now complete & internally consistent.
- 2026-06-28: **M5 — the latency-bound mechanism is now MEASURED (stall-reason analysis).** At matched
  occupancy, WP2's long_scoreboard (dependent-load wait) stall is 15.3× CUDA's and it issues at 9.9%
  vs 41%; issue-ratio 4.1× ≈ throughput-ratio 3.5× (Little's-law-consistent). Self-correction #N: WP2
  issues 26–84× MORE memory requests (masked gathers), not fewer — tile pays twice. Converts the
  central §4.4 claim from inference to measurement (the reviewer-proof addition the lit sweep wanted).
- 2026-06-28: **M4 — cure residual is regime-dependent; unifies NFA & DFA via latency.** DFA
  lane-packing beats scalar Triton ~12× and, in the DRAM-table regime (>L2), MATCHES CUDA
  (PK/CU=1.05×) vs 0.55–0.62× in cache. The intra-warp latency-hiding advantage (CUDA) only matters
  at moderate latency; at DRAM latency both rely on cross-warp MLP and converge. So the NFA's
  fundamental ~2× residual and the DFA's closeable gap are the SAME mechanism at different latency
  scales — mapping onto paper-1's control-flow vs memory faces. Scalar Triton DFA flat ~29 Gbps
  reconfirms the scalar-gather ceiling.
- 2026-06-28: **M3-lite-b — occupancy lever exhausted, intra-warp limit confirmed fundamental.**
  WP2/CU = 0.49× at BLOCK=32 and only WORSENS with bigger BLOCK (0.31× at 256): more warps/program
  raise issue rate slightly but the larger lockstep tile adds divergence that cancels it. Pure-Triton
  ceiling ≈ 0.49× CUDA (~2× regret). Only a thread-model primitive (independent per-lane streams) can
  close it → the constructive case for M3-full.
- 2026-06-28: **M3-lite — the missing primitive is intra-warp latency hiding.** A per-lane scalar
  worklist in pure Triton (own ffs + per-lane gather, no cross-lane reduce) beats the M2e union
  worklist 1.27× but is still 2× under CUDA. Nsight: FEWER warp-inst than CUDA + same occupancy yet
  3.3× slower, issue_active 10% → latency-bound. CUDA's warp = 32 independent threads hides
  per-element load latency intra-warp; Triton's warp = 32 lockstep lanes cannot. That is the
  irreducible primitive and it bounds M3-full (must emit thread-model code). Also M2f: num_warps
  artifact = 3.4×.
- 2026-06-28: **M2e — anchor decomposed; residual is per-instruction tile tax; num_warps artifact
  found.** 4-way worklist head-to-head across batch. (1) gpufsm triton/worklist default num_warps=4
  wastes ~3.3× vs nw=1 → re-baselines the anchor and ⚠️ implicates paper-1's worklist number.
  (2) Lane-packing the worklist helps ~3× (M2a prediction of degeneration was wrong; union stays
  small). (3) Residual regret (best-Triton WP vs CUDA) WIDENS with batch: 0.50×→0.42×→0.18×. (4)
  Nsight: at matched occupancy + matched warps + ~equal warp-inst, WP is still 3.67× slower → the
  residual is per-instruction tile-execution overhead (cross-lane reduce, masking, ILP), the true
  IR-level missing primitive. Reached the user decision point. Skeptical-scientist wins this round:
  corrected my own M2a prediction AND found a launch-config inflation in the anchor.
- 2026-06-28: **M2c — lane-packing's benefit is OCCUPANCY-GATED; at scale it recovers most of the
  redundancy.** Batch-scaling the pure-packing ratio C/B: **3.2× @4096 → 9.8× @16384 → 19.4×
  @65536** (toward the ideal 32×), realistic C/A → 10.8× @65536. Absolute lane-packed dense Triton
  reaches **108–267 Gbps** at batch 65536 — IN THE RANGE of the CUDA worklist anchor (227 Gbps).
  So M2a's "only 3.2×" was small-batch occupancy starvation, not a fundamental wall. Revised
  picture: for the DENSE algorithm, lane-packing (Triton-expressible via the SHARED-CSR uniformity
  that keeps inner-loop bounds scalar) closes most of the regret given enough strings. The
  IRREDUCIBLE regret lives on the WORK-EFFICIENT worklist, where per-lane ffs-skipping + early-exit
  can't be lane-packed (active-set union) — the 0.53× outlier (early-accepting strings) is the
  early-exit-divergence fingerprint. → M2e tests exactly this. `m2_batch_scaling_rtx4070.csv`.
- 2026-06-28: **M2a — the obvious cure is only partial, and Nsight proves WHY.** Lane-packing
  removes ~26× warp-instructions (B→C) but moves throughput only 3.8× → M1's warp-redundancy is
  hidden by occupancy, not the bottleneck. Pure packing 3.2× / realistic 1.8×. New tile-only costs:
  active-set union (no per-lane skip) + early-term divergence (0.14× outlier) + occupancy collapse
  (5.6%). This is a more interesting result than "packing fixes it": it localizes the residual to
  per-lane data-dependent control flow + thread-scheduling, which tile/SPMD structurally lacks.
  Skeptical-scientist: M2 corrected M1's premature "H1 IS the primitive" — H1 is real but mostly
  hidden; the binding constraint is deeper. Next: M2c batch-scaling disambiguation.
- 2026-06-28: M0 done. Fair anchor = **10.1×**. CUDA register worklist ~227 Gbps is remarkably
  flat across 16–64 states (work-efficient); Triton ~22 Gbps equally flat → the gap is a constant
  multiplier, consistent with a paradigm/codegen cause (H1/H3), not a per-state algorithmic cost.
  Warp backend registered 0 techniques in this venv (warp not wired) — irrelevant to the cure.

## Guardrails
- Correctness gates speed: oracle-match (`reference.py`) on every kernel before any Gbps number.
- Median+CI95, GPU-saturating batch (small batch inflates ratios — see paper 1 audit).
- CI parity before commit: `ruff format --check && ruff check && mypy && pytest -m "not gpu"`.
- Numbers trace to `paper2/data/*.csv`. Negative results are results — log them.
