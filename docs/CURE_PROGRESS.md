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
- [ ] **M3-full (deferred) — actual Triton-MLIR `tl.scalar_program` op + lowering.** Future work.
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
