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
- [ ] **M3 — constructive MLIR primitive** (gated on M2; M2 shows the Triton-expressible cure caps
  at ~2–4×, so the full cure likely needs the IR-level thread/scalar-program lowering).
- [ ] **M4 — generalize (DFA gather) + write-up + artifact.**

## Next concrete actions (do these in order)
1. **M2e — lane-pack the WORK-EFFICIENT worklist (the crux that reconnects to the M0 anchor).**
   M2c settled the dense case: lane-packing IS occupancy-gated and recovers most of the warp-
   redundancy at scale (see finding). But the M0 anchor (10×) was the WORKLIST (ffs O(active)), not
   the dense scan. The genuine missing primitive should bite HERE: lane-packing the worklist forces
   processing the active-set UNION across 32 lanes (can't ffs-skip per-lane) + no per-lane early
   exit. Build a lane-packed worklist and measure: does it FAIL to beat the scalar worklist (union
   cost) even at large batch? If so, that isolates per-lane data-dependent control flow as the
   irreducible primitive (dense is lane-packable, work-efficient is NOT). Oracle-gate; sweep batch.
   This is the decisive figure: "lane-packing rescues the dense kernel but NOT the work-efficient
   one → the regret that survives is per-lane control flow."
2. **M2d — cheap, high-value: does the REAL triton/worklist waste Nx from default num_warps?**
   The gpufsm `_worklist_kernel` launches with default num_warps=4 → 4 warps/program ALL run one
   string redundantly (M1 saw 128 threads/program). Test triton/worklist at num_warps=1 vs 4 — if
   ~4× free, a big chunk of the 10× anchor is just a launch-config artifact (must be disclosed, and
   re-baselines the anchor). Quick edit/standalone.
3. **M2b (only if a gap remains):** `tl.inline_asm_elementwise` PTX for the int64 bitset ops —
   bounds the residual H2/H3 (codegen/int64) after the control-flow effects are accounted.
4. **M3 / USER DECISION POINT:** once M2c/M2d are in, the picture determines whether the full cure
   needs building the MLIR `tl.scalar_program` lowering (weeks, Triton-from-source) or whether the
   paper stands on "the obvious tile-level cure recovers only ~Nx → the missing primitive is
   IR-level per-lane control flow" (strong CGO/CC story WITHOUT the build). Surface to user then.

## Findings log (append-only, newest first)
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
