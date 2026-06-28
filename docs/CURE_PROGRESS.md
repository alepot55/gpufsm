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
- [ ] **M2 — cure prototype, no compiler rebuild** (lane-packed P-strings/program; inline-PTX).
- [ ] **M3 — constructive MLIR primitive** (gated on M2).
- [ ] **M4 — generalize (DFA gather) + write-up + artifact.**

## Next concrete actions (do these in order)
1. **M2a — lane-packed Triton worklist (THE cure prototype).** Make each Triton program process
   `BLOCK` strings, one per lane: state working set = a `[BLOCK]`-shaped int64 tile (lane j =
   string j's `cur` bitmask). The per-string inner loops (`while bits`, `for k in CSR row`) have
   per-lane-divergent trip counts → cannot use `range(load,load)`. Express via a UNIFORM outer
   structure + per-lane masking:
     - outer symbol loop: `for pos in range(max_len)` (uniform; mask lanes past their length).
     - state-extraction: vectorize `ffs` over the `[BLOCK]` tile (libdevice.ffs is elementwise).
     - CSR inner loop: the hard part — per-lane rows differ. Options to try, in order:
       (i) gather row bounds as tiles, loop `k` to a uniform max bound, mask inactive lanes
           (`tl.where`), accept wasted lanes; measure if still a net win (warp-inst should drop ~32×).
       (ii) if (i) too divergent, restrict M2a to NFAs with bounded out-degree (pad CSR rows to a
            constant D) so the inner loop is a uniform `for k in range(D)` over a `[BLOCK,D]` gather.
   Oracle-gate (bit-for-bit vs reference) BEFORE any Gbps. Start ≤32 states / int64 tile (int64
   ok; H2 is second-order). Target: recover ≥2× of the 10× (→ ≤5×); ideally approach ~32× warp-inst
   reduction. Write `experiments/cure/m2_lane_packed.py` + `paper2/data/m2_lane_packed_rtx4070.csv`.
   Re-profile with Nsight to confirm warp-inst/string drops ~32× (the mechanism, not just the time).
2. **M2b (bound H2/H3):** `tl.inline_asm_elementwise` PTX for the bitset ops — bounds the residual
   after lane-packing. Only if M2a leaves a meaningful gap.
3. **M3 decision:** if M2a closes the gap → "latent primitive + front-end" paper; if masking
   overhead caps it → motivates the MLIR `tl.scalar_program`/per-lane-serial-range lowering (M3).

## Findings log (append-only, newest first)
- 2026-06-28: M0 done. Fair anchor = **10.1×**. CUDA register worklist ~227 Gbps is remarkably
  flat across 16–64 states (work-efficient); Triton ~22 Gbps equally flat → the gap is a constant
  multiplier, consistent with a paradigm/codegen cause (H1/H3), not a per-state algorithmic cost.
  Warp backend registered 0 techniques in this venv (warp not wired) — irrelevant to the cure.

## Guardrails
- Correctness gates speed: oracle-match (`reference.py`) on every kernel before any Gbps number.
- Median+CI95, GPU-saturating batch (small batch inflates ratios — see paper 1 audit).
- CI parity before commit: `ruff format --check && ruff check && mypy && pytest -m "not gpu"`.
- Numbers trace to `paper2/data/*.csv`. Negative results are results — log them.
