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
- [ ] **M1 — decompose the 10× (H1 warp-uniform waste / H2 int64 emul / H3 codegen).** NEXT.
- [ ] **M2 — cure prototype, no compiler rebuild** (lane-packed P-strings/program; inline-PTX).
- [ ] **M3 — constructive MLIR primitive** (gated on M2).
- [ ] **M4 — generalize (DFA gather) + write-up + artifact.**

## Next concrete actions (do these in order)
1. **M1a (no sudo):** test H2 — build a triton/worklist int32 variant (≤32 states) and compare
   its regret vs the int64 path. If regret unchanged → int64 emulation is NOT the cause (points
   to H1). Cheap, decisive.
2. **M1b (sudo Nsight):** profile triton/worklist vs cuda/worklist — warp execution efficiency,
   achieved occupancy, DRAM%, eligible warps, issue stalls. Expect H1 signature: Triton low warp
   efficiency (scalar work executed warp-uniformly, ~1/32 lanes useful), both latency-bound.
   sudo is passwordless per CLAUDE.md: `sudo /usr/local/cuda/bin/ncu`.
3. **M2a:** lane-packed Triton worklist — `[BLOCK]`-shaped state tile, lane j = string j, per-lane
   divergent control flow via uniform max-trip loop + masking. Oracle-gate, then measure gap
   closure. Start ≤32 states / int32.

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
