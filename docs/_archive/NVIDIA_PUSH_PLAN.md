# NVIDIA push — turning the gaps into strengths (north-star plan)

Ultimate goal: **a hire-grade body of work for NVIDIA** (compiler / GPU-perf / Triton-MLIR).
The two papers + the cure are *instrumental*. This doc is the campaign north-star; the autonomous
loop reads it first and `docs/NVIDIA_PUSH_PROGRESS.md` (findings, newest first) to pick the next move.
Branch: `dev` (PR→main at milestones). CI parity before every commit:
`ruff format src tests scripts paper*/figures.py && ruff check src tests && mypy src/gpufsm && pytest -m "not gpu"`.

## The 7 weaknesses → the strength each must become
1. **Single consumer GPU** → multi-arch evidence (A100/H100), paradigm-not-arch confirmed cross-silicon.
2. **Cure is detection+proof+out-of-band nvcc, not a real in-compiler transform** → an actual in-compiler
   lowering of the irregular region (even a scoped sub-case) that runs end-to-end and closes the gap.
3. **Narrow domain (automata)** → the SAME mechanism shown on a workload NVIDIA cares about (MoE routing /
   ragged attention / sparse), oracle-gated, in the regret-law framework.
4. **No upstream contribution** → a Triton GitHub RFC/issue proposing the missing per-lane sub-tile
   loop/exit primitive, that the maintainers engage with.
5. **Not peer-reviewed** → submitted (arXiv → CGO/PLDI/ASPLOS) and ideally landed.
6. **Solo signal** → public, reproducible artifact + RFC discussion = external validation surface.
7. **Interview ownership** → a written derivation/whiteboard pack so every claim (mechanism, the MLIR
   pass, warp-scheduling/ITS/PTX-SASS) is owned cold.

## The 5 fronts (leverage order) — concrete steps + done-criteria
### F1 — Triton RFC (highest signal; self-contained to DRAFT)
- Research Triton's contribution norms + cuTile/Tile-IR + Gluon/TLX current state (web).
- Write a complete, postable RFC: problem, the lock-step IR signature, the structural wall (scf.condition
  is single-i1), the proposed `tt.scalar_region` / per-lane sub-tile loop/exit op, semantics, lowering
  sketch (below TritonGPU via ITS), the 3.9–4.2× evidence, prior-art differentiation.
- DONE when `docs/rfc/triton-per-lane-region.md` is reviewer-ready; user posts it.

### F2 — ML-domain witness (self-contained; START HERE for code)
- Add an oracle-gated tile-vs-thread witness in the regret-law framework on an ML-irregular kernel:
  candidate = **MoE/top-k token routing** (ragged per-expert gather, data-dependent counts) or
  **ragged/jagged attention** (variable seq-len inner loop). Pick the one whose lock-step signature is
  cleanest and whose thread-lowering is tractable (M10-style nvcc).
- DONE when: oracle bit-exact, tile-vs-thread regret measured + Nsight-attributed, row added to the
  regret law (`paper2/data/landmark/regret_law.csv`), folded into the paper's generality section.

### F3 — real in-compiler lowering (the core hard win)
- Beyond detection: implement the actual transform. Likely below TritonGPU (TritonGPU→LLVM/NVVM) since
  the structural wall forbids in-tile-IR. Start with the narrowest sound sub-case that compiles + runs +
  is oracle-correct + measurably closes the gap, even partially.
- DONE when a kernel the selector marks is lowered *by the compiler itself* (not nvcc shell-out) and runs
  correctly faster than the tile path. Honest partial counts; document the exact blocker if it stalls.

### F4 — multi-GPU (hardware-gated on user cloud pod)
- `bash scripts/run_cross_arch.sh` on A100/H100; fold the cross-arch confirmation into Threats/Limitations.
- Note already in CLAUDE.md: an A100 cross-arch for PAPER 1 was done (regret_a100.csv); P2's harness is ready.

### F5 — submission (user owns accounts)
- arXiv prep (endorsement caveat noted in memory), then CGO/PLDI/ASPLOS per SUBMISSION_PAPER2.md. Also the
  ⚠️ paper-1 num_warps disclosure before HPEC (deadline 7 Jul) — flag, do not touch paper 1 without go-ahead.

## Autonomous operating rules
- One committed artifact (or honest "verified, no change") per iteration; tree clean; CI parity; push dev.
- Use web research (WebSearch/WebFetch / deep-research skill) whenever a front needs external grounding;
  synthesize sources, cite them, verify claims.
- Maintain `docs/NVIDIA_PUSH_PROGRESS.md` (read first / update last). When it or CURE_PROGRESS grows too
  large, COMPACT it (summarize resolved items, keep open threads) — and compact project memory likewise.
- Never stop at "done": after a front lands, find the next improvement toward the hire. Be a rigorous
  researcher — adversarial self-check, honest corrections, every number CSV-traced.
- Self-contained now: F2 (code), F1 (draft), F3 (start). Gated: F4 (pod), F5 (accounts).
