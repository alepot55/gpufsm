# Second-GPU TODO (hardware-gated — do NOT attempt on the local RTX 4070)

Items deferred until a second GPU architecture is available (planned, per user). All are
*re-runs* of existing, committed scripts on new hardware — not re-implementations. Everything
regenerates from one command set; only the CSVs/figures change.

## Why a second GPU (the external-validity gap)
Results are from one GPU (RTX 4070, sm_89, 6 MB L2). The *qualitative* claims are
architecture-independent (the 2×2, the capability→cost map, and the Triton↔Gluon control are
properties of the DSL compilers / hold at compile time). Two *quantities* are L2- and
SM-count-dependent and are the camera-ready cross-arch confirmation.

## Falsifiable predictions to test on a ≥40 MB-L2 GPU (e.g. A100 80 GB / H100)
1. **DFA L2-knee shifts right. ✅ CONFIRMED (A100, 2026-06-26).** Ran `scripts/second_gpu_quick.sh`
   on an A100 80GB PCIe (40 MB L2): CUDA throughput stays high to ~16 MB then drops through
   32–48 MB to a DRAM plateau — the knee moved from ~6–8 MB (4070) to ~32–48 MB, i.e. ~6×, tracking
   the 6.7× larger L2. Data: `paper/data/dfa_knee_a100.csv`; integrated into §6.5 + Limitations.
   (The earlier quick run mismatched on an old Triton 3.0 image; the richer `second_gpu_rich.sh`
   re-ran on Triton 3.7 with all three backends oracle-matching — see item 2.)
2. **Regret factors rescale but the 2×2 pattern holds. ✅ CONFIRMED (A100, 2026-06-26).**
   `scripts/second_gpu_rich.sh` on the A100 (current Triton 3.7 stack, Warp 1.14): NFA regret vs
   CUDA = Triton **3.2×** (flat across 32/48/64, 3 seeds), Warp **0.80–0.83×** — rescaled from the
   4070's 6–8× / 0.9× but the *structure* is identical (Triton tile/SPMD pays, Warp thread-SIMT
   ≤ CUDA). DFA: Triton flat **~24–30 Gbps** across 2–128 MB (≈ the 4070's 29–32 → arch-independent
   scalar ceiling), CUDA 73–531 Gbps exploiting the 40 MB L2 (DFA regret 3–18×). Data:
   `paper/data/regret_a100.csv`, `paper/data/dfa_knee_rich_a100.csv`. Integrated into §6.5 + Threats
   (External) + Limitations. This also retires the "re-validate Triton/Warp on a Triton-3.5 stack"
   camera-ready caveat — done on Triton 3.7. The cost-model *constants* rescale (per-backend fits),
   but the *relative* regret (the claim) reproduces.
3. **Causal ablation cliff persists.** `scripts/ablate_scalar_control.py` — the Triton
   tile-vs-scalar cliff (16× on the 4070) should remain large; CUDA/Warp scalar-recurrence should
   remain ceiling-free.
4. **Worklist/warp speedup + occupancy.** Re-run `scripts/bench_worklist_warp.py` and Nsight
   (`scripts/profile_target.py`) — expect the same latency-bound signature (DRAM low, L2-resident)
   on a bigger L2, and re-confirm the warp occupancy fix.

## Stretch (needs more than a re-run; only if pursuing top-tier absolute numbers)
- ngAP/ANG-class absolute throughput: a block-cooperative active-set with shared-memory frontier
  privatization or memoization/non-blocking multi-symbol. This is the algorithmic gap we
  explicitly do not close in the current contribution (see `docs/KERNEL_EXPERIMENTS.md`).

## Protocol when the 2nd GPU is available
- `gpufsm env` to capture the new GPU/driver/toolkit versions.
- Re-run the scripts above; commit the new CSVs alongside the 4070 ones (do not overwrite —
  add a `_<arch>` suffix), regenerate figures, add a cross-arch column/row to the relevant tables.
- Update the paper's *External validity* (Limitations) from "single GPU" to the cross-arch result.
