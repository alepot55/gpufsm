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
1. **DFA L2-knee shifts right.** `scripts/sweep_dfa.py` — the CUDA peak should move from ~6 MB
   (4070 L2) to ~the new L2 size; the ~2.2× drop past L2 should persist. A clean falsifiable
   prediction of the memory-bound reading. (Triton should stay flat ~tens of Gbps regardless.)
2. **Regret factors may rescale but the 2×2 pattern holds.** Re-fit the cost model
   (`scripts/calibrate_costmodel.py`) and re-measure the regret (`scripts/sweep_techniques.py`):
   absolute Triton regret (6–8×) and Warp (0.6–0.9×) may shift with the arch, but regret must
   still track the *paradigm column*, not the *height row*.
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
