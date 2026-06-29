# `experiments/cure/` — reproducibility index (paper 2: "From Diagnosis to Cure")

Every claim in `paper2/gpufsm2.tex` traces to one of these runnable artifacts and the versioned CSV it
writes under `paper2/data/`. Figures are regenerated from those CSVs by `paper2/figures.py`.

**Stacks.** Most artifacts use the system stack: `.venv/bin/python` (Triton 3.5.1, torch 2.9.1+cu128,
gpufsm built `+CUDA`), run from the repo root. The artifacts marked **[src-triton]** additionally need
the from-source Triton 3.8 that carries our `tritongpu-thread-region` MLIR pass — prefix the command with
`PYTHONPATH=$HOME/m3full_build/triton-src/python` (build recipe: `docs/P2_PASS_DESIGN.md`). All
throughput artifacts are oracle-gated (bit-for-bit vs a CPU reference) before any number is reported.

## The decomposition — NFA worklist regret (§3, `sec:decomp`/`sec:resid`)
| Artifact | Supports | Run | Writes |
|---|---|---|---|
| `m0_anchor.py` | the 10.1× anchor (Triton 22 vs CUDA 227 Gbps) | `.venv/bin/python experiments/cure/m0_anchor.py` | `paper2/data/m0_anchor_rtx4070.csv` |
| `m2f_numwarps.py` | Component A: `num_warps` launch artifact (~3.4×) | `.venv/bin/python experiments/cure/m2f_numwarps.py` | `m2f_numwarps_rtx4070.csv` |
| `m2_lane_packed.py` | Component B: pure lane-packing (3.2→19.4×), occupancy-gated | `.venv/bin/python experiments/cure/m2_lane_packed.py` | `m2_lane_packed_rtx4070.csv` |
| `m2e_worklist_packed.py` | decomposition table / Fig.1 throughput ladder | `.venv/bin/python experiments/cure/m2e_worklist_packed.py` | `m2e_worklist_packed_rtx4070.csv` |
| `m3_lite_scalarlane.py` | Component C residual (per-lane WP2 vs CUDA, ~2×) + `launch_wp2` | `.venv/bin/python experiments/cure/m3_lite_scalarlane.py` | `m3_lite_rtx4070.csv` |
| `m3_lite_b_occupancy.py` | BLOCK sweep worsens regret (lock-step divergence) | `.venv/bin/python experiments/cure/m3_lite_b_occupancy.py` | `m3_lite_b_occupancy_rtx4070.csv` |
| `m4_dfa.py` | §4 regime crossover: DFA matches CUDA (1.05×) past L2 | `.venv/bin/python experiments/cure/m4_dfa.py` | `m4_dfa_rtx4070.csv` |
| `m9_multiword.py` | multi-word (>64 state) scaling check | `.venv/bin/python experiments/cure/m9_multiword.py` | `m9_multiword_rtx4070.csv` |

## The cure (§5, `sec:implemented`)
| Artifact | Supports | Run | Writes |
|---|---|---|---|
| `m10_scalar_program.py` | the implemented cure: SP/WP2 = 4.2×, SP/CU = 2.15×, oracle-gated | `.venv/bin/python experiments/cure/m10_scalar_program.py` | `m10_scalar_program_rtx4070.csv` |

## Generality — the regret law (§6, `sec:law`, `fig:law`)
Six oracle-gated tile-vs-thread witnesses; the unified table is `paper2/data/landmark/regret_law.csv`.
| Artifact | Witness / channel | Run | Writes |
|---|---|---|---|
| `landmark_bfs.py` | graph pointer-chase = negative control (1.00×) | `.venv/bin/python experiments/cure/landmark_bfs.py` | `landmark/bfs_rtx4070.csv` |
| `landmark_spmv.py` | tile-lowering baseline (uniform 1.9×) + divergence increment (power-law) | `.venv/bin/python experiments/cure/landmark_spmv.py` | `landmark/spmv_rtx4070.csv` |
| `landmark_rejection.py` | masked-lane waste, pure control-flow (4.0×) | `.venv/bin/python experiments/cure/landmark_rejection.py` | `landmark/rejection_rtx4070.csv` |
| `landmark_hashprobe.py` | dependent-load, clean gather (1.4×) | `.venv/bin/python experiments/cure/landmark_hashprobe.py` | `landmark/hashprobe_rtx4070.csv` |

## In the compiler — P2 (§5, `sec:compiler`)
The pass sources are version-controlled in `triton_thread_region_pass/` (`ThreadRegion.cpp` +
`registration.patch` + `README.md`, base Triton commit `c05aa65`); apply + rebuild per that README.
| Artifact | Supports | Run | Output |
|---|---|---|---|
| `p2_ttgir_probe.py` **[src-triton]** | the lock-step signature in real TTGIR | `PYTHONPATH=$HOME/m3full_build/triton-src/python .venv/bin/python experiments/cure/p2_ttgir_probe.py` | `landmark/p2_lockstep.ttgir` |
| `p2_pass_verify.py` **[src-triton]** | the detection pass fires in `libtriton` (ON→tag, OFF→none) | `PYTHONPATH=$HOME/m3full_build/triton-src/python .venv/bin/python experiments/cure/p2_pass_verify.py` | stdout (VERIFIED) |
| `p2_lowering_wall.py` | in-IR lowering structurally impossible (`scf.condition` is `i1`) | `.venv/bin/python experiments/cure/p2_lowering_wall.py` | stdout (WALL CONFIRMED) + uses `triton_thread_region_pass/perlane_while_attempt.mlir` |
| `p2_selector.py` **[src-triton]** | automatic detect→route→thread cure (3.9×), neg-control on tile | `.venv/bin/python experiments/cure/p2_selector.py` | `landmark/p2_selector_rtx4070.csv` |

> `p2_selector.py` runs detection in a from-source-Triton subprocess itself, so the top-level command
> uses the system `.venv`; only its internal detection step needs the src-triton `PYTHONPATH` (it sets it).

## Cross-architecture — P3 (hardware-gated; `Threats`/`Limitations`)
| Artifact | Supports | Run | Writes |
|---|---|---|---|
| `p3_cross_arch.py` | regret follows the paradigm, not the arch (persists on A100/H100) | `.venv/bin/python experiments/cure/p3_cross_arch.py` (or `bash scripts/run_cross_arch.sh`) | `paper2/data/cross_arch/regret_<gpu>.csv` |

`p3_cross_arch.py` is non-mutating: it snapshots, runs, reads, and restores each committed
`*_rtx4070.csv`, writing this GPU's numbers separately under `cross_arch/`.
