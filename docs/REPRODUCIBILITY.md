# Reproducibility / Artifact Guide

The project is designed to be re-run end-to-end by anyone, without privileged access.
Every figure and headline number regenerates from committed code + versioned CSVs.

## Environment

CPU-only (reference + bit-packed spec, no GPU) installs cleanly:

```bash
python -m pip install -e ".[dev]"
gpufsm env            # capture python / numpy / backend availability + versions
```

GPU backends (Triton / CUDA / Warp) — on a CUDA box:

```bash
# Triton + Warp are pure-Python wheels; CUDA needs the toolkit and is built via CMake.
pip install -e ".[dev,triton,warp]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON
```

Gotchas (learned on the reference host, RTX 4070 / CUDA 13.x):
- **`GPUFSM_BUILD_CUDA=ON` as an env var is NOT enough** — scikit-build-core reads the
  define from `pyproject.toml`; pass it via `--config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON`.
- **Toolkit newer than the driver's max CUDA** => embedded PTX is rejected at load
  (`PTX ... unsupported toolchain`). CMake defaults to **real SASS only**
  (`75/80/86/89-real`); override `-DCMAKE_CUDA_ARCHITECTURES` for other GPUs. Avoid bare
  arch numbers and `native` (they embed PTX).
- On PEP 668 ("externally-managed") hosts use a venv: `python -m venv --system-site-packages .venv`.

Pin the exact GPU, driver and toolkit versions in any reported result (the sweep CSV
records GPU + torch/triton/warp/cuda versions per row).

## Correctness (the cross-implementation validation the prior study lacked)

```bash
pytest -m "not gpu" -q    # 32 CPU tests: reference vs bit-packed (300-case fuzz), cost model,
                          # ANML loader (fixture + round-trip), API, CLI, datasets
pytest -m gpu -q          # on a GPU box: every backend/technique verdict == the CPU reference
                          #   (examples, fuzz, batch run_batch, >64-state multiword batch)
gpufsm verify             # cross-backend agreement on the example suite (0 failures expected)
```

`gpufsm.reference` is the single oracle (latch-first-match). The bit-packed spec
(`gpufsm.bitmap`) and every GPU technique are checked bit-identical to it.

## Headline results -> exact commands

| Claim | Command | Artifact |
|---|---|---|
| Throughput sweep (median+CI95), all techniques x sizes | `python scripts/sweep_techniques.py` | `paper/data/sweep_techniques.csv` |
| Cost-model calibration + abstraction-regret ratios | `python scripts/calibrate_costmodel.py` | `paper/data/costmodel_rtx4070.csv`, `docs/RESULTS_COSTMODEL.md` |
| Figures (throughput, worklist speedup, memory ablation, regret) | `python paper/figures.py` | `paper/figures/fig_*.pdf` / `.png` |
| Worklist 15-170 Gbps vs full-scan; memory axes within noise | sweep CSV rows | `fig_throughput_vs_states`, `fig_memory_ablation` |
| Abstraction regret: Triton 6-8x throughput / 10.1x fit (full-scan), ~6.5x (worklist) vs CUDA; Warp 0.6-0.9x | calibrate + sweep | `fig_abstraction_regret`, `docs/RESULTS_COSTMODEL.md` |
| DSL expressibility (CUDA/Warp express; Triton strains; Gluon cannot) | n/a (documented + probed) | `docs/DSL_EXPRESSIVENESS.md` |

Figures depend **only** on committed CSVs, so the paper rebuilds deterministically. The
sweep/calibration scripts skip unsupported (backend, technique, size) cells (e.g. Triton/Warp
worklist > 64 states) with a log line rather than failing.

## Profiling (Nsight)

GPU performance counters are admin-gated on the reference host; see `docs/PROFILING.md` for
the one-time enable (`sudo`, or `NVreg_RestrictProfilingToAdminUsers=0` + reboot) and the
`ncu` / `scripts/profile_target.py` recipe. The compute-bound claim does **not** depend on
counters — it is established by controlled ablation (`multistream_shared`, modeled CSR
traffic = 0, ties `multistream`) and 1/n^2 scaling.

## Data

- Small fixtures are vendored under `data/`.
- Real ANMLZoo automata fetched on demand via `gpufsm.io.datasets.ensure`, which **verifies
  SHA-256** and refuses unverified downloads. Pinned (public `jackwadden/ANMLZoo` mirror):
  `levenshtein` (2787 states), `hamming` (11349 states) — pure-homogeneous, `all-input`.
  `gpufsm.io.anml.load_anml` parses them with correct all-input/start-of-data semantics
  (fixtures in `tests/test_anml.py`).
- Real-suite validation: `python scripts/run_anmlzoo.py [levenshtein|hamming]` fetches, loads,
  runs on `worklist_global`, and checks GPU == reference (0 mismatches). Also
  `tests/test_anmlzoo_gpu.py` (gpu-marked; skips offline/CPU-only).

## Paper

The current working draft is `paper/DRAFT.md` (Markdown; LaTeX migration pending). All its
numbers trace to the CSVs/docs above.
