# Artifact Appendix

SIGPLAN/USENIX-style artifact-evaluation appendix for **gpufsm — "The Two Faces of
Abstraction Regret"**. Companion to `docs/REPRODUCIBILITY.md` (full claim→command map) and
the paper's *Artifact Availability* section.

## Abstract

The artifact is the `gpufsm` repository: a single registry-based framework with a CPU
reference oracle, GPU backends (CUDA / Triton / Warp) and a Gluon expressibility probe, plus
the versioned result CSVs and the figure generator. It reproduces every figure, table, and
headline number in the paper from committed data via one command, validates every backend
bit-for-bit against the oracle, and demonstrates the falsifiable Triton↔Gluon control. The
CPU portion runs with no GPU; the GPU portion runs on any CUDA device (results in the paper
are from an RTX 4070, sm_89, 6 MB L2).

## Artifact check-list (metadata)

- **Algorithm:** NFA simulation (active-set + ε-closure) and DFA dense-table walk; a
  work-efficient worklist and a block-parallel (warp-per-string) variant.
- **Program:** Python 3.10+ package `gpufsm`; CUDA C++ kernels (pybind11); Triton & Warp kernels.
- **Compilation:** CMake ≥3.18 via scikit-build-core (CUDA, opt-in); Triton/Warp are wheels.
- **Run-time environment:** Linux; CUDA toolkit for GPU backends; no admin rights required
  (Nsight counters are optional and *not* needed for any claim).
- **Hardware:** any CUDA GPU for the GPU claims; CPU-only suffices for correctness + the cost model.
- **Metrics:** throughput (Gbps), median + bootstrap CI95; predicted-vs-measured fit error.
- **Output:** CSVs in `paper/data/`, figures in `paper/figures/`, pass/fail from `pytest`.
- **Experiments:** correctness suite, technique sweep, cost-model calibration, DFA table-size
  sweep, warp-vs-global speedup, Gluon falsifiability probe.
- **Disk / time:** < 1 GB; CPU suite < 1 min; full GPU sweeps a few minutes on one GPU.
- **Public:** yes (MIT). **Archived DOI:** Zenodo DOI minted at first tagged release.

## Installation

```bash
# CPU-only (correctness + cost model; no GPU needed):
python -m pip install -e ".[dev]"

# GPU backends (CUDA built via CMake; Triton/Warp wheels) — on a CUDA box:
pip install -e ".[dev,triton,warp]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON
gpufsm env      # records python / numpy / backend availability + versions
```

## Experiment workflow

```bash
pytest -m "not gpu"                      # correctness on CPU (oracle + bit-packed spec)
pytest -m gpu                            # GPU backends == oracle, bit-for-bit (needs a GPU)
python scripts/sweep_techniques.py       # technique throughput sweep -> sweep CSV
python scripts/calibrate_costmodel.py    # cost-model fit -> costmodel CSV
python scripts/sweep_dfa.py              # DFA table-size sweep -> dfa_regret CSV (L2 knee)
python scripts/bench_worklist_warp.py    # block-parallel speedup -> worklist_warp{,_batch} CSV
python scripts/bench_worklist_shared.py  # shared-mem working-set ablation -> worklist_shared CSV
python scripts/ablate_scalar_control.py  # CAUSAL tile-vs-scalar cliff -> scalar_ablation CSV
python scripts/regret_multiseed.py       # multi-seed regret robustness -> regret_multiseed CSV
python scripts/validate_costmodel.py     # cost-model holdout (predictive for CUDA, not Triton)
python scripts/gluon_probe.py            # Gluon falsifiability probe (exit 0 = expected failure)
python paper/figures.py                  # regenerate all figures from the CSVs
```

## Evaluation and expected results (claims → commands)

| Paper claim | Command | Expected |
| --- | --- | --- |
| Every backend == CPU oracle (latch-first-match) | `pytest -m gpu` | all pass, 0 mismatches |
| 6 real ANMLZoo automata validate bit-for-bit | `pytest tests/test_anmlzoo_gpu.py -m gpu` | 6 passed |
| Faithful kernel is compute-bound (memory axes inert) | `scripts/sweep_techniques.py` | multistream ≈ _shared ≈ _async within CI |
| NFA abstraction regret = paradigm | `scripts/calibrate_costmodel.py` | Triton 6–8× / 10× fit, Warp 0.6–0.9× vs CUDA |
| DFA memory-bound L2 knee | `scripts/sweep_dfa.py` | CUDA peaks ~6 MB then ~2.4× drop; Triton flat ~30 Gbps |
| Block-parallel warp worklist speedup | `scripts/bench_worklist_warp.py` | 3–9× vs single-thread on real automata at a saturating batch (larger at small batch) |
| CAUSAL: scalar-control cliff in Triton | `scripts/ablate_scalar_control.py` | tile ~1320 vs scalar ~81 Gbps = ~16× cliff (saturating batch) |
| Regret is multi-seed robust | `scripts/regret_multiseed.py` | Triton 7.1× median, Warp 0.85× median (5 seeds × 3 sizes) |
| Shared-mem working-set inert | `scripts/bench_worklist_shared.py` | 0.99–1.10× vs warp (layout not the bottleneck) |
| Cost model predictive for CUDA only | `scripts/validate_costmodel.py` | holdout CUDA 2.7% / Triton 45%; LOO stable / unstable |
| Gluon cannot express the kernel (falsifiable) | `scripts/gluon_probe.py` | "EXPECTED FAILURE … no scalar load", exit 0 |

Exact figures and the full mapping are in `docs/REPRODUCIBILITY.md`; all numbers trace to
`paper/data/*.csv`.

## Reusability

Adding a backend or technique is one file + one `@register(Backend, technique)` line; the CSR
NFA representation and the oracle are shared, so a new kernel is compared apples-to-apples and
gated for correctness automatically.

## Archival / DOI

On the first tagged release we mint a Zenodo DOI from the GitHub release (Zenodo–GitHub
integration), pin it in `CITATION.cff` (`doi:`/`identifiers:`) and the paper, and attach the
exact `paper/data/*.csv` used for the submitted figures.
