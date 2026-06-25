# Reproducibility

The project is designed to be re-run end-to-end by anyone, without privileged access.

## Environment

```bash
python -m pip install -e ".[dev]"            # CPU-only: reference + bit-packed, no GPU needed
python -m pip install -e ".[dev,triton]"     # + Triton backend (requires a CUDA GPU)
GPUFSM_BUILD_CUDA=ON python -m pip install -e ".[dev]"   # + CUDA backend (requires toolkit + GPU)
gpufsm env                                   # capture python / numpy / backend availability
```

Pin the exact GPU, driver and toolkit versions in any reported results.

## Correctness

```bash
pytest -m "not gpu" -q     # CPU: reference vs bit-packed (incl. 300-case fuzz), API, CLI, datasets
pytest -m gpu -q           # on a GPU box: Triton/CUDA verdicts must equal the CPU reference
```

The CPU reference (`gpufsm.reference`) is the single oracle. The bit-packed simulator and every GPU
backend are checked against it on the full suite — this is the cross-implementation validation the prior
study lacked.

## Benchmarks

```bash
gpufsm sweep --repeats 10 --out results/sweep.csv   # mean/std/CI95 per (backend, technique)
```

- Always report `warmup` and `repeats`; the CSV carries `mean_ms`, `std_ms`, `ci95_ms`.
- Kernel time and transfer time are reported separately.
- Re-run on ≥2 distinct GPUs for the paper.

## Data

- Small fixtures are vendored under `data/`.
- The large ANMLZoo/AutomataZoo suite is fetched on demand via `gpufsm.io.datasets.ensure`, which
  **verifies SHA-256** before use and caches the result. No private/SharePoint links.

## Figures and paper

```bash
# regenerate figures strictly from versioned CSVs, then build the paper
python paper/generate_figures.py
latexmk -pdf paper/main.tex
```

Figures must depend only on committed CSVs so the paper rebuilds deterministically.
