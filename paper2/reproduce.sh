#!/usr/bin/env bash
# Reproduce every paper-2 number and figure from scratch on a CUDA box.
# Each experiment is correctness-gated against the CPU oracle before reporting throughput,
# and writes a versioned CSV under paper2/data/; figures.py regenerates the plots from those CSVs.
#
# Usage:  bash paper2/reproduce.sh           # throughput + figures (no sudo)
#         bash paper2/reproduce.sh --nsight  # also re-run the Nsight mechanism profiles (needs sudo ncu)
#
# Env: a venv with torch+triton and gpufsm built (+CUDA); run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
PY="${PY:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "== M0  anchor (triton/worklist vs cuda/worklist = ~10x) =="
$PY experiments/cure/m0_anchor.py
echo "== M2f num_warps artifact (~3.4x) =="
$PY experiments/cure/m2f_numwarps.py
echo "== M2/M2c lane-packing, occupancy-gated =="
for B in 4096 16384 65536; do M2_N_STRINGS=$B $PY experiments/cure/m2_lane_packed.py; done
echo "== M2e worklist head-to-head (decomposition) =="
for B in 4096 16384 65536; do M2_N_STRINGS=$B $PY experiments/cure/m2e_worklist_packed.py; done
echo "== M3-lite per-lane worklist (residual 0.51x) + M3-lite-b occupancy sweep =="
$PY experiments/cure/m3_lite_scalarlane.py
$PY experiments/cure/m3_lite_b_occupancy.py
echo "== M4 DFA regime crossover =="
$PY experiments/cure/m4_dfa.py
echo "== M9 multi-word (>64 states, correctness) =="
$PY experiments/cure/m9_multiword.py
echo "== M10 the cure implemented (scalar_program -> threads, 4.2x) =="
M2_N_STRINGS=16384 $PY experiments/cure/m10_scalar_program.py
echo "== Gluon expressibility probe (layout != control flow) =="
$PY scripts/gluon_probe.py || true

if [[ "${1:-}" == "--nsight" ]]; then
  echo "== Nsight mechanism profiles (sudo ncu) — see docs/CURE_PROGRESS.md for the exact metric sets =="
  echo "   (M1/M2e/M3-lite/M5/M5b/M4/M10 profiles; require sudo and pin the toolkit version)"
fi

echo "== Regenerate figures from the versioned CSVs =="
$PY paper2/figures.py
echo "DONE. Numbers in paper2/data/*.csv ; figures in paper2/figures/*.png ; paper paper2/gpufsm2.tex"
