#!/usr/bin/env bash
# LANDMARK P3 — one-command cross-architecture re-validation (A100/H100/...).
# Re-runs the regret-law witnesses + M10 cure + P2 selector on the present GPU and compares each
# tile-vs-thread regret to the committed RTX4070 baseline. The falsifiable prediction: regret follows
# the execution PARADIGM, not the arch, so every witness's regret persists in direction (divergent >1,
# pointer-chase ~1) while absolute throughput rescales. Writes paper2/data/cross_arch/regret_<gpu>.csv.
#
# Run on a fresh cloud pod (from the repo root):  bash scripts/run_cross_arch.sh
# Needs: CUDA + nvcc (for the thread kernels) + torch/triton. The gpufsm CUDA ext is built if missing
# (only the automata/M10 path uses it; the other witnesses are self-contained nvcc+triton). The P2
# selector additionally needs the from-source Triton with the thread_region pass (see
# docs/P2_PASS_DESIGN.md); if absent the selector is skipped and the regret comparison still runs.
set -uo pipefail

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python"

echo "== GPU =="
"$PY" -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

echo "== ensure gpufsm CUDA ext (for the automata/M10 path) =="
"$PY" -c "import gpufsm.backends.cuda._cuda" 2>/dev/null && echo "  ext present" || \
  "$PY" -m pip install -q -e . --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON \
     --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$("$PY" -c 'import torch;cc=torch.cuda.get_device_capability(0);print(f"{cc[0]}{cc[1]}-real")')" \
     2>&1 | tail -2 || echo "  (ext build failed; automata witness will be skipped, others still run)"

echo "== run cross-arch re-validation =="
PYTHONPATH="$(pwd)" "$PY" experiments/cure/p3_cross_arch.py

echo
echo "Done. Compare paper2/data/cross_arch/regret_*.csv against the RTX4070 baseline in"
echo "paper2/data/landmark/regret_law.csv. 'persists=1' on every row => paradigm-not-arch confirmed."
