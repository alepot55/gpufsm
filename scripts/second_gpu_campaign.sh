#!/usr/bin/env bash
# Cross-architecture campaign for the SECOND GPU (e.g. A100 / H100 / L40S).
# Re-runs the committed experiments and saves results under paper/data/<ARCH>/ WITHOUT
# overwriting the local RTX 4070 CSVs (those are restored from git after each run).
# Total compute is minutes; the cost is dominated by the one-time CUDA build.
#
# Usage on a freshly-rented cloud GPU box (CUDA + git already present):
#   git clone <repo> gpufsm && cd gpufsm
#   bash scripts/second_gpu_campaign.sh a100        # label = arch tag, no spaces
#
# What it produces (cross-arch falsifiable predictions, see docs/TODO_SECOND_GPU.md):
#   paper/data/a100/dfa_regret.csv          -> DFA L2-knee should shift to the bigger L2
#   paper/data/a100/sweep_techniques.csv    -> regret may rescale but the 2x2 pattern must hold
#   paper/data/a100/costmodel.csv           -> cost-model refit
#   paper/data/a100/worklist_warp.csv (+_batch), worklist_shared.csv, scalar_ablation.csv,
#   regret_multiseed.csv, real_automata_throughput.csv
# Then: commit paper/data/<ARCH>/ and push; the main session integrates a cross-arch column.
set -euo pipefail

ARCH="${1:?usage: second_gpu_campaign.sh <arch-tag>  (e.g. a100, h100, l40s)}"
OUT="paper/data/${ARCH}"
mkdir -p "$OUT"

echo "== [1/3] environment =="
# Install into the EXISTING environment (RunPod PyTorch images already ship torch+triton+CUDA
# devel/nvcc). Do NOT create a venv -- that would reinstall torch and risk a CUDA mismatch.
# Core install + CUDA build; Warp (thread-SIMT backend) added separately; base-env triton reused.
pip install -q -e . --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON
pip install -q warp-lang || echo "  (warp-lang install failed; Warp lines will be skipped)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); \
print('L2 MB:', torch.cuda.get_device_properties(0).L2_cache_size/1e6); \
import triton; print('triton', triton.__version__)" || true
python -m gpufsm.cli env 2>/dev/null || true

echo "== [2/3] run experiments (re-runs; minutes) =="
# Each script writes a *_rtx4070.csv by default; we copy it out then restore the original.
run() {  # run <script-with-args>  <produced-csv...>
  local script="$1"; shift
  echo "  -> $script"
  eval "python $script" || { echo "     (skipped/failed: $script)"; return 0; }
  for f in "$@"; do
    [ -f "paper/data/$f" ] && cp "paper/data/$f" "$OUT/${f/_rtx4070/}" && git checkout -- "paper/data/$f" 2>/dev/null || true
  done
}
run "scripts/sweep_dfa.py"               dfa_regret_rtx4070.csv
run "scripts/calibrate_costmodel.py"     costmodel_rtx4070.csv
run "scripts/sweep_techniques.py $OUT/sweep_techniques.csv"   # this one takes an out path directly
run "scripts/bench_worklist_warp.py"     worklist_warp_rtx4070.csv worklist_warp_batch_rtx4070.csv
run "scripts/bench_worklist_shared.py"   worklist_shared_rtx4070.csv
run "scripts/ablate_scalar_control.py"   scalar_ablation_rtx4070.csv
run "scripts/regret_multiseed.py"        regret_multiseed_rtx4070.csv

echo "== [3/3] done =="
echo "Results under $OUT/ :"; ls -1 "$OUT" || true
echo
echo "Next: git add $OUT && git commit -m 'data($ARCH): cross-arch re-run' && git push"
echo "(Then the main session integrates a cross-arch column and updates External-validity.)"
