#!/usr/bin/env bash
# COST-OPTIMIZED A100/H100 validation. Two phases; PHASE 1 is cheap+robust (no build), PHASE 2 (the cure)
# builds Triton with MAX parallelism (fast on a big pod). Minimal experiment set — only what the paper needs.
#   Cheap (regret-law only, ~10-15min):   bash scripts/a100_validate.sh
#   Full  (+ built-cure on A100, ~+15min): CURE=1 bash scripts/a100_validate.sh
set -uo pipefail
PY="${PY:-python}"; command -v "$PY" >/dev/null || PY=python3
GPU=$("$PY" -c "import torch;print(torch.cuda.get_device_name(0).replace(' ','_') if torch.cuda.is_available() else 'NOCUDA')" 2>/dev/null || echo NOCUDA)
NPROC=$(nproc)
echo "== GPU=$GPU  cores=$NPROC  (cost = minutes x hourly-rate; keep it short) =="
mkdir -p paper2/data/cross_arch
t0=$(date +%s 2>/dev/null || echo 0)

# deps (stock triton+torch are enough for PHASE 1; the CUDA image already has nvcc)
"$PY" -c "import torch,triton" 2>/dev/null || pip install -q torch triton 2>&1 | tail -1

# ---------- PHASE 1 (CHEAP, no build): regret-law cross-arch — closes the single-GPU gap for the thesis ----------
echo "== PHASE 1: regret-law cross-arch (stock triton + nvcc) =="
PYTHONPATH="$(pwd)" "$PY" experiments/cure/p3_cross_arch.py 2>&1 | tee paper2/data/cross_arch/regret_${GPU}.log || echo "(phase1 issues; see log)"

# ---------- PHASE 2 (OPT-IN CURE=1): build Triton+cure with MAX parallelism, validate the built cure ----------
if [ "${CURE:-0}" = "1" ]; then
  echo "== PHASE 2: build Triton + cure pass (MAX_JOBS=$NPROC) then validate =="
  TS="${TRITON_SRC:-$HOME/triton_a100}"
  [ -d "$TS/.git" ] || git clone --depth 1 https://github.com/triton-lang/triton.git "$TS"
  CURE="$(pwd)/experiments/cure/triton_thread_region_pass"
  cp "$CURE/ThreadRegion.cpp" "$TS/lib/Dialect/TritonGPU/Transforms/ThreadRegion.cpp"
  cp "$CURE/Passes.td"        "$TS/include/triton/Dialect/TritonGPU/Transforms/Passes.td"
  CML="$TS/lib/Dialect/TritonGPU/Transforms/CMakeLists.txt"
  grep -q MLIRNVVMDialect "$CML" || sed -i 's/^  LINK_LIBS PUBLIC$/  LINK_LIBS PUBLIC\n  MLIRNVVMDialect\n  MLIRLLVMDialect/' "$CML"
  ( cd "$TS" && git apply --3way "$CURE/pipeline_wiring.patch" 2>/dev/null || patch -p1 <"$CURE/pipeline_wiring.patch" 2>/dev/null || echo "  (apply wiring manually if needed)" )
  echo "  building (MAX_JOBS=$NPROC — this is the cost; a big pod finishes in ~15min)..."
  ( cd "$TS" && MAX_JOBS=$NPROC pip install -e . 2>&1 | tail -3 )
  export PYTHONPATH="$TS/python:$(pwd)"
  # MINIMAL cure set: f3 (headline) + generalize; skip spmv/moe on the pod to save time (RTX4070 has them)
  for k in f3_hoist_verify cure_generalize; do
    echo "-- $k (masked vs retire) --"
    TRITON_ALWAYS_COMPILE=1 GPUFSM_THREAD_REGION=       "$PY" experiments/cure/$k.py 2>/dev/null | tee -a paper2/data/cross_arch/cure_${GPU}.log
    TRITON_ALWAYS_COMPILE=1 GPUFSM_THREAD_REGION=retire "$PY" experiments/cure/$k.py 2>/dev/null | tee -a paper2/data/cross_arch/cure_${GPU}.log
  done
fi
t1=$(date +%s 2>/dev/null || echo 0)
echo "== DONE in ~$(( (t1-t0)/60 )) min on $GPU. git add paper2/data/cross_arch/*_${GPU}.* && commit. =="
echo "== SHUT DOWN THE POD NOW to stop billing. =="
