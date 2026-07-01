#!/usr/bin/env bash
# TOMORROW'S A100/H100 RUN — one command. Two phases; phase 1 is fast+reliable, phase 2 is the stretch.
# From repo root on a fresh RunPod (CUDA + torch + triton + nvcc present):  bash scripts/a100_validate.sh
set -uo pipefail
PY="${PY:-.venv/bin/python}"; [ -x "$PY" ] || PY="python"
GPU=$("$PY" -c "import torch;print(torch.cuda.get_device_name(0).replace(' ','_') if torch.cuda.is_available() else 'NOCUDA')")
echo "== GPU: $GPU =="
mkdir -p paper2/data/cross_arch

# ---------- PHASE 1 (FAST, ~15-20min, stock triton+nvcc): regret-law cross-arch ----------
# Validates the CORE thesis (regret follows paradigm not arch) on the datacenter GPU. No from-source build.
echo "== PHASE 1: regret-law cross-arch witnesses =="
bash scripts/run_cross_arch.sh 2>&1 | tee paper2/data/cross_arch/phase1_${GPU}.log || echo "(phase1 had issues; check log)"

# ---------- PHASE 2 (STRETCH, ~60-90min: build from-source Triton + the cure pass, then validate) ----------
# The built cure (per-lane retirement lowering) is the paper's headline; this checks it holds on A100.
echo "== PHASE 2: build from-source Triton + cure pass, validate the built cure =="
TS="${TRITON_SRC:-$HOME/triton_a100}"
if [ ! -d "$TS" ]; then
  git clone https://github.com/triton-lang/triton.git "$TS" && ( cd "$TS" && git checkout c05aa65 2>/dev/null || echo "(using triton main)" )
fi
# apply the cure pass (version-controlled in this repo)
CURE="$(pwd)/experiments/cure/triton_thread_region_pass"
cp "$CURE/ThreadRegion.cpp" "$TS/lib/Dialect/TritonGPU/Transforms/ThreadRegion.cpp"
cp "$CURE/Passes.td"        "$TS/include/triton/Dialect/TritonGPU/Transforms/Passes.td"
# CMakeLists: add NVVM/LLVM dialect link deps (needed by the M1 LLVM-dialect pass)
CML="$TS/lib/Dialect/TritonGPU/Transforms/CMakeLists.txt"
grep -q MLIRNVVMDialect "$CML" || sed -i 's/^  LINK_LIBS PUBLIC$/  LINK_LIBS PUBLIC\n  MLIRNVVMDialect\n  MLIRLLVMDialect/' "$CML"
# add the CMake target line + Passes.h.inc GEN_PASS is handled by Passes.td; the .cpp is already listed if ThreadRegion.cpp was in CMake (it is, from the base pass). Then the pybind + pipeline wiring:
( cd "$TS" && git apply --3way "$CURE/pipeline_wiring.patch" 2>/dev/null || patch -p1 < "$CURE/pipeline_wiring.patch" 2>/dev/null || echo "  (wiring patch: apply manually if it failed)" )
echo "  building triton (this is the ~60-90min part)..."
( cd "$TS" && pip install -e python 2>&1 | tail -3 ) || ( cd "$TS" && pip install -e . 2>&1 | tail -3 ) || echo "  (build failed; see output)"
# validate the built cure masked vs retire on A100
export PYTHONPATH="$TS/python:${PYTHONPATH:-}"
for k in f3_hoist_verify cure_generalize cure_spmv cure_moe; do
  echo "-- $k --"
  TRITON_ALWAYS_COMPILE=1 GPUFSM_THREAD_REGION=      "$PY" experiments/cure/$k.py 2>/dev/null | tee -a paper2/data/cross_arch/cure_${GPU}.log
  TRITON_ALWAYS_COMPILE=1 GPUFSM_THREAD_REGION=retire "$PY" experiments/cure/$k.py 2>/dev/null | tee -a paper2/data/cross_arch/cure_${GPU}.log
done
echo "== DONE. Compare paper2/data/cross_arch/*_${GPU}.* to the RTX4070 baselines. =="
