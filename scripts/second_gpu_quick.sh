#!/usr/bin/env bash
# BUDGET-MINIMAL 2nd-GPU run (~10-12 min, ~$0.25 on an A100). Tests the ONE falsifiable
# cross-arch prediction: the DFA L2-knee shifts from ~6 MB (RTX 4070) to ~the new GPU's L2.
# Builds for ONLY the local arch (fast; skipped if already built), runs a CUDA+Triton DFA
# table-size sweep (no Warp -> avoids its sticky CUDA-716 and saves time), oracle-gated and
# tolerant of a backend that won't run, then prints a CSV to paste back.
# Run on a RunPod PyTorch pod (torch+triton+nvcc already present):
#   git clone https://github.com/alepot55/gpufsm && cd gpufsm
#   git checkout worktree-bright-oak-9zos   # if the script isn't on the cloned branch
#   bash scripts/second_gpu_quick.sh
set -euo pipefail

echo "== build (local arch only -> fast; skipped if already built) =="
if python -c "import gpufsm.backends.cuda._cuda" 2>/dev/null; then
  echo "  CUDA extension already present -> skip build"
else
  ARCHS="$(python -c 'import torch;cc=torch.cuda.get_device_capability(0);print(f"{cc[0]}{cc[1]}-real")' 2>/dev/null || echo 80-real)"
  echo "  building for sm_$ARCHS"
  pip install -q -e . --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON \
      --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCHS"
fi

python - <<'PY'
import statistics, random, torch
from gpufsm.dfa import random_dfa, simulate_dfa
from gpufsm.dfa_api import run_dfa_batch

name = torch.cuda.get_device_name(0)
p = torch.cuda.get_device_properties(0)
l2 = next((getattr(p, a) for a in ("L2_cache_size", "l2_cache_size") if getattr(p, a, None)), None)
print(f"\nGPU: {name} | L2 ~ {(l2/1e6 if l2 else float('nan')):.0f} MB  "
      f"(RTX 4070 was 6 MB; the CUDA knee should peak near this L2)\n")

NSTR, SLEN, RUNS, WARM = 4096, 256, 7, 2
rng = random.Random(0)
batch = [bytes(rng.randint(0, 255) for _ in range(SLEN)) for _ in range(NSTR)]
bits = NSTR * SLEN * 8

# oracle-gate each backend; keep only the ones that run + match (CUDA is the key line)
dfa = random_dfa(8192, accept_prob=0.02, seed=8192)
vb = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 40))) for _ in range(32)]
ref = [simulate_dfa(dfa, b) for b in vb]
backends = []
for be in ("cuda", "triton"):
    try:
        got = [(r.accepted, r.match_len) for r in run_dfa_batch(dfa, vb, backend=be)]
        if got == ref:
            backends.append(be); print(f"{be}: oracle OK")
        else:
            print(f"{be}: MISMATCH -> skip")
    except Exception as e:
        print(f"{be}: failed ({type(e).__name__}) -> skip")

def med(d, be):
    for _ in range(WARM): run_dfa_batch(d, batch, backend=be)
    return statistics.median(run_dfa_batch(d, batch, backend=be)[0].kernel_ms for _ in range(RUNS))

print("\nbackend,num_states,table_mb,throughput_gbps,gpu")
for n in [2048, 8192, 16384, 32768, 40960, 49152, 81920, 131072]:
    d = random_dfa(n, accept_prob=0.02, seed=n)
    for be in backends:
        g = bits / (med(d, be) * 1e-3) / 1e9
        print(f"{be},{n},{n // 1024},{g:.1f},{name.replace(',', '')}")
print("\n=== PASTE EVERYTHING FROM 'backend,num_states' DOWN BACK TO THE ASSISTANT ===")
PY
