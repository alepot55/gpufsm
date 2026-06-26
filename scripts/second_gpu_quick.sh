#!/usr/bin/env bash
# BUDGET-MINIMAL 2nd-GPU run (~10-12 min, ~$0.25 on an A100). Tests the ONE falsifiable
# cross-arch prediction: the DFA L2-knee shifts from ~6 MB (RTX 4070) to ~the new GPU's L2.
# Builds for ONLY the local arch (fast), runs a CUDA+Triton DFA table-size sweep (no Warp, to
# avoid its intermittent sticky CUDA-716 and save time), validates one size vs the oracle, and
# prints a CSV to paste back. Run on a RunPod PyTorch pod (torch+triton+nvcc already present):
#   git clone https://github.com/alepot55/gpufsm && cd gpufsm
#   bash scripts/second_gpu_quick.sh
set -euo pipefail

echo "== build (local arch only -> fast) =="
ARCHS="$(python -c 'import torch;cc=torch.cuda.get_device_capability(0);print(f"{cc[0]}{cc[1]}-real")' 2>/dev/null || echo 80-real)"
echo "  building for sm_$ARCHS"
pip install -q -e . --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON \
    --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCHS"

python - <<'PY'
import statistics, random
import torch
from gpufsm.dfa import random_dfa, simulate_dfa
from gpufsm.dfa_api import run_dfa_batch

name = torch.cuda.get_device_name(0)
l2 = torch.cuda.get_device_properties(0).L2_cache_size / 1e6
print(f"\nGPU: {name} | L2 = {l2:.0f} MB  (RTX 4070 was 6 MB; knee should peak near this L2)")

# table_kb == num_states (256 int32 = 1 KB/state). Sizes bracket the L2 (MB == states/1024).
SIZES = [2048, 8192, 16384, 32768, 40960, 49152, 81920, 131072]
NSTR, SLEN, RUNS, WARM = 4096, 256, 7, 2
rng = random.Random(0)
batch = [bytes(rng.randint(0, 255) for _ in range(SLEN)) for _ in range(NSTR)]
bits = NSTR * SLEN * 8

# correctness gate on one mid size (cheap)
dfa = random_dfa(8192, accept_prob=0.02, seed=8192)
vb = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 40))) for _ in range(32)]
ref = [simulate_dfa(dfa, b) for b in vb]
for be in ("cuda", "triton"):
    got = [(r.accepted, r.match_len) for r in run_dfa_batch(dfa, vb, backend=be)]
    assert got == ref, f"{be} != oracle"
print("oracle check: cuda & triton == reference  OK\n")

def med(dfa, be):
    for _ in range(WARM): run_dfa_batch(dfa, batch, backend=be)
    return statistics.median(run_dfa_batch(dfa, batch, backend=be)[0].kernel_ms for _ in range(RUNS))

print("backend,num_states,table_mb,throughput_gbps,gpu")
rows = []
for n in SIZES:
    d = random_dfa(n, accept_prob=0.02, seed=n)
    for be in ("cuda", "triton"):
        g = bits / (med(d, be) * 1e-3) / 1e9
        line = f"{be},{n},{n/1024:.0f},{g:.1f},{name.replace(',','')}"
        print(line); rows.append(line)
print("\n=== PASTE EVERYTHING FROM 'backend,num_states' DOWN BACK TO THE ASSISTANT ===")
PY
