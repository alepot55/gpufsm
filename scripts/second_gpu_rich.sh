#!/usr/bin/env bash
# RICHER 2nd-GPU run (~10-15 min, ~$0.25). Cross-arch confirmation of the CENTERPIECE:
#   (1) the 2x2 NFA regret (Triton vs Warp vs CUDA) -- does the paradigm gap hold/rescale?
#   (2) the DFA knee with all three backends (the Triton-flat line, now on a Triton-3.5 stack)
#   (3) the causal tile-vs-scalar cliff
# Upgrades Triton (the pod image ships 3.0, which mis-codegens our kernels) + installs Warp.
# Build is skipped if already present. Every backend is oracle-gated and skipped if it won't run.
# Run on the SAME pod (after second_gpu_quick.sh):  bash scripts/second_gpu_rich.sh
set -euo pipefail

echo "== ensure build + Triton 3.5 + Warp =="
python -c "import gpufsm.backends.cuda._cuda" 2>/dev/null && echo "  ext present" || \
  pip install -q -e . --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON \
     --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$(python -c 'import torch;cc=torch.cuda.get_device_capability(0);print(f"{cc[0]}{cc[1]}-real")')"
pip install -q -U "triton>=3.5" 2>&1 | tail -1 || echo "  (triton upgrade failed; will use whatever is present)"
pip install -q warp-lang 2>&1 | tail -1 || echo "  (warp install failed)"
python -c "import triton,torch;print('triton',triton.__version__,'| torch',torch.__version__)" || true

python - <<'PY'
import statistics, random, torch
from gpufsm import NFABuilder, ANY_SYMBOL, simulate, run_batch
from gpufsm.dfa import random_dfa, simulate_dfa
from gpufsm.dfa_api import run_dfa_batch
name = torch.cuda.get_device_name(0)
print(f"\nGPU: {name}\n")

def mk_nfa(n, seed):
    rng = random.Random(seed); b = NFABuilder()
    for _ in range(n): b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)): b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()

batch = [bytes(b"abcde"[i % 5] for i in range(256)) for _ in range(2048)]
bits = 2048 * 256 * 8
def medms(nfa, be, te):
    for _ in range(2): run_batch(nfa, batch, be, te)
    return statistics.median(run_batch(nfa, batch, be, te)[0].kernel_ms for _ in range(7))

# which NFA backends run + match the oracle
probe = mk_nfa(48, 7)
pin = [bytes(ord(random.Random(i).choice("abcde")) for _ in range(20)) for i in range(8)]
ref = [simulate(probe, d) for d in pin]
ok = {}
for be in ("cuda", "triton", "warp"):
    try:
        got = [(r.accepted, r.match_len) for r in run_batch(probe, pin, be, "multistream")]
        ok[be] = (got == ref); print(f"NFA {be}: {'OK' if ok[be] else 'MISMATCH'}")
    except Exception as e:
        ok[be] = False; print(f"NFA {be}: failed ({type(e).__name__})")

print("\n# (1) NFA regret 2x2 vs CUDA  [regret = cuda/dsl ; <1 means DSL faster]")
print("num_states,triton_regret,warp_regret")
for n in (32, 48, 64):
    tr, wr = [], []
    for seed in range(3):
        nfa = mk_nfa(n, 1000 + seed * 7 + n)
        c = bits / (medms(nfa, "cuda", "multistream") * 1e-3) / 1e9
        if ok.get("triton"): tr.append(c / (bits / (medms(nfa, "triton", "multistream") * 1e-3) / 1e9))
        if ok.get("warp"):   wr.append(c / (bits / (medms(nfa, "warp", "multistream") * 1e-3) / 1e9))
    t = f"{statistics.median(tr):.1f}" if tr else "n/a"
    w = f"{statistics.median(wr):.2f}" if wr else "n/a"
    print(f"{n},{t},{w}")

# DFA knee with all backends that match the oracle
dfa8 = random_dfa(8192, accept_prob=0.02, seed=8192)
vb = [bytes(random.Random(i).randint(0,255) for _ in range(24)) for i in range(32)]
dref = [simulate_dfa(dfa8, x) for x in vb]
dbe = []
for be in ("cuda", "triton", "warp"):
    try:
        g = [(r.accepted, r.match_len) for r in run_dfa_batch(dfa8, vb, backend=be)]
        (dbe.append(be) if g == dref else None)
    except Exception: pass
print(f"\n# (2) DFA knee  (backends matching oracle: {dbe})")
print("backend,num_states,table_mb,throughput_gbps")
dbatch = [bytes(random.Random(99).randint(0,255) for _ in range(256)) for _ in range(4096)]
dbits = 4096*256*8
def dmed(d, be):
    for _ in range(2): run_dfa_batch(d, dbatch, backend=be)
    return statistics.median(run_dfa_batch(d, dbatch, backend=be)[0].kernel_ms for _ in range(7))
for n in (2048, 8192, 16384, 32768, 40960, 49152, 81920, 131072):
    d = random_dfa(n, accept_prob=0.02, seed=n)
    for be in dbe:
        print(f"{be},{n},{n//1024},{dbits/(dmed(d,be)*1e-3)/1e9:.1f}")
print("\n=== PASTE FROM '# (1)' DOWN BACK TO THE ASSISTANT ===")
PY
