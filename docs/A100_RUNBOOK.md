# RunPod A100 — minimal-cost runbook (datacenter validation for the TACO paper)

**Goal:** the datacenter numbers that close the "single consumer GPU" gap, for the least money.
**Cost = minutes on the pod × hourly rate.** So: cheapest A100, run fast, SHUT DOWN immediately.

## Pick the cheapest GPU
- RunPod **Community Cloud** (or **Spot**) **A100 80GB** ≈ **$0.8–1.2/hr** (much cheaper than Secure Cloud/H100).
  An A100 has enough cores/RAM for a fast parallel Triton build. Don't pay for H100 — not needed.
- Image: any recent **PyTorch / CUDA 12.x–13.x** template on **Ubuntu 22.04+** (Python 3.10–3.12). Phase 1
  uses stock Triton so the env is flexible; phase 2 (the build) just needs cmake/ninja/clang (present in
  the CUDA dev images, or `apt-get install -y cmake ninja-build clang`).

## Two modes — pick by budget
| Mode | What it gets | Time | ~Cost |
|---|---|---|---|
| **CHEAP** (default) | regret-law persists cross-arch — closes the single-GPU gap for the paper's core thesis | ~10–15 min | **~$0.3–0.4** |
| **FULL** (`CURE=1`) | + the built cure (4.15x) validated on A100 (builds Triton, MAX-parallel) | ~25–30 min | **~$0.5–1** |

Recommended: **CHEAP is enough** for the submission (the cure's cross-arch is deferrable to a revision if a
reviewer asks). Do FULL only if you want the flagship datacenter datapoint now (still <$1).

## Exact steps (copy-paste on the pod terminal)
```bash
# 1. get the repo (private → use a token or `gh auth`; or scp it up)
git clone https://github.com/alepot55/gpufsm.git && cd gpufsm      # or your remote
# 2. run (CHEAP):
bash scripts/a100_validate.sh
#    or FULL (adds the built-cure validation):
#    CURE=1 bash scripts/a100_validate.sh
# 3. commit the results (writes paper2/data/cross_arch/*_<gpu>.*):
git add paper2/data/cross_arch/ && git commit -m "data: A100 cross-arch validation" && git push
# 4. ⚠️ STOP THE POD IMMEDIATELY (biggest cost mistake = leaving it running).
```
Then ping me — I fold the numbers into the paper (results + Threats) and it's done.

## Cost-saving notes
- The script installs only what's needed, builds with `MAX_JOBS=$(nproc)` (fast on a big pod), and runs a
  MINIMAL experiment set (skips SpMV/MoE on the pod — already measured on the RTX 4070).
- Biggest saver: **terminate the pod the second the script prints DONE.** A forgotten pod for a day = ~$25.
- Advanced (skip the build entirely, saves ~10 min): the RTX-4070 pre-built cure `libtriton.so` (167 MB
  stripped) can be uploaded and reused since the compiler is host-side — ask me and I'll prep the tarball +
  `runpodctl send` steps (only worth it if the build is slow on your chosen pod).
