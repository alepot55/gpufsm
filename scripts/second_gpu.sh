#!/usr/bin/env bash
# One entry point for every second-GPU run. Replaces second_gpu_{quick,rich,campaign}.sh,
# a100_validate.sh and run_cross_arch.sh, which differed only in which measurements they
# invoked while each carrying its own copy of the build step.
#
#   scripts/second_gpu.sh quick        # DFA knee only          (~10 min, ~$0.25 on an A100)
#   scripts/second_gpu.sh rich         # NFA 2x2 + DFA knee     (~15 min)
#   scripts/second_gpu.sh cross-arch   # the regret-law witnesses (p3_cross_arch)
#   scripts/second_gpu.sh campaign     # everything, into paper2/data/cross_arch/
#
# Throughputs are NOT comparable across GPUs. What these runs test is that the *shape*
# holds: Triton pays the regret and Warp does not, the CUDA DFA knee tracks L2.
set -euo pipefail

PROFILE="${1:-rich}"
PY="${PY:-python}"
command -v "$PY" >/dev/null || PY=python3

echo "== environment =="
"$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" || {
    echo "no CUDA device visible — nothing to do"; exit 0; }

# Build the extension for the local architecture only: building all four takes minutes of
# paid GPU time for cubins this machine will never load.
if "$PY" -c "import gpufsm.backends.cuda._cuda" 2>/dev/null; then
    echo "== CUDA extension already built =="
else
    ARCH="$("$PY" -c 'import torch;cc=torch.cuda.get_device_capability(0);print(f"{cc[0]}{cc[1]}-real")')"
    echo "== building the CUDA extension for sm_${ARCH%-real} =="
    "$PY" -m pip install -q -e . \
        --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON \
        --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCH"
fi
"$PY" -m pip install -q warp-lang 2>/dev/null || echo "  (warp-lang unavailable; Warp rows will be skipped)"

echo "== profile: $PROFILE =="
case "$PROFILE" in
    quick)
        "$PY" -m experiments.cure.validation.second_gpu --profile quick
        ;;
    rich)
        "$PY" -m experiments.cure.validation.second_gpu --profile rich
        ;;
    cross-arch)
        "$PY" -m experiments.cure.validation.p3_cross_arch
        ;;
    campaign)
        "$PY" -m experiments.cure.validation.second_gpu --profile rich
        "$PY" -m experiments.cure.validation.p3_cross_arch
        "$PY" -m experiments.cure.milestones.m0_anchor
        ;;
    *)
        echo "unknown profile '$PROFILE' (quick|rich|cross-arch|campaign)" >&2
        exit 2
        ;;
esac

echo
echo "Done. Results are in paper2/data/cross_arch/ — compare the SHAPE against the"
echo "RTX 4070 baseline, not the absolute throughputs."
