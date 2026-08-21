# The two halves of the cure, and why they had never coexisted

This directory carries a research pass in two pieces that were developed against different
upstream Triton commits. Getting them into one build on 21 Aug 2026 took three attempts, and
the reasons are worth writing down because none of them is obvious from the files.

| file | what it is |
|---|---|
| `ThreadRegion.cpp` | **two** passes: `TritonGPUThreadRegion` (detect the lock-step signature, plus a `hoist` mode that is the reduce-hoist rewrite) and `TritonGPULowerThreadRegionRetire` (an **older** LLVM-level retirement, superseded) |
| `ThreadRegion_detect_only.cpp` | the same file with the superseded second pass removed; this is what builds |
| `registration.patch` | declares `TritonGPUThreadRegion` in `Passes.td`, adds it to CMakeLists, binds it in `passes.cc`, wires it into `make_ttgir` |
| `perlane_retire_full.patch` | the **current** retirement, as `third_party/nvidia/.../PerLaneLoopRetirement.cpp`, wired into `make_llir` behind `per_lane_loop_retirement` |
| `pipeline_wiring.patch` | an earlier wiring of both, superseded by the two above |
| `Passes.td` | a full copy of upstream's `Passes.td` with **both** declarations; `registration.patch` adds only the first |

## The three traps

1. **`perlane_retire_full.patch` alone gives you no detection.** The wheel the paper's cure
   numbers come from carries only the retirement. Running a detection check against it
   silently reports "not detected" for a kernel that plainly has the signature.

2. **`git apply --3way` fails on the second patch, plain `git apply` works.** The retire
   patch is applied to the worktree without committing, so the index still matches HEAD and
   the three-way merge has no blob to work from. It fails with `does not match index` on all
   four shared files. Plain `git apply` matches on context and succeeds.

3. **`ThreadRegion.cpp` does not compile once `registration.patch` is applied**, with
   `expected template-name before '<'` at the `LowerThreadRegionRetirePass` declaration.
   The tablegen base class for that second pass does not exist, because `registration.patch`
   declares only `TritonGPUThreadRegion`. Remove the superseded pass, or add the second
   declaration from the local `Passes.td`. Removing is right: the retirement in the wheel is
   the newer implementation.

## Recipe that works

```bash
python scripts/modal_triton.py build --tree cure --ref 81a46fa \
    --patch experiments/cure/triton_thread_region_pass/perlane_retire_full.patch
python scripts/modal_triton.py run --cpu --tree cure \
    --upload experiments/cure/triton_thread_region_pass/registration.patch \
    --upload experiments/cure/triton_thread_region_pass/ThreadRegion_detect_only.cpp \
    --cmd 'cp -f /work/ThreadRegion_detect_only.cpp lib/Dialect/TritonGPU/Transforms/ThreadRegion.cpp' \
    --cmd 'git apply /work/registration.patch' \
    --cmd 'MAX_JOBS=32 CCACHE_DIR=/work/ccache TRITON_BUILD_WITH_CCACHE=true python -m pip install -e . --no-build-isolation'
```

The volume needs roughly 18 GB free for a tree plus its venv, and it fills silently: a full
volume surfaces as `No space left on device` in the middle of `pip install torch`, which
looks like a network failure and is not.
