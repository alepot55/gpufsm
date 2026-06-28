# `thread_region` — the in-Triton compiler pass (P2)

This directory version-controls the gpufsm changes to the Triton compiler (which live in a separate
source tree, `~/m3full_build/triton-src`), so the landmark P2 contribution is preserved and
reproducible independent of that build tree.

**Status: DETECTION half DONE and VERIFIED inside `libtriton`.** A new TritonGPU pass
`tritongpu-thread-region` compiles into `libtriton.so`, runs in the NVIDIA `make_ttgir` pipeline
(env-gated), detects the lock-step irregular-region signature, tags each match with a
`ttg.thread_region_candidate` attribute, and is a clean no-op when disabled.
Verified by `experiments/cure/p2_pass_verify.py` (pass ON → attribute present; pass OFF → absent;
kernel still correct). The tile→thread *lowering* half is the next step (see `docs/P2_PASS_DESIGN.md`).

## Contents
- `ThreadRegion.cpp` — the pass (detection): walks the module, matches an `scf.while` whose iter-args
  are `#blocked` tile tensors and whose `scf.condition` derives from a `tt.reduce`-to-scalar (the
  lock-step gate = masked-lane waste / issue deficit made syntactic), marks + remarks each match.
- `registration.patch` — the four supporting edits: the pass def in `Passes.td`, the `CMakeLists.txt`
  source entry, the Python binding `add_thread_region` in `python/src/passes.cc`, and the env-gated
  insertion into `make_ttgir` in `third_party/nvidia/backend/compiler.py`.

## Reproduce
Base Triton commit: `c05aa65087a9a1a6b8a08fdbb474aba834d5cddf` (built locally as Triton 3.8.0).
Build recipe (the hard prerequisite): see `docs/P2_PASS_DESIGN.md` (cmake<4, nanobind==2.10.2,
python3.12-dev, direct `cmake --build`).

```bash
cd ~/m3full_build/triton-src                     # base commit above
git apply /path/to/registration.patch            # the 4 registration edits
cp /path/to/ThreadRegion.cpp lib/Dialect/TritonGPU/Transforms/ThreadRegion.cpp
cmake --build . -j 8                             # incremental: TableGen regen + relink (~minutes)
# verify:
PYTHONPATH=$HOME/m3full_build/triton-src/python .venv/bin/python experiments/cure/p2_pass_verify.py
# enable in any compile:
GPUFSM_THREAD_REGION=1 PYTHONPATH=... .venv/bin/python your_kernel.py
```
