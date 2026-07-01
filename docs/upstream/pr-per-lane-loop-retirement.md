# [NVIDIA] Add opt-in per-lane loop retirement pass (`nv-per-lane-loop-retirement`)

RFC with full motivation, measurements, and the equivalence argument: https://github.com/triton-lang/triton/issues/10773

Triton lowers data-dependent while-loops (`while tl.max((j < trip).to(tl.int32)) > 0`)
to warp-lock-step form: the latch warp-reduces the per-lane predicate
(`nvvm.redux.sync max`) and branches on the uniform result, so every lane of a
warp iterates to the warp's **max** trip count and pays a warp-collective
reduction per iteration. On sm_70+ that schedule is a codegen choice, not a
hardware requirement.

This PR adds an **experimental, opt-in** module pass over the LLVM dialect
(`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/PerLaneLoopRetirement.cpp`)
that redirects such latches to the per-lane predicate — each lane retires
independently under independent thread scheduling — deletes the dead
cross-lane reduction, and reconverges the warp with `bar.warp.sync` at the
loop exit.

## Enablement

- `CUDAOptions.per_lane_loop_retirement: bool` (participates in the cache
  key; settable per kernel launch), default from the
  `TRITON_ENABLE_PERLANE_LOOP_RETIREMENT` env knob — same pattern as
  `ptx_options`. Gated on `capability >= 70`.

## Safety: a static verifier proves equivalence before rewriting

The loop is left untouched unless the verifier proves observational
equivalence with the lock-step schedule:

- latch is `cond_br(icmp sgt|ne (redux.sync max|umax|or|add
  (zext/select-normalized i1), full-warp-mask), 0)`;
- body free of NVVM ops (collectives, barriers, nested redux latches),
  calls, atomics, unpredicated `llvm.store`, and inline asm that stores or
  synchronizes (predicated gathers are safe);
- single loop exit; unconditionally-entered preheader (so the captured
  `activemask` names exactly the lanes that reach the reconvergence);
- every live-out loop-carried value **frozen** on lane-inactive iterations
  (`select(pred, x, old)` / masked-identity updates, followed through the
  struct-typed block-argument projections of the real lowering);
- the per-lane predicate is **monotone**: once false it cannot re-arm under
  continued lock-step execution (frozen/invariant operands, or a
  constant-step induction moving away from an ordered comparison). This
  condition is easy to miss -- live-out freezing alone is not sufficient --
  and the tests include the counterexample.

## Measurements (RTX 4070, sm_89; all runs oracle-checked bit-exact)

| | baseline | retired | |
|---|---|---|---|
| canonical divergent loop (geometric trips) | 100-167 us (clocks) | ~40 us | **2.5-4.2x** |
| issued instructions (Nsight) | 36.1M | 0.92M | **39x fewer** |
| PTX `redux.sync` | 1 | 0 | eliminated |
| SASS `REDUX.MAX.S32` | 1 | 0 | eliminated |

Across six controlled trip distributions the baseline follows
`t = 50.3 + 1.08 * E[warp-max trip]` us (R^2 = 0.998) while the retired
kernel is a flat ~43 us floor; the law holds on four held-out distributions
(mean err 5%). Real workloads with the same latch (power-law CSR SpMV, MoE
top-k routing) gain 1.14-1.25x — gather-bound, so less control to recover.

## Testing

- `test/Conversion/nv_per_lane_loop_retirement.mlir`: three positive cases
  (plain scalars, `add`-reduction latch, struct-carried real-lowering shape)
  and six negative cases (collective in body, **unfrozen live-out**,
  **non-monotone predicate**, min-redux, partial mask, side exit) — all
  FileCheck'd including the verifier remark.
- Full local lit suite passes.
- Standalone e2e bench/oracle script in the PR comment: compiles the kernel
  both ways via the launch option, asserts bit-exact output and PTX
  `redux.sync` 1 -> 0.

## Files

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/PerLaneLoopRetirement.cpp` (new)
- `third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Passes.td` (pass def)
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeLists.txt`
- `third_party/nvidia/triton_nvidia.cc` (python binding)
- `bin/RegisterTritonDialects.h` (triton-opt registration)
- `third_party/nvidia/backend/compiler.py` (options field + pipeline wiring)
- `python/triton/knobs.py` (env knob)
- `test/Conversion/nv_per_lane_loop_retirement.mlir` (lit tests)
