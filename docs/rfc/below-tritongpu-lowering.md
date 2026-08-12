# Below-TritonGPU per-lane lowering — design, bounded payoff, and the exact blocker

The structural wall (P2): per-lane loop termination is inexpressible in TritonGPU's structured tile IR
(`scf.condition` takes a single `i1`). The built reduce-hoist pass (`GPUFSM_THREAD_REGION=hoist`) trims
the lock-step loop's per-iteration cross-lane reduce (1.55x) but, by construction, keeps the tile
lock-step (threads-per-instruction stays 32 — no lane retires early). The FULL cure — per-lane
(sub-warp) retirement — must be lowered BELOW TritonGPU. This note pins the hook-point, the op
semantics, the reconvergence, and the exact blocker, and bounds the payoff with measurement.

## Bounded payoff (measured, this RTX 4070; lock-step kernel acc[i]=sum_{j<trip[i]} j, power-law trips)
| variant | time | speedup | threads-per-instruction |
|---|---|---|---|
| Triton tile (lock-step `while tl.max`) | 155.6 us | 1.00x | 32 (no retirement) |
| reduce-hoist (built in-compiler pass)  | 100.4 us | 1.55x | 32 (no retirement) |
| **CUDA thread (one/element, retiring)**| **27.6 us** | **5.64x** | **11.65 (per-lane retirement)** |

So on this kernel the full cure is worth ~5.6x; the in-IR reduce-hoist captures 1.55x; the remaining
~3.6x is **exactly** the per-lane retirement (threads-per-instruction 32 -> 11.65) that the wall blocks
in TritonGPU. (Repro: experiments/cure/f3_full_cure_bound.py + f3_hoist_verify.py;
paper2/data/landmark/f3_hoist_rtx4070.csv.)

## The exact hook-point (pipeline location)
NVIDIA backend (third_party/nvidia/backend/compiler.py):
- `make_ttgir` (line ~261): the tile-level pipeline; this is where `tritongpu-thread-region` runs
  (detect + reduce-hoist). The per-lane lowering can NOT live here — the tile IR can't express it.
- `make_llir` (line ~370): `passes.convert.add_scf_to_cf(pm)` (~384) lowers SCF -> CF, then
  `nvidia.passes.ttgpuir.add_to_llvmir(pm, ...)` (~396) lowers TritonGPU+CF -> LLVM/NVVM. **This is the
  hook-point.** A per-lane `tt.scalar_region` / `serial_range` op must be lowered by a custom conversion
  pattern in TritonGPUToLLVM (third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/), BEFORE/around `add_scf_to_cf`,
  emitting a per-lane LLVM loop rather than a uniform tile loop.

## Op semantics at the LLVM level (what the lowering emits)
A `tt.scalar_region` marks a region where a `#blocked` tile with `sizePerThread=1, threadsPerWarp=32`
is viewed as 32 independent per-lane scalars. The lowering:
1. **Extract the per-lane scalar.** Each lane already physically holds its tensor element (the blocked
   layout maps element i -> lane i). Lower the carried tile values to per-lane LLVM scalars (the
   distributed representation already gives this; the layout conversion is the subtlety).
2. **Per-lane CFG.** Emit a standard LLVM loop with `cf.cond_br` / `llvm.cond_br` on the **per-lane** i1
   (`%j_lane < %trip_lane`). Because the branch value differs per thread, the GPU SIMT engine diverges
   per lane; on Volta+ Independent Thread Scheduling, finished lanes retire while others continue —
   precisely the thread-model behavior (threads-per-instruction drops below 32).
3. **Carried state in registers.** The per-lane loop body updates each lane's scalar accumulator in
   registers (no cross-lane op inside the region).
4. **Reconverge at the exit.** Insert a `bar.warp.sync` (NVVM `nvvm.bar.warp.sync` / `__syncwarp`) at
   the region boundary so downstream tile ops see a coherent, reconverged tile again.

## The exact blocker (why this is multi-week, not a one-iteration pass)
1. **A new op + TableGen + verifier** (`ttg.scalar_region` / `tt.serial_range`) with region semantics
   and the "no cross-lane ops inside" constraint.
2. **The crux — per-lane scalar extraction from the distributed layout.** Lowering a carried
   `tensor<32xi32, #blocked>` to a per-lane LLVM scalar AND keeping the tile state coherent across a
   *divergent* loop is the hard part: the existing TritonGPUToLLVM patterns assume uniform control flow
   over the tile; a divergent per-lane loop breaks invariants the layout/codegen rely on (e.g. masked
   vectorized loads, the value/struct packing of distributed tensors).
3. **Pipeliner / scheduling interaction.** The software pipeliner and multibuffering assume lock-step
   tiles; the region must opt out and not be re-tiled by later passes.
4. **Reconvergence correctness** across the whole warp (and block) at the region exit, and correct
   interaction with shared-memory/async ops outside the region.
5. **Testing**: oracle-gated end-to-end + IR-level (triton-opt) + Nsight (threads-per-instruction must
   drop below 32, matching the thread target).

Each is a real chunk; (2) is the genuine multi-week risk. The reduce-hoist (built) is the safe in-IR
slice; this note is the concrete plan + bound for the rest, and is exactly the design the RFC's
`tt.scalar_region` proposal needs. Honest status: NOT implemented; specified + bounded + hook-point
pinned.
