# Building the cure — per-lane loop retirement BELOW TritonGPU (flagship, multi-week)

Goal: make lanes of a tile whose data-dependent loop has finished STOP issuing work (CUDA-ITS-style
per-lane retirement), instead of the current masked lock-step (tile issues the union of all lanes' paths).
This is the ~3.6× residual cure (bounded: full = 5.64×, in-IR reduce-hoist already captures 1.55×). Attacks
weakness #2 ("the cure is unbuilt"). Local-doable on RTX 4070; datacenter only for camera-ready validation.

## Code-grounded map (from deep recon of triton-src v3.11.3, HEAD c05aa65)

**Key correction:** the per-lane scalar does NOT exist at scf→cf time (the tile is one opaque SSA value there).
It materializes only INSIDE `add_to_llvmir` when the distributed struct is unpacked. ⇒ the rewrite is a NEW
LLVM-dialect pass inserted AFTER `add_to_llvmir`.

### Pipeline (`third_party/nvidia/backend/compiler.py`, make_llir)
- `add_scf_to_cf` (line 384): scf.while → cf.cond_br; tile still `tensor<…,#blocked>` as a cf block arg;
  condition is a single i1 from `tt.reduce(or of cmpi slt %j,%trip)`.
- `add_to_llvmir` (line 396) = ConvertTritonGPUToLLVM (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/
  TritonGPUToLLVM.cpp:84`): tensors→`LLVM::StructType`; tt.reduce→warp shuffles (`NVVM::ShflOp`); a 2nd
  partial conversion (:224-233) cf→llvm.cond_br.
- **INSERT NEW PASS at line 397** (after add_to_llvmir, before add_initialize_ws_cluster_barriers).

### Primitives to use
- Unpack per-lane scalar: `unpackLLElements(loc, struct, rewriter)` (`lib/Conversion/TritonGPUToLLVM/
  Utility.cpp:1027`) → for sizePerThread=1, `[0]` is this lane's element (= trip[lane]).
- Per-lane branch: `LLVM::ICmpOp::create(b, loc, slt, jLane, tripLane)` → rewrite the `llvm.cond_br`.
- Reconverge: `createSyncWarp(loc, builder)` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp:130`
  → `NVVM::SyncWarpOp` with mask 0xffffffff). No `llvm.experimental.convergence.*` in tree; Triton uses NVVM
  sync intrinsics + structural reconvergence.
- API style this tree MANDATES: `OpTy::create(builder, loc, ...)` (NOT builder.create<OpTy>); deprecation=error.

## Milestones (each ends build-verified; ~50min rebuild per .td/pipeline change, fast relink for .cpp-only)

- **M0 — marker propagation.** Add a `retire` mode to `ThreadRegion.cpp` that (before scf_to_cf) stamps the
  matched while's gating `tt.reduce` with an attr that survives into add_to_llvmir (and have the reduce
  lowering copy it onto the emitted ShflOp). VERIFY: `triton-opt … | grep` the attr on the reduce. (small,
  in a known file)
- **M1 — new LLVM-dialect pass `LowerThreadRegion` (gate GPUFSM_THREAD_REGION=retire), inserted at
  compiler.py:397.** On each `LLVM::LLVMFuncOp`: find the `llvm.cond_br` whose condition reaches the marked
  `NVVM::ShflOp` warp reduction; extract `tripLane = unpackLLElements(tripStruct)[0]` and `jLane`; build
  `condL = icmp slt jLane, tripLane`; rewrite the cond_br to use condL; delete the now-dead ShflOp reduce;
  insert `createSyncWarp` at the loop's single structural exit. VERIFY: lowered PTX/SASS shows a per-lane
  `@p bra` + `bar.warp.sync` (not a uniform reduce-gated loop); oracle-correct output on the f3 kernel.
- **M2 — structural guards (correctness).** BAIL OUT of the rewrite if the loop body contains any cross-lane
  op (`NVVM::ShflOp`/`NVVM::Barrier`/shared `llvm.store`/`tt.dot`) — those need all lanes converged; or if
  the loop is software-pipelined (`cp.async`/num_stages>1). Assert single-exit before rewriting.
- **M3 — end-to-end measurement.** Run the f3_hoist lock-step kernel through the new path; confirm
  oracle-correct AND measure speedup vs masked baseline (target: approach the 5.64× thread bound; the
  reduce-hoist alone was 1.55×). Nsight: threads-per-instruction should drop <32 (per-lane retirement).
- **M4 — generalize + paper.** Show it fires on ≥2 irregular witnesses (automata + one more); fold the
  built-cure result into paper2 (turns "diagnosed" → "built", the flagship contribution); then the construct
  framing (`tt.scalar_region`) + cost-model for the top-venue version.

## Risks (the research flagged these as the parts that could sink it)
- Coherent SSA for other tile values live across the loop (accumulators): safe iff each lane writes only its
  own struct register AND no cross-lane op in the body → enforced by M2 guards.
- Pipeliner / async-copy assume uniform trip counts → M2 excludes pipelined loops.
- `bar.warp.sync(0xffffffff)` at exit needs ALL 32 lanes to reach it → assert single-exit / no early
  function return inside the loop.

Provenance of this map: deep code recon 2026-06-30 (agent), every claim file:line-cited above.
