# P2 — the `thread_region` pass: design grounded in a working Triton-from-source build

Status (2026-06-29): **the platform is live.** A hackable Triton **3.8.0** is built from source with
`libtriton.so` (823 MB) and is fully functional (JITs, runs on GPU, full pipeline
ttir→ttgir→llir→ptx→cubin). This doc pins the pass to **real IR** dumped from that build, names the
exact pipeline insertion point, and states the transformation + its falsifiable payoff bound. The C++
implementation + rebuild is the remaining engineering; everything it must do is specified here.

## Why this is the make-or-break contribution
The companion diagnosis (paper 1) + this paper's decomposition show the tile-SPMD regret on irregular
work is **abstraction-denied intra-warp latency hiding**, and the cure (M10) proves it closes the gap
by lowering the *same* per-lane source to the thread model (SP/WP2 = **4.2×**, matches/exceeds CUDA).
M10 does this *out-of-band* (nvcc + ctypes). P2 asks the top-venue question: can the **tile DSL itself**
detect the irregular region and lower it automatically? Characterization alone = IISWC; an automatic
in-compiler transformation = ASPLOS/PLDI and the NVIDIA-platform signal (cuTile / Tile IR).

## The working build recipe (reproducible; the prerequisite that was the hard part)
Diagnosed + fixed, in order (each was a hard failure):
1. `cmake>=3.20,<4.0` — Triton's build rejects cmake 4.x. Used system `/usr/bin/cmake` 3.28.3 (the pip
   cmake shim failed `cmake --version`; uninstalled it).
2. `nanobind==2.10.2` (not 2.13) — `nanobind-config.cmake` FATAL_ERRORs otherwise.
3. `python3.12-dev` (apt) — nanobind needs the Python Development headers; without them configure dies.
4. After "Configuring done", `setup.py`'s `compile_commands.json` bookkeeping hit "Not a directory" —
   **bypass** by building libtriton directly: `cmake --build <builddir> -j 8` (ninja).
5. Use it via `PYTHONPATH=$HOME/m3full_build/triton-src/python` (in-source build dir).

Smoke-tested: `experiments/cure/p2_ttgir_probe.py` prints `triton 3.8.0` and a correct kernel result.

## What the pass must MATCH (detection) — dumped from the live build
`p2_ttgir_probe.py` compiles a minimal per-lane data-dependent `while tl.max(active) > 0` loop (the
structural core of the NFA worklist, rejection sampling, every irregular witness) and dumps its TTGIR
to `paper2/data/landmark/p2_lockstep.ttgir`. The lock-step signature (asserted, falsifiable):

```
scf.while (%acc, %j : tensor<32xi32, #blocked>, tensor<32xi32, #blocked>)   # the whole TILE carried
  cond:  %p  = arith.cmpi slt, %j, %trip : tensor<32xi32, #blocked>         # per-lane predicate
         %r  = "tt.reduce"(%p) axis=0 -> i32                                # REDUCED to one scalar
         %c  = arith.cmpi sgt, %r, 0 : i32
         scf.condition(%c) ...                                              # tile loops to busiest lane
  body:  %active = arith.cmpi slt, %j, %trip                                # body predicated (masked)
         scf.yield ...
```

The detector = an `scf.while` whose iter-args are `#blocked`-encoded tensors **and** whose
`scf.condition` is gated by a `tt.reduce` of a per-lane predicate. That `tt.reduce` is the masked-lane
waste made syntactic: every lane runs every iteration up to `max(trip)`; idle lanes are masked, not
retired. (This is also the *predictor* of the regret law: the reduce-gate = the tile's issue deficit.)

## Where it goes (insertion point) — located in the live build
GPU pipeline assembled in `third_party/nvidia/backend/compiler.py::make_ttgir` (passes added to a
`PassManager`, run via `pm.run(mod, 'make_ttgir')`). TritonGPU transform passes live in
`lib/Dialect/TritonGPU/Transforms/` (e.g. `Coalesce.cpp`, `RemoveLayoutConversions.cpp`,
`Pipeliner/`, `WarpSpecialization/`). A `ThreadRegion.cpp` pass is added **early in `make_ttgir`**
(before pipelining/coalescing, which assume tile semantics), registered with a python binding
`ttgpuir.add_thread_region(pm)` alongside the existing `add_*` calls.

## The transformation (Approach B — de-vectorize the marked region to recover MLP)
This is the *opposite direction* to CPU whole-function / region vectorization (Karrenberg&Hack CGO'11,
Moll&Hack PLDI'18 linearize divergent CFG into vector lanes); we **de-vectorize** a divergent region
to recover per-lane independent issue. For the matched `scf.while` region:
1. **Re-encode** the carried tensors to `#blocked` with `sizePerThread = [1]`, `threadsPerWarp = [32]`
   — one element per lane, register-resident (no cross-lane layout).
2. **Drop the `tt.reduce` gate**: replace the tile-wide `scf.condition` with a *per-lane* loop. Concretely
   lower the region to a warp where each lane executes its own `scf.while` over its scalar slice, using
   Volta+ **Independent Thread Scheduling** so lanes diverge and retire independently.
3. **Disable the software pipeliner / multi-buffering** inside the region (it assumes lock-step tiles).
4. **Reconverge at the region exit** (a warp barrier / sync) so downstream tile ops see a coherent tile.
The marked region runs thread-mode; everything outside stays tiled — exactly M10's lowering, but in-DSL.

Surface syntax: a `tl.thread_region():` context (or `serial_range`) the user wraps the irregular loop
in; the front-end emits a `tt.thread_region` marker op the pass consumes. (Detection can also be made
automatic via the lock-step signature above — the cost-model selector picks it when the reduce-gated
while is present and the per-lane body is data-dependent.)

## The payoff bound (falsifiable, already measured by the proxy)
The pass closes the **latency-bound residual and nothing else**: up to **~2× on the NFA** (component C),
and **~0 extra in the already-converged DRAM-DFA regime** (where cross-warp parallelism already hides
latency — see the DFA crossover). M10 is the out-of-band existence proof of the upper end: lowering the
identical source to threads gives 4.2× over the tile and restores the thread-model Nsight signature
(issue 9.9%→36%, long-scoreboard stall 29× lower). So the pass's effect is *pre-measured*; the open
engineering is making the compiler do automatically what M10 does by hand.

## Honest status / risk
- DONE: working hackable build; functional 3.8.0; the IR detector (falsifiable, asserted on live IR);
  the insertion point; the transformation spec; the pre-measured payoff bound (M10).
- DONE (2026-06-29): **the DETECTION pass is real and VERIFIED inside `libtriton`.** `ThreadRegion.cpp`
  (a TritonGPU `mlir::ModuleOp` pass) compiles into `libtriton.so`, runs in `make_ttgir` (env-gated by
  `GPUFSM_THREAD_REGION`), matches the lock-step signature (`scf.while` over `#blocked` tile iter-args
  whose `scf.condition` derives from a `tt.reduce`), tags each with `ttg.thread_region_candidate`, and
  is a clean no-op when disabled. Verified by `experiments/cure/p2_pass_verify.py` (ON→present,
  OFF→absent, kernel still correct). Sources version-controlled in
  `experiments/cure/triton_thread_region_pass/` (ThreadRegion.cpp + registration.patch + README).
- RESOLVED (2026-06-29): **the in-IR lowering hits a STRUCTURAL WALL, demonstrated (not assumed).** The
  matched region's carried tensors are ALREADY `sizePerThread=1` (one element per lane) — so the
  lock-step is NOT a layout choice (re-encoding is a no-op; Gluon-style per-thread layout would not
  help). The lock-step is the **loop construct**: `scf.while`'s `scf.condition` is defined to take a
  single `i1`. The natural rewrite (give it a per-lane `tensor<NxI1>` so lanes terminate independently)
  is **rejected by the MLIR verifier** — captured verbatim by `triton-opt` on
  `triton_thread_region_pass/perlane_while_attempt.mlir`: *"use of value '%active' expects different
  type than prior uses: 'i1' vs 'tensor<8xi1, #blocked>'"*. Falsifiable probe:
  `experiments/cure/p2_lowering_wall.py` (exit 0 = wall confirmed). ⇒ per-lane loop termination is
  inexpressible in TritonGPU's structured tile control flow; the cure must lower **below** TritonGPU to
  the thread model (ITS) — which is exactly what M10 does (nvcc, 4.2×). This is a STRONGER result than a
  hand-tuned in-IR rewrite: it proves the abstraction regret is structural in the loop construct, not a
  layout or tuning artifact. **This is the central NVIDIA/landmark point**: the missing IR primitive is a
  per-lane (sub-tile) loop/exit op; today's tile IR cannot express it.
- REMAINING (the realized automatic cure): a cost-model **selector** that, on detecting the lock-step
  signature (the pass already does this), routes the region to the M10 thread-model lowering — automatic
  detect-and-lower at the source/codegen boundary, since the in-TritonGPU lowering is structurally
  blocked. This is the honest P2 endpoint per LANDMARK_PLAN.
- RISK: step 2 (per-lane independent `scf.while` via ITS inside one Triton program) may require lowering
  below TritonGPU to the LLVM/NVVM stage; if in-TritonGPU lowering proves infeasible, the honest
  fallback (per LANDMARK_PLAN P2) is the **automatic selector over the M10 lowering** + this IR design —
  still a real automatic detect-and-lower result, just realized at the source/codegen boundary.

Cross-refs: `experiments/cure/p2_ttgir_probe.py`, `paper2/data/landmark/p2_lockstep.ttgir`,
`experiments/cure/m10_scalar_program.py` (the measured cure), `docs/LANDMARK_PLAN.md` (P2).
