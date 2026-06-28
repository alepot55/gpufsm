// LANDMARK P2 — the lowering wall, made falsifiable.
//
// The thread_region transform wants each LANE to terminate its loop independently (recovering the
// thread model's per-lane control / intra-warp latency hiding). The natural in-IR rewrite of the
// matched lock-step `scf.while` is: give `scf.condition` a PER-LANE `tensor<...xi1>` predicate
// instead of the `tt.reduce`-to-scalar `i1`. This file is exactly that rewrite.
//
// Run it through the built `triton-opt` (see p2_lowering_wall.py). MLIR REJECTS it:
//   error: use of value '%active' expects different type than prior uses:
//          'i1' vs 'tensor<8xi1, #ttg.blocked<...>>'
// because `scf.condition` is defined to take a single `i1`. So per-lane loop termination is
// structurally inexpressible over a tile tensor in TritonGPU's structured control flow: the only
// loop-over-a-tensor is uniform (runs to the reduction of the per-lane predicate). The abstraction
// regret is in the LOOP CONSTRUCT, not the layout (the carried tensors are already sizePerThread=1).
// The cure must therefore lower BELOW TritonGPU to the thread model (ITS) — which is what M10 does.
//
// Falsifiable: if a future Triton/MLIR accepts a per-lane scf.condition (or adds a per-lane loop op),
// this stops erroring and the "structural wall" claim must be revisited.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @perlane(%trip: tensor<8xi32, #blocked>) {
    %c0 = arith.constant dense<0> : tensor<8xi32, #blocked>
    %c1 = arith.constant dense<1> : tensor<8xi32, #blocked>
    %r:2 = scf.while (%acc = %c0, %j = %c0) : (tensor<8xi32, #blocked>, tensor<8xi32, #blocked>) -> (tensor<8xi32, #blocked>, tensor<8xi32, #blocked>) {
      %active = arith.cmpi slt, %j, %trip : tensor<8xi32, #blocked>
      // per-lane condition (NOT reduced to a scalar i1) -- this is what MLIR rejects:
      scf.condition(%active) %acc, %j : tensor<8xi32, #blocked>, tensor<8xi32, #blocked>
    } do {
    ^bb0(%acc: tensor<8xi32, #blocked>, %j: tensor<8xi32, #blocked>):
      %jn = arith.addi %j, %c1 : tensor<8xi32, #blocked>
      scf.yield %acc, %jn : tensor<8xi32, #blocked>, tensor<8xi32, #blocked>
    }
    tt.return
  }
}
