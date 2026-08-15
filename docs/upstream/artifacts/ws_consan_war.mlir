#layout = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90"} {

// The buffer dies before the op, so the allocator is free to place the
// sanitizer's capture scratch on the same bytes. The read of %0 must be ordered
// against the capture stores that the lowering emits into that scratch, and
// those stores happen before the entry barriers.
tt.func @ws_consan_war(%arg0: tensor<1xi64>) {
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  ttg.local_store %arg0, %0 : tensor<1xi64> -> !ttg.memdesc<1xi64, #layout, #smem, mutable>
  ttg.local_load %0 : !ttg.memdesc<1xi64, #layout, #smem, mutable> -> tensor<1xi64>
  ttg.warp_specialize() attributes {"consan.extra_capture_bytes" = 256 : i32}
  default {
    ttg.warp_yield
  } : () -> ()
  tt.return
}

}
