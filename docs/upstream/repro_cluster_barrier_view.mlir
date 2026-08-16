#blockedTmaSrc = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedTmaDst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#nvmmaTma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEncTma = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CONTROLLO: destinazione = allocazione diretta (come il test gia' in-tree).
  tt.func @control_direct(%input: tensor<64x128xf16, #blockedTmaSrc>, %desc: !tt.tensordesc<64x128xf16, #nvmmaTma>) -> tensor<64x128xf16, #blockedTmaDst> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    %cvt = ttg.convert_layout %input : tensor<64x128xf16, #blockedTmaSrc> -> tensor<64x128xf16, #blockedTmaDst>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmmaTma>, !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %dst : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    tt.return %cvt : tensor<64x128xf16, #blockedTmaDst>
  }

  // CASO: identico, ma la destinazione e' una VISTA (memdesc_index) di un multi-buffer.
  tt.func @case_indexed_view(%input: tensor<64x128xf16, #blockedTmaSrc>, %desc: !tt.tensordesc<64x128xf16, #nvmmaTma>) -> tensor<64x128xf16, #blockedTmaDst> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    %cvt = ttg.convert_layout %input : tensor<64x128xf16, #blockedTmaSrc> -> tensor<64x128xf16, #blockedTmaDst>
    %bufs = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable>
    %dst = ttg.memdesc_index %bufs[%c0] : !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmmaTma>, !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %bufs : !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    tt.return %cvt : tensor<64x128xf16, #blockedTmaDst>
  }

  // CONTROLLO-RIUSO: destinazione diretta, poi il buffer viene riusato.
  tt.func @control_direct_reuse(%input: tensor<64x128xf16, #blockedTmaSrc>, %desc: !tt.tensordesc<64x128xf16, #nvmmaTma>, %v: tensor<64x128xf16, #blockedTmaDst>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmmaTma>, !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %dst : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    %reuse = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_store %v, %reuse : tensor<64x128xf16, #blockedTmaDst> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %reuse : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    tt.return
  }

  // CASO-RIUSO: destinazione = vista indicizzata, poi il buffer viene riusato.
  tt.func @case_indexed_view_reuse(%input: tensor<64x128xf16, #blockedTmaSrc>, %desc: !tt.tensordesc<64x128xf16, #nvmmaTma>, %v: tensor<64x128xf16, #blockedTmaDst>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    %bufs = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable>
    %dst = ttg.memdesc_index %bufs[%c0] : !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmmaTma>, !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %bufs : !ttg.memdesc<1x64x128xf16, #nvmmaTma, #smem, mutable>
    %reuse = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_store %v, %reuse : tensor<64x128xf16, #blockedTmaDst> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %reuse : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    tt.return
  }
}
