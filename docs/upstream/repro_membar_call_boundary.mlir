#blockedCallSrc = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mmaCall = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#blockedCallDst = #ttg.dot_op<{opIdx = 0, parent = #mmaCall, kWidth = 2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // La callee comincia con una convert_layout: il suo PRIMO accesso allo scratch e' una SCRITTURA.
  tt.func private @callee_writes_first() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // Il chiamante lascia in sospeso una LETTURA dello stesso scratch, poi chiama.
  // WAR: servirebbe una barriera PRIMA della tt.call.
  tt.func @caller_pending_read_then_call() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %call = tt.call @callee_writes_first() : () -> tensor<128x32xf16, #blockedCallDst>
    %sum = arith.addf %call, %cvt : tensor<128x32xf16, #blockedCallDst>
    tt.return %sum : tensor<128x32xf16, #blockedCallDst>
  }
}
