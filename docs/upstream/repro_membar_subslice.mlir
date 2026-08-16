// CASO: subslice attraverso un reinterpret che cambia il sistema di coordinate.
// Le due zone coprono gli STESSI byte [512,1024) di un buffer da 1024 B.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @reinterpret_then_subslice
tt.func public @reinterpret_then_subslice(%x: tensor<16x16xf16>) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
  %v1 = ttg.memdesc_subslice %alloc[16, 0] : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16>
  ttg.local_store %x, %v1 : tensor<16x16xf16> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16>
  %r = ttg.memdesc_reinterpret %alloc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
  %v2 = ttg.memdesc_subslice %r[8, 0] : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x32xf16, #shared, #smem, mutable, 16x32>
  %y = ttg.local_load %v2 : !ttg.memdesc<8x32xf16, #shared, #smem, mutable, 16x32> -> tensor<8x32xf16>
  tt.return
}

// -----
// CONTROLLO NEGATIVO: stessi byte, stesso buffer, NESSUN reinterpret.
// Qui l'analisi deve mettere la barriera.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @control_no_reinterpret
tt.func public @control_no_reinterpret(%x: tensor<16x16xf16>) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
  %v1 = ttg.memdesc_subslice %alloc[16, 0] : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16>
  ttg.local_store %x, %v1 : tensor<16x16xf16> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16>
  %v2 = ttg.memdesc_subslice %alloc[16, 0] : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16>
  %y = ttg.local_load %v2 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 32x16> -> tensor<16x16xf16>
  tt.return
}
