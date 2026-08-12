#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @_perlane_while(%arg0: !llvm.ptr<1> {tt.pointee_type = i32}, %arg1: !llvm.ptr<1> {tt.pointee_type = i32}, %arg2: i32, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg1 : !llvm.ptr<1> to !tt.ptr<i32>
    %1 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<1> to !tt.ptr<i32>
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.bitcast %4 : i32 to i32
    %6 = llvm.mlir.undef : !llvm.struct<(i32)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(i32)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.bitcast %9 : i32 to i32
    %11 = llvm.mlir.undef : !llvm.struct<(i32)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(i32)> 
    %13 = nvvm.read.ptx.sreg.ctaid.x : i32
    %14 = llvm.mul %13, %2 : i32
    %15 = llvm.mlir.constant(0 : index) : i32
    %16 = nvvm.read.ptx.sreg.tid.x : i32
    %17 = llvm.mlir.constant(127 : i32) : i32
    %18 = llvm.and %16, %17 : i32
    %19 = llvm.mlir.constant(32 : i32) : i32
    %20 = llvm.urem %18, %19 : i32
    %21 = ttg.warp_id {omitUniformHint}
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.shl %20, %27 : i32
    %29 = llvm.or %26, %28 : i32
    %30 = llvm.mlir.constant(5 : i32) : i32
    %31 = llvm.shl %21, %30 : i32
    %32 = llvm.or %29, %31 : i32
    %33 = llvm.mlir.constant(31 : i32) : i32
    %34 = llvm.and %32, %33 : i32
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.lshr %34, %35 : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.or disjoint %36, %38 : i32
    %40 = llvm.xor %25, %39 : i32
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = llvm.xor %40, %41 : i32
    %43 = llvm.add %42, %15 : i32
    %44 = llvm.mlir.undef : !llvm.struct<(i32)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(i32)> 
    %46 = llvm.bitcast %14 : i32 to i32
    %47 = llvm.mlir.undef : !llvm.struct<(i32)>
    %48 = llvm.insertvalue %46, %47[0] : !llvm.struct<(i32)> 
    %49 = llvm.extractvalue %48[0] : !llvm.struct<(i32)> 
    %50 = llvm.extractvalue %45[0] : !llvm.struct<(i32)> 
    %51 = llvm.add %49, %50 : i32
    %52 = llvm.mlir.undef : !llvm.struct<(i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<(i32)> 
    %54 = llvm.bitcast %arg2 : i32 to i32
    %55 = llvm.mlir.undef : !llvm.struct<(i32)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(i32)> 
    %57 = llvm.extractvalue %53[0] : !llvm.struct<(i32)> 
    %58 = llvm.extractvalue %56[0] : !llvm.struct<(i32)> 
    %59 = llvm.icmp "slt" %57, %58 : i32
    %60 = llvm.mlir.undef : !llvm.struct<(i1)>
    %61 = llvm.insertvalue %59, %60[0] : !llvm.struct<(i1)> 
    %62 = llvm.bitcast %arg0 : !llvm.ptr<1> to !llvm.ptr<1>
    %63 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr<1>)> 
    %65 = llvm.extractvalue %64[0] : !llvm.struct<(ptr<1>)> 
    %66 = llvm.extractvalue %53[0] : !llvm.struct<(i32)> 
    %67 = llvm.getelementptr %65[%66] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
    %68 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.struct<(ptr<1>)> 
    %70 = llvm.extractvalue %69[0] : !llvm.struct<(ptr<1>)> 
    %71 = llvm.extractvalue %61[0] : !llvm.struct<(i1)> 
    %72 = llvm.extractvalue %7[0] : !llvm.struct<(i32)> 
    %73 = llvm.mlir.undef : vector<1xi32>
    %74 = llvm.mlir.constant(0 : index) : i32
    %75 = llvm.insertelement %72, %73[%74 : i32] : vector<1xi32>
    %76 = llvm.bitcast %75 : vector<1xi32> to i32
    %77 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %70, %71 : (!llvm.ptr<1>, i1) -> i32
    %78 = llvm.bitcast %77 : i32 to vector<1xi32>
    %79 = llvm.mlir.constant(0 : index) : i32
    %80 = llvm.extractelement %78[%79 : i32] : vector<1xi32>
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %80, %81[0] : !llvm.struct<(i32)> 
    llvm.br ^bb1(%7, %7 : !llvm.struct<(i32)>, !llvm.struct<(i32)>)
  ^bb1(%83: !llvm.struct<(i32)>, %84: !llvm.struct<(i32)>):  // 2 preds: ^bb0, ^bb2
    %85 = builtin.unrealized_conversion_cast %84 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %86 = builtin.unrealized_conversion_cast %83 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %87 = builtin.unrealized_conversion_cast %86 : tensor<32xi32, #blocked> to !llvm.struct<(i32)>
    %88 = builtin.unrealized_conversion_cast %85 : tensor<32xi32, #blocked> to !llvm.struct<(i32)>
    %89 = llvm.extractvalue %88[0] : !llvm.struct<(i32)> 
    %90 = llvm.extractvalue %82[0] : !llvm.struct<(i32)> 
    %91 = llvm.icmp "slt" %89, %90 : i32
    %92 = llvm.mlir.undef : !llvm.struct<(i1)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(i1)> 
    %94 = llvm.extractvalue %93[0] : !llvm.struct<(i1)> 
    %95 = llvm.zext %94 : i1 to i32
    %96 = llvm.mlir.undef : !llvm.struct<(i32)>
    %97 = llvm.insertvalue %95, %96[0] : !llvm.struct<(i32)> 
    %98 = llvm.extractvalue %97[0] : !llvm.struct<(i32)> 
    %99 = llvm.mlir.constant(-1 : i32) : i32
    llvm.cond_br %91, ^bb2(%83, %84 : !llvm.struct<(i32)>, !llvm.struct<(i32)>), ^bb3
  ^bb2(%100: !llvm.struct<(i32)>, %101: !llvm.struct<(i32)>):  // pred: ^bb1
    %102 = builtin.unrealized_conversion_cast %101 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %103 = builtin.unrealized_conversion_cast %100 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %104 = builtin.unrealized_conversion_cast %103 : tensor<32xi32, #blocked> to !llvm.struct<(i32)>
    %105 = builtin.unrealized_conversion_cast %102 : tensor<32xi32, #blocked> to !llvm.struct<(i32)>
    %106 = llvm.extractvalue %105[0] : !llvm.struct<(i32)> 
    %107 = llvm.extractvalue %82[0] : !llvm.struct<(i32)> 
    %108 = llvm.icmp "slt" %106, %107 : i32
    %109 = llvm.mlir.undef : !llvm.struct<(i1)>
    %110 = llvm.insertvalue %108, %109[0] : !llvm.struct<(i1)> 
    %111 = llvm.extractvalue %110[0] : !llvm.struct<(i1)> 
    %112 = llvm.extractvalue %105[0] : !llvm.struct<(i32)> 
    %113 = llvm.extractvalue %7[0] : !llvm.struct<(i32)> 
    %114 = llvm.select %111, %112, %113 : i1, i32
    %115 = llvm.mlir.undef : !llvm.struct<(i32)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(i32)> 
    %117 = llvm.extractvalue %104[0] : !llvm.struct<(i32)> 
    %118 = llvm.extractvalue %116[0] : !llvm.struct<(i32)> 
    %119 = llvm.add %117, %118 : i32
    %120 = llvm.mlir.undef : !llvm.struct<(i32)>
    %121 = llvm.insertvalue %119, %120[0] : !llvm.struct<(i32)> 
    %122 = builtin.unrealized_conversion_cast %121 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    %123 = llvm.extractvalue %105[0] : !llvm.struct<(i32)> 
    %124 = llvm.extractvalue %12[0] : !llvm.struct<(i32)> 
    %125 = llvm.add %123, %124 : i32
    %126 = llvm.mlir.undef : !llvm.struct<(i32)>
    %127 = llvm.insertvalue %125, %126[0] : !llvm.struct<(i32)> 
    %128 = builtin.unrealized_conversion_cast %127 : !llvm.struct<(i32)> to tensor<32xi32, #blocked>
    llvm.br ^bb1(%121, %127 : !llvm.struct<(i32)>, !llvm.struct<(i32)>)
  ^bb3:  // pred: ^bb1
    %129 = llvm.mlir.constant(-1 : i32) : i32
    nvvm.bar.warp.sync %129 : i32
    %130 = llvm.bitcast %arg1 : !llvm.ptr<1> to !llvm.ptr<1>
    %131 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %132 = llvm.insertvalue %130, %131[0] : !llvm.struct<(ptr<1>)> 
    %133 = llvm.extractvalue %132[0] : !llvm.struct<(ptr<1>)> 
    %134 = llvm.extractvalue %53[0] : !llvm.struct<(i32)> 
    %135 = llvm.getelementptr %133[%134] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
    %136 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %137 = llvm.insertvalue %135, %136[0] : !llvm.struct<(ptr<1>)> 
    %138 = llvm.extractvalue %137[0] : !llvm.struct<(ptr<1>)> 
    %139 = llvm.extractvalue %87[0] : !llvm.struct<(i32)> 
    %140 = llvm.extractvalue %61[0] : !llvm.struct<(i1)> 
    %141 = llvm.mlir.constant(0 : i32) : i32
    %142 = nvvm.read.ptx.sreg.tid.x : i32
    %143 = llvm.mlir.constant(127 : i32) : i32
    %144 = llvm.and %142, %143 : i32
    %145 = llvm.mlir.constant(32 : i32) : i32
    %146 = llvm.urem %144, %145 : i32
    %147 = ttg.warp_id {omitUniformHint}
    %148 = llvm.mlir.constant(3 : i32) : i32
    %149 = llvm.and %147, %148 : i32
    %150 = llvm.icmp "eq" %149, %141 : i32
    %151 = llvm.mlir.undef : vector<1xi32>
    %152 = llvm.bitcast %139 : i32 to i32
    %153 = llvm.mlir.constant(0 : i32) : i32
    %154 = llvm.insertelement %152, %151[%153 : i32] : vector<1xi32>
    %155 = llvm.bitcast %154 : vector<1xi32> to i32
    %156 = llvm.and %150, %140 : i1
    %157 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %155, %138, %156 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    llvm.return
  }
}

