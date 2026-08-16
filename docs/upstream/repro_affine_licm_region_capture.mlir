func.func @f(%m: memref<4xi64>, %init: i64, %outside: i64) -> i64 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c2 = arith.constant 2 : i32
  %one = arith.constant 1 : i64
  %r = affine.for %i = 0 to 4 iter_args(%acc = %init) -> (i64) {
    %v = affine.load %m[%i] : memref<4xi64>
    %s = scf.for %j = %c0 to %c4 step %c2 iter_args(%a = %outside) -> (i64) : i32 {
      %o = arith.ori %v, %one : i64
      %n = arith.addi %a, %o : i64
      scf.yield %n : i64
    }
    %acc2 = arith.addi %acc, %s : i64
    affine.yield %acc2 : i64
  }
  return %r : i64
}
