<!-- DRAFT PR #3 for github.com/triton-lang/triton (arm 1). User submits.
     Built in local Triton (v3.11.3): branch fold-ptr-roundtrip off upstream c05aa65.
     Reviewer: @lezcano. Mirrors the existing CanonicalizeIntToPtrOfPtrToInt. Independent of the other PRs. -->

# Canonicalize `ptr_to_int(int_to_ptr(x)) -> x` (complete the inverse pair)

## What

Triton already canonicalizes one direction of the pointer/integer round-trip —
`int_to_ptr(ptr_to_int(p)) -> p` (`CanonicalizeIntToPtrOfPtrToInt`) — but not the mirror. This adds
`ptr_to_int(int_to_ptr(x)) -> x`, so both inverse round-trips are eliminated.

## Why

Composed reinterpretations (`int_to_ptr` then `ptr_to_int`, e.g. when an integer address is materialized as
a pointer and immediately read back) currently survive `-canonicalize`. The existing one-directional fold
shows the round-trip elimination is intended; this completes the symmetry.

## How

`include/triton/Dialect/Triton/IR/TritonOps.td`: `let hasCanonicalizer = 1;` on `TT_PtrToIntOp`.
`lib/Dialect/Triton/IR/Ops.cpp`: `CanonicalizePtrToIntOfIntToPtr`, a direct mirror of the existing
`CanonicalizeIntToPtrOfPtrToInt` — it matches `ptr_to_int(int_to_ptr(val))` and replaces it with
`bitcast(val)` to the result type (exactly as the existing pattern does for the opposite direction). A
same-type bitcast then folds away via `BitcastOp::fold`, so the common int64->ptr->int64 case reduces to the
original value.

## Test

`test/Triton/canonicalize.mlir`: `ptr_to_int_of_int_to_ptr` asserts both casts disappear and the function
returns its input.

## Verification

Built `triton-opt` from this branch; `ptr_to_int(int_to_ptr(%x))` canonicalizes to `return %x`; FileCheck
passes on the full `test/Triton/canonicalize.mlir`. (Verified; branch fold-ptr-roundtrip @ 0541b42.)

## Regression safety (verified)
The only `ptr_to_int(int_to_ptr(...))` round-trip in `test/` is the new positive case here; `test/Triton/ops.mlir` uses independent casts and runs no `-canonicalize`, so this pattern cannot fire there → no CI regression.
