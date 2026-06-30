<!-- DRAFT PR #2 for github.com/triton-lang/triton (arm 1). User submits.
     Built + verified in local Triton (v3.11.3): branch fold-bitcast @ b68445d off upstream c05aa65.
     Reviewer: @lezcano. Template: merged #10734, #9971. Independent of the split/join PR. -->

# Fold nested bitcasts: `bitcast(bitcast(x)) -> bitcast(x)`

## What

`BitcastOp::fold` only folded the same-type identity (`bitcast(x : T) : T -> x`). It did not look through
a nested bitcast, so `bitcast(bitcast(x))` survived `-canonicalize`. This extends the folder:

- round-trip `bitcast(bitcast(x : A->B) : B->A) -> x`
- chain `bitcast(bitcast(x : A->B) : B->C) -> bitcast(x : A->C)` (single cast)

## Why

Bitcast chains arise from composed reinterpret casts in mixed-dtype / type-punning kernels (e.g. a helper
that reinterprets through an intermediate type). They are pure no-ops at the bit level and should collapse.

## How

`lib/Dialect/Triton/IR/Ops.cpp`, `BitcastOp::fold`: if the source is itself a `bitcast`, either return its
source (when the round-trip returns to the original type — the identity) or rewrite in place to cast the
original source straight to this type. Safe because `tt.bitcast` carries `SameOperandsAndResultShape`,
`SameOperandsAndResultEncoding`, and equal bitwidth, so all three types in the chain share shape, encoding,
and bitwidth — the collapsed single cast is well-formed. Mirrors the existing `trans(trans(x))` in-place
fold idiom.

## Test

`test/Triton/canonicalize.mlir`: `bitcast_roundtrip` (i32->f32->i32 folds to the input) and `bitcast_chain`
(i16->f16->bf16 collapses to a single i16->bf16 cast).

## Verification

Built `triton-opt` from this branch; direct repro folds `bitcast(bitcast(x))` to `return %x`; FileCheck
passes on the full `test/Triton/canonicalize.mlir`.
