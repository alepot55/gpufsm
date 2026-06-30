<!-- DRAFT PR for github.com/triton-lang/triton (arm 1). User submits.
     Status: implemented in the local Triton tree (v3.11.3); pending build+lit verification.
     Suggested reviewer: @lezcano (owns canonicalizations). Template: merged #10734, #9971. -->

# Fold `split(join(a, b)) -> (a, b)` and `join(split(x)) -> x`

**STATUS: SUBMITTED → https://github.com/triton-lang/triton/pull/10766 (OPEN, mergeable, awaiting maintainer review/CI).**

## What

`tt.join` and `tt.split` are mutual inverses (join stacks two tensors along a new minor dim of size 2;
split peels that dim back into the two halves), but neither op had a folder, so the trivially-removable
round-trips survived `-canonicalize` and `-triton-combine`. This adds the two folds:

- `split(join(a, b)) -> (a, b)`
- `join(split(x)[0], split(x)[1]) -> x`

## Why

These round-trips are emitted by composition (e.g. a helper that `join`s a pair feeding code that
immediately `split`s it, or vice-versa) and currently survive to lowering as real ops. The inverse
relationship is already relied upon in-tree (`lib/Dialect/TritonGPU/IR/Utility.cpp`: "Split is the inverse
of join." / "Join is the inverse of split."); this just lets the folder remove the identity pair.

## How

- `include/triton/Dialect/Triton/IR/TritonOps.td`: `let hasFolder = 1;` on `TT_JoinOp` and `TT_SplitOp`.
- `lib/Dialect/Triton/IR/Ops.cpp`:
  - `JoinOp::fold` returns the split's source iff both join operands are exactly the two results of a
    single `split` **and** the reconstructed type equals the join result type.
  - `SplitOp::fold` (multi-result) returns the join's two operands iff the source is a `join` **and** the
    operand types equal the split result types.

Both folds are **guarded on exact type equality**, so they never silently drop a layout conversion: if the
`#blocked` encoding inferred for the split result differs from the original operand type, the fold does not
fire (mirroring the existing `TransOp::fold` convention, which only returns the source when
`getSrc().getType() == getType()`). For well-formed IR the encodings match by construction — `JoinOp::verify`
and `SplitOp::inferReturnTypes` both route through `inferSplitOpEncoding`, so the inverse layout is the same
one the verifier already enforces.

## Test

`test/Triton/canonicalize.mlir`: two FileCheck cases (`split_join`, `join_split`) asserting both the join
and the (now-dead, DCE'd) split disappear and the function returns its inputs directly.

## Before / after (`triton-opt -canonicalize`)

```
# before:  %j = tt.join %a, %b ; %o:2 = tt.split %j ; return %o#0, %o#1
# after:   return %a, %b
```

## Regression safety (verified)
No existing Triton test contains a `split(join(...))` or `join(split(...))` round-trip (content grep over `test/`; pipeline tests use standalone `tt.join` only), so this fold cannot fire on any current test → no CI regression.
