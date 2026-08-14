# Commenti pronti da incollare sulle PR upstream (2026-08-14)

Entrambi i branch sono già stati aggiornati su `main` e pushati su `alepot55/triton`,
quindi la CI sta girando sulla base di oggi. Manca solo il commento (questa sessione
non può scrivere su `triton-lang/triton`).

---

## PR #10766 — https://github.com/triton-lang/triton/pull/10766

Refreshed onto current `main` so CI runs against today's base.

Re-checked the folds against #10749, which landed since: both guard on exact type
equality, which is strictly stricter than the new `ignoreRegBroadcast` relaxation in
`JoinOp::verify` / `SplitOp::isCompatibleReturnTypes`, so they stay conservative — a
round-trip whose halves differ only by register broadcasting simply does not fold.

@ThomasRaoux @peterbell10 this has been unchanged and green since 2 Jul. Happy to
close it if you'd rather not carry the extra folds — just don't want it sitting in
the queue.

---

## PR #10780 — https://github.com/triton-lang/triton/pull/10780

Refreshed onto current `main`, plus one more fix in the same file.

The example 3 python block does not parse today:

```
IndentationError: unindent does not match any outer indentation level
```

`dump_stages_hook` and `override_stages` indent their early-return guard at two
spaces while the rest of the body sits at four. Reindented those five lines;
extracting every ` ```python ` block from the README and running `compile()` over it
now succeeds for all four blocks — example 3 was the only one failing before.

@CRobeck the static/dynamic split we settled on is unchanged: examples 1–2 precompute
`PLUGIN_KEY`/`PLUGIN_HASH`, while example 3's `override_stages` re-derives its key
from `compiler_override.py`'s current content on every call.
