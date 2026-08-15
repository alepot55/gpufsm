# Draft della PR upstream (da rileggere prima di aprirla)

Titolo:

```
[MEMBAR] Treat entry into a warp_specialize default region as a sync point
```

Corpo (inglese, stile della repo):

---

`ttg.warp_specialize` lowers to two CTA-wide barriers between the capture stores and the branch into
the default region (`lowerWarpSpecialize`), so shared memory effects that happen before the op are
already synchronized when the default region starts. The membar analysis does not model this, so it
inserts a redundant `ttg.barrier` at the top of the default region whenever a shared memory access
precedes the op.

This teaches `getLocalBarrierStages` about that barrier. On the membar test corpus it removes N
redundant barriers and changes nothing else.

The rule is keyed on the op having no scratch buffer rather than on it having no captures. When the
op owns a scratch buffer, the capture stores into it happen *before* those barriers, so the
rendezvous that matters is the `betweenMemoryEffects` one the scratch path already models. The two
conditions are not interchangeable: backends override the scratch size, and the concurrency
sanitizer reserves capture bytes on every `warp_specialize` regardless of its captures, so keying on
`getCaptureSize()` would delete a barrier that is still required under `--instrumentation-mode=consan`.

The barriers are genuinely CTA-wide on both in-tree backends: the NVIDIA callback emits
`llvm.nvvm.barrier.cta.sync.all` and the AMD one a `rocdl.barrier`, and `lowerKernelBarriers` runs
before `lowerWarpSpecialize`, so they are not rewritten into warp-group named barriers.

Tests: the three existing cases that lose the redundant barrier are updated, and a new case pins both
directions of the rule (no captures -> barrier removed; captures -> barrier kept).

Left out on purpose: `ttg.warp_yield` also lowers to a CTA-wide barrier and modelling it removes one
more barrier, but it deserves its own tests and its own review, so it is not bundled here.

---

## Note operative

- **Destinatario**: `@Jokeren` (CODEOWNER di `lib/Analysis/Membar.cpp`). Assenso già registrato
  sull'idea da ThomasRaoux su #8374: *"yeah warp_specialization op can count as a bar sync"*.
- **Branch**: nuovo, dal `main` corrente, un solo file di codice + i test. Mai legare questa modifica
  ad altro (regola imparata da #11311 vs #10780).
- **CI**: outside contributor ⇒ la run resta in `action_required` finché un maintainer non approva.
  Fare **tutti** i push in una volta, poi congelare il branch.
- **Numeri**: sostituire `N` col delta misurato di v2 prima di aprire.
- **Da citare nel thread solo se chiesto**: la misura è `triton-opt -test-print-membar` sul corpus
  membar, non un benchmark. Il guadagno runtime non è stato misurato e la PR non lo rivendica.
