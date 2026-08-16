# Barriera di cluster mancante con TMA multicast su una vista (issue #11328)

**Stato:** [issue #11328](https://github.com/triton-lang/triton/issues/11328), aperta il 16 ago 2026.
Segnalazione e non PR: la correzione ovvia **rompe**, vedi sotto.

## Il difetto

`ClusterBarrierInsertion.cpp:328` usa `allocation.getBufferIds(value)`, che legge solo `valueBuffer`
(popolato sul risultato della `local_alloc`). Le **viste** (`ttg.memdesc_index`, ...) stanno in
`aliasBuffer` e si vedono solo con `getAllBufferIdsWithAliases`. Le altre **tre** chiamate nello stesso
file (`:84`, `:97`, `:163`) usano la versione con alias: la quarta no.

Conseguenza: una `async_tma_copy_global_to_local {multicast}` che scrive attraverso una vista non
finisce ne' in `syncWriteSlices` ne' in `syncReadSlices` → nessuna `ttng.cluster_barrier` prima che
quei byte vengano riusati. Essendo multicast, le scritture atterrano anche nella shared memory
dell'altro CTA.

## Verifica (eseguita)

`docs/upstream/repro_cluster_barrier_view.mlir`, due funzioni identiche tranne la destinazione:

| destinazione | `cluster_barrier` tra riuso e store |
|---|---|
| `ttg.local_alloc` diretta | **presente** |
| `ttg.memdesc_index` (vista) | **assente** |

Entrambi i buffer a `allocation.offset = 0` → stessi byte. L'argomento piu' forte e' l'incoerenza del
compilatore con se stesso: per lo stesso programma con destinazione diretta la barriera la mette lui.
La `wait_barrier` non copre il caso, e lo dicono i commenti in-tree
(`test/TritonNvidiaGPU/membar-cluster.mlir:684-689`, `:834`).

## Perche' la correzione da una parola NON basta

Cambiando `:328` in `getAllBufferIdsWithAliases`:
- il test nuovo passa e fallisce senza patch ✅
- ma la pipeline **aborta**: `LLVM ERROR: scratch buffer operations should not have any shared memory
  dependencies` su `@local_gather_subslice_other_cta` (`test/Conversion/tritonnvidiagpu_to_llvm.mlir`).

Seguendo gli alias, un'operazione puo' avere **sia** uno scratch buffer **sia** dipendenze shared
esplicite, cosa che il controllo a `:349-354` vieta. `Membar.cpp:293-301` ha lo stesso controllo ma
**con una deroga** per `LocalAtomicScatterRMWOp` ⇒ le due analisi sono divergenti.

Indovinare quali operazioni derogare dentro un'analisi di concorrenza non e' una scelta nostra: nella
issue sono offerte le due forme possibili. Albero Modal con il tentativo: `--tree ws5`.

## Lezione di metodo

Il fix "una parola, incoerenza evidente" sembrava il piu' sicuro dei tre trovati stamattina ed e'
l'unico che si e' rotto. La suite completa lo ha detto in un giro; l'insieme dei falliti confrontato
con la baseline e' cio' che lo ha reso visibile (un test in piu': `Conversion/tritonnvidiagpu_to_llvm.mlir`).
Vedi [[upstream-contribution-method]].
