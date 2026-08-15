# Contributo upstream in preparazione: `warp_specialize` come punto di sincronizzazione in membar

Stato: **patch v2 in verifica, 2026-08-15.** Sostituisce il bersaglio precedente
(`WaitBarrier ↔ TMA`, scartato — vedi `MEMBAR_CANSKIPBARSYNC.md` e la sezione "perché no" qui sotto).

## Il buco

`lowerWarpSpecialize` (`lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp`) emette **due
barriere CTA-wide** tra le store delle capture e il salto dentro la default region, e una a ogni
`warp_yield`. Il callback NVIDIA è `llvm.nvvm.barrier.cta.sync.all`, quello AMD `ROCDL::BarrierOp`:
barriere di CTA vere, non named barrier di warp group (verificato: `lowerKernelBarriers` gira
**prima** di `lowerWarpSpecialize`, quindi non le retrocede).

La membar analysis non le sa. Risultato: inserisce una `ttg.barrier local` ridondante in cima alla
default region ogni volta che un accesso a shared memory precede la `warp_specialize`.

## La patch (v2)

In `getLocalBarrierStages`, **non** in `containsLocalBarrier`:

```cpp
if (isa<triton::gpu::WarpSpecializeOp>(op)) {
  stages.beforeMemoryEffects = !hasScratchBarrier;
  return stages;
}
```

## Perché v1 era sbagliata (trovato dalla review avversariale, non dal test)

v1 stava in `containsLocalBarrier` con il gate `wsOp.getCaptureSize() == 0`. Due difetti, entrambi
invisibili ai lit test:

1. **Il gate misura la quantità sbagliata.** La precondizione vera è *"quest'op possiede uno scratch
   buffer?"*, che è una proprietà dell'**allocazione**, non dell'op. `PrepareConSanCaptures` riserva
   byte di capture su **ogni** `ttg.warp_specialize`, anche senza capture, e i size-fn dei backend li
   sommano allo scratch. Quindi sotto **ConSan** un'op con `getCaptureSize() == 0` ha comunque uno
   scratch buffer, le store delle capture precedono le barriere, e v1 cancellava una barriera
   necessaria. I lit test non lo vedono: `triton-opt -test-print-membar` usa lo scratch-size fn di
   default, cioè l'unica configurazione in cui i due predicati coincidono.
2. **`containsLocalBarrier` ha un secondo consumatore**: `TMemBarrierInsertion.cpp:247`, che governa
   la **tensor memory**, non la shared. Là un `true` fa `sync(); return;` saltando la contabilità
   TMEM — cambio di comportamento non testato (`tmem_barrier_insertion.mlir` non contiene
   `warp_specialize`) e per giunta giustificato da un argomento che parla d'altro.

Il gate corretto è `!hasScratchBarrier`, che l'analisi calcola già tre righe sopra.

## Cosa è stato lasciato fuori di proposito

La regola su `WarpYieldOp` (ogni yield è anch'esso una barriera CTA-wide). Ha retto all'attacco —
il conteggio degli arrivi lo dimostra: su ogni path ciascuna warp esegue esattamente tre barriere
CTA-wide, e l'accoppiamento `warp_return ↔ yield` è forzato — ma vale una barriera sola e richiede
test in più. Si propone dopo, separata: **mai legare un pezzo non controverso a uno che richiede
discussione** (regola imparata da #11311 vs #10780).

## Numeri

Corpus `triton-opt -test-print-membar`, barriere emesse:

| file | baseline | v1 (WS + yield) | v2 (solo WS) |
|------|---------:|----------------:|-------------:|
| `test/Analysis/test-membar.mlir` | 127 | 123 | da misurare |
| `test/Analysis/test-membar-ttng.mlir` | 27 | 27 | — |
| `test/TritonNvidiaGPU/membar-cluster.mlir` | 29 | 29 | — |

Le funzioni toccate da v1: `warp_specialize_into_default` (2→1), `default_region_cfg` (3→1),
`check_barrier_no_duplication` (2→1). Con v2 resta la barriera dopo la region in
`default_region_cfg` (era la regola yield).

## Cosa manca prima di proporla

1. Misura v2 e aggiornamento dei tre `CHECK-NEXT: ttg.barrier local` in cima alle default region.
2. Un test **positivo**: `warp_specialize` con capture, dove la barriera deve **restare** → fissa il
   gate invece di limitarsi a cancellare righe.
3. Run con `--instrumentation-mode=consan` con e senza patch: è l'unica configurazione dove i due
   predicati divergono, ed è la prima cosa che chiederà un reviewer.
4. Destinatario: **@Jokeren**, CODEOWNER di `lib/Analysis/Membar.cpp` (`.github/CODEOWNERS`).
   Assenso già registrato sull'idea: ThomasRaoux su #8374, *"yeah warp_specialization op can count
   as a bar sync"*.

## Anello debole dichiarato

La regola dipende da un contratto che non è scritto da nessuna parte: che il lowering continui a
emettere quella barriera. L'header documenta `createAllBarrier` come *"synchronize threads across the
whole CTA"* e non dice nulla sull'ordinamento della shared memory. Entrambi i backend in-tree
soddisfano la proprietà più forte, ma la PR deve dirlo esplicitamente e il commento nel codice deve
citare il punto del lowering, altrimenti la prossima modifica lo rompe in silenzio.
