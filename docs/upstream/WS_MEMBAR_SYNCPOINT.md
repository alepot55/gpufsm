# Contributo upstream in preparazione: `warp_specialize` come punto di sincronizzazione in membar

Stato: **PR aperta upstream il 2026-08-16: triton-lang/triton#11323.** Sostituisce il bersaglio precedente
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
| `test/Analysis/test-membar.mlir` | 127 | 123 | **124** |
| `test/Analysis/test-membar-ttng.mlir` | 27 | 27 | 27 |
| `test/TritonNvidiaGPU/membar-cluster.mlir` | 29 | 29 | 29 |

**Il caso discriminante è stato costruito e misurato**, non lasciato alla teoria
(`docs/upstream/artifacts/ws_consan_war.mlir`): un buffer che muore prima della `warp_specialize`,
così l'allocatore può riusarne i byte per lo scratch che il sanitizer riserva. Baseline: la barriera
prima della `warp_specialize` **c'è**. v1: **sparisce** (era il bug). v2: **resta**. È esattamente la
differenza tra il gate sbagliato e quello giusto, e gira senza GPU.

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


## Trappole trovate scrivendo i test (da non riscoprire)

1. **`test/Analysis/test-membar.mlir` ha DUE RUN line**: la seconda gira il pass membar **AMD**
   (`-test-tritonamdgpu-membar`) sullo stesso file, con lo stesso prefisso `CHECK`. Un test che
   dipende dal backend (per esempio dai byte di capture che il sanitizer riserva, che solo lo
   scratch-size fn NVIDIA somma) **fallisce sulla seconda RUN**. Il primo tentativo di test è morto
   esattamente così. Soluzione: scrivere il caso in modo backend-agnostico — capture vere invece
   dell'attributo del sanitizer — e tenere l'argomento ConSan nella prosa della PR, non nel lit test.
2. Il pass AMD annota `ttg.amdg.syncedViaAsyncWait` sulle `local_load`: le CHECK devono restare
   sottostringhe (`local_load`), non righe intere.
3. **`filecheck` di PyPI** funziona come sostituto di `FileCheck` (non è nel pacchetto LLVM che
   Triton scarica): `pip install filecheck` + symlink `FileCheck` nel venv. Non è identico
   all'originale, quindi il metodo giusto è **confrontare la lista dei test falliti prima e dopo**,
   non guardare il numero assoluto: sul baseline ne falliscono già 29 per differenze del sostituto.


## Verifica finale (misurata, non dedotta)

- **127 → 124** barriere emesse: stesso file di test non modificato, due binari diversi
  (`triton-opt` dell'albero baseline e di quello patchato nel volume Modal). È il confronto pulito:
  nessuna modifica ai test può inquinarlo.
- **Nessun nuovo lit failure** su `test/Analysis`, `test/TritonGPU`, `test/Conversion`,
  `test/TritonNvidiaGPU`, `test/NVWS`: la lista dei test falliti è identica prima e dopo (29 in
  entrambi i casi, tutti dovuti al sostituto `filecheck`, non alla patch).
- `clang-format` pulito sul file toccato.


## Verifica su H100 (Modal) — il dato che ridimensiona la portata

Stessa suite, stesso container, due compilatori (baseline e patchato):

- `python/test/unit/language/test_warp_specialization.py`: **145 passed, 1432 skipped** in entrambi i
  casi → la patch non rompe nulla su hardware vero.
- Barriere nel PTX generato: **2356 su 75 file PTX in entrambi i casi** → **la regola non scatta nel
  codice generato**.

Il motivo è strutturale e va detto nella PR invece di lasciarlo scoprire a un reviewer: le partizioni
di `ttg.warp_specialize` sono `IsolatedFromAbove`, quindi nel codice prodotto dal pipeliner l'op ha
sempre capture esplicite → possiede uno scratch buffer → il rendezvous era **già** modellato dal path
scratch. Il buco riguarda la forma **senza capture**, che è quella dei test e dei kernel Gluon scritti
a mano.

⇒ La PR va presentata per quello che è: completamento del modello, non guadagno di performance.
Rivendicare velocità qui sarebbe falso e verificabile in trenta secondi da chi la legge.


## PR aperta

<https://github.com/triton-lang/triton/pull/11323> — 3 file, +81/-4, review auto-assegnata dal
CODEOWNERS a **Jokeren** (proprietario di `lib/Analysis/Membar.cpp`) e ptillet. CI in
`action_required`: serve il click di un maintainer, quindi **branch congelato**.

Cosa è cambiato rispetto alla v2 grazie alla review pre-submit (tre reviewer indipendenti):
1. Il commento citava `lowerWarpSpecialize` in un file dove quella funzione non esiste. Il punto giusto
   è `lowerWarpSpecializeCommon` in `WarpSpecializeUtility.cpp`. È l'unica riga che giustifica
   l'assunzione cross-pass: sbagliarla avrebbe fatto perdere fiducia nel resto.
2. **Secondo effetto non dichiarato**: `beforeMemoryEffects` è letto anche da
   `hasSyncPointBeforeMemoryEffect`, quindi una wait di memoria subito prima di una
   `warp_specialize` senza capture ora rimanda la sua barriera all'ingresso della regione, e una
   `warp_group_dot_wait` in quella posizione viene marcata `warpGroupLocal`. Misurato, testato (3 test
   nuovi) e scritto nella descrizione invece che scoperto in review.
3. Un test dipendeva dal piazzamento dell'allocatore (coincidenza di offset): sostituito con due che
   non dipendono da nulla del genere.

Verifica finale: **lit con il FileCheck vero di Triton** (che sta in `python/triton/FileCheck`, non
nel pacchetto LLVM scaricato) su 5 suite → stesso identico set di 14 fallimenti pre-esistenti prima e
dopo, nessuno è un test membar; `pre-commit` verde.


## La regola `warp_yield`: quella che vale davvero (misurata il 16 ago, notte)

Tenuta fuori dalla prima PR per disciplina di scope. Misurata dopo, ed è il contrario di quello che
sembrava: **è lei quella con impatto sul codice generato.**

| albero | corpus membar | PTX generato (suite warp-specialization, H100) | test |
|---|---:|---:|---|
| baseline | 127 | 2356 barriere / 75 kernel | 145 passed |
| solo entry (PR #11323) | 124 | 2356 (invariato) | 145 passed |
| entry + yield | 123 | **2326 (−30)** | 145 passed |
| solo yield | 126 | da confermare (atteso 2326) | — |

Soundness verificata alla fonte, non per analogia: `WarpSpecializeUtility.cpp` emette
`createAllBarrier` a **ogni `WarpReturnOp`** delle partizioni (:420-426) e a **ogni `WarpYieldOp`**
della default region, con lo **stesso `switchLoopBarrierIdx`** → è letteralmente la stessa barriera
CTA-wide, quindi un rendezvous fra i gruppi di warp. Il conteggio degli arrivi torna: su ogni path
ciascuna warp ne esegue esattamente tre.

Lit: stesso identico set di 14 fallimenti pre-esistenti, nessuna regressione.

## Misura di performance (16 ago 2026, H100 su Modal)

Domanda: le barriere tolte da #11324 valgono tempo, o solo conteggio?

Metodo: `matmul_tma_ws_kernel` (da `python/test/unit/language/test_warp_specialization.py`),
4096x4096x4096 fp16, TMA su A e B, `num_warps=4` (su Hopper con 8 la warp specialization si
disattiva), correttezza verificata contro cuBLAS prima di cronometrare. Le due build girano nello
**stesso container** sulla stessa H100, per non confrontare affitti diversi.

| BN/BK/stages | bar.sync base | bar.sync patch | ms base | ms patch | delta |
|---|---|---|---|---|---|
| 128/128/2 | 25 | 24 | 0.2790 | 0.2797 | +0.25% |
| 128/128/3 | 25 | 24 | 0.2811 | 0.2810 | -0.04% |
| 128/64/2  | 25 | 24 | 0.3546 | 0.3555 | +0.25% |
| 128/64/3  | 25 | 24 | 0.2881 | 0.2870 | -0.38% |
| 128/64/4  | 25 | 24 | 0.2718 | 0.2712 | -0.22% |
| 256/64/2  | 25 | 24 | 0.2509 | 0.2510 | +0.04% |
| 256/64/3  | 25 | 24 | 0.2161 | 0.2166 | +0.23% |

**Esito: nessun guadagno misurabile.** Una barriera in meno per kernel, delta di tempo senza segno
coerente (rumore). La barriera rimossa non sta nel ciclo caldo: sta sul percorso di uscita della
regione warp-specialized, attraversato una volta per tile-loop e non per iterazione.

Conseguenze da tenere presenti:
- La PR **non rivendica** un guadagno di velocità, quindi non c'è nulla da correggere: dice
  "2356 -> 2326 bar.sync su 75 kernel", che è vero e verificato.
- Ma l'argomento a favore non e' la performance: e' che la membar analysis era **dimostrabilmente
  troppo conservativa**, e la precisione di quell'analisi e' il punto. Da usare cosi' se un
  maintainer chiede "a cosa serve".
- ⚠️ Metodo: NON ricavare conteggi per-kernel dalle cache lasciate da run precedenti. Piu' varianti
  compilate condividono lo stesso nome file (`matmul_tma_ws_kernel.ptx`) sotto hash diversi, e due
  script che ne pescano una diversa danno numeri incoerenti (mi e' successo: 25->18 vs 15->18).
  L'unico numero difendibile viene da una run controllata con cache pulita.

## Esito di #11323: CHIUSA da Jokeren il 17 ago

> "discussed with @jeffniu-openai and this seems like a micro optimization."

Verificato prima di rispondere, in `lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp:555-559`:
le due `createAllBarrier` attorno alla default region sono emesse **incondizionatamente** per ogni
`warp_specialize`, senza guardia su capture o partizioni — cioe' esattamente lo schema che Jokeren ha
disegnato nel commento. **La patch era corretta**; il rifiuto e' sul valore, non sulla correttezza.

E il valore era davvero marginale, e lo avevamo scritto noi per primi nella descrizione della PR:
*non cambia il codice generato*. Non c'era niente da difendere.

**Lezione:** il criterio [[triton-rejects-trivial-prs]] ha predetto questo esito con due giorni di
anticipo. Era gia' scritto nel nostro doc che #11323 rischiava la chiusura per "not necessary". Averlo
previsto e non aver ritirato la PR e' stata comunque la scelta giusta: ritirarsi prima di un'obiezione
regala terreno, e il costo reale e' stato una riga di risposta.

**Cosa ne abbiamo ricavato:** il primo scambio tecnico con un maintainer. Nella risposta abbiamo
indirizzato l'attenzione su #11324 (che il codice lo cambia) e soprattutto su #11325 (barriera
**mancante**, cioe' correttezza e non ottimizzazione), dicendo esplicitamente che se considera #11324
della stessa classe non discutiamo. Onesti sul debole, precisi sul forte.

## #11324: prima review evasa (17 ago)

Jokeren, commento inline su `Membar.cpp:184`:

> You can just include `triton::gpu::WarpYieldOp` and `triton::gpu::WarpReturnOp` in
> `containsLocalBarrier` and remove comments

**La sua versione era migliore, e la nostra patch era incompleta.** Verificato prima di eseguire:
`WarpReturnOp` emette la stessa `createAllBarrier` di `WarpYieldOp`
(`WarpSpecializeUtility.cpp:420-422` contro `:555-559`), e noi la ignoravamo. Includerla **raddoppia**
l'effetto. Codice: −9 righe, +2, il caso speciale in `getLocalBarrierStages` sparisce.

| build | bar.sync (75 kernel, cache pulita per build) |
|---|---|
| `main` | 2016 |
| solo `warp_yield` (v1, quella pubblicata) | 1986 |
| entrambi i terminatori (v2, come chiesto) | **1956** |

145 passed / 1432 skipped in tutte e tre. Nessuna regressione lit.

### ⚠️ Errore nostro, corretto pubblicamente

La descrizione originale della PR diceva **2356 → 2326**. I numeri **assoluti erano sbagliati**:
venivano da un conteggio fatto in modo diverso (stessa suite, stessi 75 kernel, metodo di conteggio
differente). Il **delta di 30 era giusto**. Descrizione della PR corretta a 2016 → 1956, e l'errore
**dichiarato nel commento** invece di sostituire i numeri in silenzio.

⇒ Regola: quando si ri-misura, ri-misurare **tutte** le build con lo stesso comando nella stessa run.
Confrontare un numero di oggi con uno di ieri prodotto da un altro script e' come non misurare.

### Cosa non siamo riusciti a fare, detto in chiaro

Tre forme di test minimale per isolare `warp_return` (partizione che scrive + lettura del chiamante;
il contrario; due partizioni sullo stesso buffer) → **nessuna** mostra differenza tra `main` e la
patch. Quel mezzo cambiamento si vede solo nella suite completa. Nella risposta e' offerta la
riduzione di un kernel vero a test, invece di aggiungere un test che non dimostra nulla.
