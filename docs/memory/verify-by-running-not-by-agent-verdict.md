---
name: verify-by-running-not-by-agent-verdict
description: Un revisore avversariale ha confermato 10 candidati su 11 - eseguendoli, 2 su 7 erano falsi. Il timbro degli agenti non e' una prova
metadata:
  type: feedback
---

Caccia bug LLVM/MLIR del 17 ago 2026 (`wf_f8e41f40-1d9`, 16 agenti): 5 cercatori su modalita'
diverse, poi **un revisore avversariale per candidato con mandato esplicito di refutare** e istruzione
di default a `isReal=false` in caso di dubbio.

Risultato del revisore: **10 confermati su 11**, quasi tutti "high confidence".

Risultato eseguendo `mlir-opt` sui repro, su `main` pulito:

| candidato | verdetto agente | esecuzione |
|---|---|---|
| coalescing SCF (iter_arg catturato) | high | OK - `addi %arg5, %arg5` |
| `index.cmp` su sottrazione che wrappa | high | OK - la sub sparisce |
| `vector.multi_reduction` dims fuori range | high | OK - verifier passa, poi rc=134 |
| SCF senza `cf` fra i dependentDialects | high | OK - rc=134 in `cf::BranchOp::create` |
| mem2reg su `memref<0xf32>` | high | OK - rc=134 in `VectorType::get` |
| copia dati affine / dominanza | high | FALSO - rc=0, nessuna violazione |
| `getStaticTripCount` che wrappa | high | FALSO - la store resta dentro il ciclo |

**Why:** un tasso di conferma del 91% significa che il filtro non filtra. Il difetto e' strutturale,
non del prompt: chiedere a un LLM di refutare un'analisi plausibile scritta da un altro LLM produce
concordanza, perche' entrambi ragionano sullo stesso codice con gli stessi bias. Il revisore
avversariale serve a **ordinare** i candidati, non a promuoverli.

Nota sui due falsi: `getStaticTripCount` **contiene davvero** il difetto (il percorso a bound
costanti clampa a 0, quello a mappe affini fa `uint64_t value = constExpr.getValue()` e un -10
diventa 1.8e19). Cio' che non esiste e' il **danno**: LICM non sposta comunque la store. Un bug
latente senza manifestazione non e' pubblicabile — nessun test lo pinna, e il maintainer chiede
"e allora?". Distinguere sempre *il difetto* dal *sintomo osservabile*: si presenta il secondo.

**How to apply:**

1. Nessun numero e nessuna PR prima di aver **eseguito** il repro sul compilatore vero. Vale
   esattamente come la regola dell'oracolo per i backend GPU di questa repo.
2. Batch: un solo contenitore, tutti i repro insieme, un marcatore per caso e `rc=` stampato.
   Costa quanto uno e li discrimina tutti.
3. Attenzione al fallimento a meta' script: parentesi non quotate in
   `-pass-pipeline=builtin.module(func.func(...))` hanno ucciso lo script dopo il primo caso, e
   l'output sembrava un successo parziale. Quotare sempre l'argomento e verificare con `bash -n`.
4. Prima di innamorarsi: cercare la PR concorrente. Su coalescing esisteva gia'
   [#216494](https://github.com/llvm/llvm-project/pull/216494) aperta da un altro contributore.

Vedi [[pick-uncontested-bugs-not-design-changes]] per la scelta del bersaglio e
[[never-poll-a-job-you-cant-see]] per l'harness.
