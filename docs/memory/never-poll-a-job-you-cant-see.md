---
name: never-poll-a-job-you-cant-see
description: Due guasti annidati che hanno bruciato 90 minuti due volte - git su volume di rete, e cicli di attesa su un file che non arrivera' mai
metadata:
  type: feedback
---

Il 17 ago 2026 un agente del workflow di caccia bug e' rimasto fermo **92 minuti**. Sopra c'erano
due guasti distinti, e nessuno dei due si annuncia:

**1. `git status` / `git diff` su un volume Modal.** Il volume `llvm-upstream` e' un filesystem di
rete con ~432.000 inode e la build dentro il repo. Ogni comando git che tocca il working tree fa
uno `stat` per file: non calcola, striscia. Sembra identico a un processo bloccato. Lo stesso
comando senza la parte git torna in ~2 minuti.

**2. Il ciclo di attesa su un file che non arrivera'.** L'agente aveva lanciato il comando in
background su `run3.log` e poi aspettava con
`for i in $(seq 1 55); do [ -s run3.log ] && break; sleep 10; done`. La build a monte era gia'
fallita (`git checkout -f <sha>` -> rc 128, il commit non c'era nel clone parziale), quindi il file
restava vuoto e ogni chiamata bruciava 9 minuti. Ripetuta quattro volte.

**Why:** e' la stessa famiglia dei 42 shell `sleep 3500` orfani del giorno prima. Un osservatore che
fallisce in silenzio e' peggio di nessun osservatore: e' indistinguibile dalla calma, e nessuno va a
controllare una cosa che sembra star lavorando.

**How to apply:**

- Nei prompt dei subagent, **vietare esplicitamente** Modal e i cicli di attesa. Non basta non
  chiederli: un agente che vuole verificare li reinventa. Il divieto va scritto con il motivo.
- Su un albero montato via rete, mai comandi git sul working tree. Solo `mlir-opt` (o l'eseguibile
  che serve) con **path assoluto**: `PATH` nel container non contiene la build.
- Chi orchestra tiene per se' l'unico volume di build. Cinque agenti che compilano in parallelo
  esauriscono gli inode del volume. Il lavoro parallelizzabile e' scrivere la patch e il test, e
  quello si fa in **git worktree locali separati** (`git worktree add --detach`), uno per agente.
- Sbloccare un agente in attesa scrivendo nel file che aspetta e' legittimo e piu' pulito che
  ucciderlo: gli si dice cosa e' successo e conclude con la confidenza giusta.
- Diagnosi rapida di un workflow fermo: `comm -23` tra gli `agentId` con evento `started` e quelli
  con qualsiasi altro evento nel `journal.jsonl` da' subito chi non ha mai chiuso.

Vedi [[modal-gpu-harness-gotchas]] per le altre trappole dell'harness e
[[verify-by-running-not-by-agent-verdict]] per perche' l'esecuzione resta l'unico giudice.
