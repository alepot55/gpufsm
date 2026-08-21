---
name: judge-a-fix-by-the-invariant-not-the-symptom
description: ho approvato pubblicamente una fix upstream perché il crash spariva; il maintainer l'ha respinta perché indeboliva un invariante, e aveva ragione
metadata:
  type: feedback
---

# Il crash sparito non è il criterio

Il 21 ago 2026 su triton#11396 whutsunxu proponeva di togliere il `report_fatal_error` in
`ClusterBarrierInsertion.cpp` e di **gestire** la dipendenza (controlla intersezione, inserisci la
barriera, sincronizza). Avevo costruito e misurato: crash sparito su entrambi i riproduttori, due
suite verdi, nessuna doppia barriera. Ho scritto pubblicamente che era **meglio** della deroga che
avevo abbozzato io.

Un'ora dopo Jokeren ha messo `CHANGES_REQUESTED`: *"Normal ops shouldn't have any shared memory
dependency, but instead only `LocalAtomicScatterRMWOp` is special."* Togliere l'assert non risolve
un caso: **smette di controllare tutti gli altri**. Il fix giusto tiene l'assert e deroga per l'unica
op che ha davvero shared memory fra gli operandi — cosa che whutsunxu ha poi confermato enumerando
le op con scratch buffer (`convert_layout`, `local_atomic_scatter_rmw`, `warp_specialize`).

**Why:** avevo verificato eseguendo, come dice [[verify-by-running-not-by-agent-verdict]], e
l'esecuzione era corretta. L'errore era **cosa** avevo messo alla prova. Un test dice se il sintomo
è sparito; non dice se l'invariante che quel codice difendeva vale ancora. Un assert che non scatta
più perché l'hai tolto passa qualsiasi suite.

**How to apply:** prima di dire che una patch è buona, chiedersi *cosa smette di essere controllato*.
Per una patch che rimuove o allarga un controllo, le domande sono due e la seconda non è testabile
con un riproduttore:

1. il caso rotto ora passa? (lo dice il build)
2. per quali **altri** input il controllo non parla più? (lo dice solo leggere l'invariante)

Corollario di registro: su una PR di qualcun altro, un giudizio comparativo del tipo "meglio di X"
è un'opinione, e va speso solo dopo la domanda 2. Riportare le misure non costa nulla ed è sempre
utile; sponsorizzare un approccio sì. Vedi [[upstream-review-dynamics]].
