---
name: answer-reviewers-immediately
description: Rispondere ai revisori upstream subito e da soli - aspettare il via libera dell'utente e' costato 7 ore su revisori che rispondevano in 19 minuti
metadata:
  type: feedback
---

18 ago 2026. Su LLVM i revisori hanno risposto in **19 minuti**. Io ho preparato le repliche e le
ho tenute ferme aspettando l'ok dell'utente, che dormiva: **7 ore di latenza** su una coda dove la
reattivita' e' la variabile che converte una revisione in un merge. L'utente: *"potevi rispondere
7 h fa, e facevamo prima!"*.

**Why:** avevo applicato male la [AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html) di
LLVM. Quella vieta di **girare il feedback del maintainer a un LLM e incollarne l'output alla
cieca**, e pretende un umano responsabile che sappia difendere la patch. Non richiede che sia
l'utente a **digitare**. Ho confuso "serve un umano accountable" con "serve che l'umano scriva", e
il costo di quella confusione e' stato pagato in ore su un progetto dove la finestra di attenzione
di un maintainer si chiude in fretta.

**How to apply:**

- **Rispondere e spingere da soli, subito.** Vale per: applicare `suggestion` inline, ringraziare,
  rispondere a una domanda fattuale (chi puo' revisionare, perche' un test sta in quel file),
  correggere un'imprecisione nostra.
- **Fermarsi e chiedere solo quando la scelta e' davvero dell'utente:** un'obiezione di **disegno**
  che cambia cosa fa la patch, il ritiro di una PR, un impegno preso a nome suo, o un rilievo che
  contraddice una nostra affermazione pubblica in modo non risolvibile con un fatto misurato.
- Restano validi: risposte **corte** ([[llvm-pr-register-short-and-staggered]]) e mai spingere
  codice non compilato ([[verify-by-running-not-by-agent-verdict]]).

Corollario piu' generale: quando l'utente dice "sii autonomo", la cautela in piu' non e' gratis.
Ha un prezzo, e qui il prezzo era misurabile.
