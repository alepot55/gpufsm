---
name: llvm-requires-ai-disclosure
description: LLVM obbliga a dichiarare l'uso di AI nella PR e pretende un umano che sappia difendere la patch in review
metadata:
  type: reference
---

`llvm/llvm-project` ha una **[AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)**, e il bot
di benvenuto alla prima PR chiede esplicitamente di rispondere confermando di averla letta.

Cosa impone:

1. **Dichiarare l'uso di AI** "in the pull request description, commit message, or wherever authorship
   is normally indicated", con esempio di trailer `Assisted-by:`. Non c'e' formula obbligatoria, ma la
   dichiarazione si'.
2. **Human in the loop, non formale:** *"Contributors must read and review all LLM-generated code or
   text before they ask other project members to review it"*, *"The contributor is always the author
   and is fully accountable"*, e soprattutto **non si puo' girare il feedback del maintainer a un LLM**:
   l'autore deve capire e difendere la patch.

**Why:** e' una condizione di ammissibilita', non galateo. Una PR assistita da AI e non dichiarata e'
una violazione, e su un progetto dove si costruisce reputazione costa piu' di quanto valga la PR.

**How to apply:** la dichiarazione va messa **prima o subito dopo** l'apertura (su llvm#216605 e'
stata aggiunta entro pochi minuti via `gh api -X PATCH repos/.../pulls/N -F body=@file`; `gh pr edit`
falliva per un errore GraphQL sui Projects classic, non correlato). La **risposta di conferma al bot
la scrive l'utente**, non l'agente: e' una dichiarazione a suo nome. All'utente va prima spiegata la
patch in modo che la dichiarazione sia vera, e va anticipata l'obiezione probabile del reviewer con
la risposta pronta.

Altri progetti non hanno questa regola: e' specifica di LLVM. Vedi
[[pick-uncontested-bugs-not-design-changes]] per come si sceglie il bersaglio.
