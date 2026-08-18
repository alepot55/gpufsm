# Autorizzazione permanente ad agire upstream

Il **18 ago 2026** l'utente ha dato un'autorizzazione esplicita e generale: *"ti autorizzo a fare
sempre, procedi e ricordalo"*. Copre le azioni verso l'esterno sul lavoro upstream, che fino a quel
momento fermavo alla bozza:

- **rispondere ai revisori** su `llvm/llvm-project` e `triton-lang/triton`, dentro il thread inline;
- **pushare** sui rami del fork `alepot55/*` che alimentano le nostre PR;
- applicare i rilievi e aggiornare le PR senza chiedere conferma a ogni giro.

È arrivata dopo tre giri in cui preparavo la risposta e poi mi fermavo a chiedere. Il costo di quel
loop era già documentato in [[answer-reviewers-immediately]]: tenere ferme le repliche è costato 7
ore, e la policy chiede **un umano responsabile, non che digiti lui**. Questa autorizzazione è
esattamente ciò che quella nota diceva servire.

## Cosa NON copre

Non è un'autorizzazione a **sottomettere il paper**. Premere SUBMIT su HotCRP è irreversibile, passa
dall'account dell'utente e vale una sola volta per ciclo: resta suo, ed è la ragione per cui PPoPP
2027 è stata persa (account creato, submit mai premuto). Il sentinella ASPLOS avvisa, non sottomette.

Regola pratica che resta valida anche con l'autorizzazione: **verificare prima di rispondere.** Il
merito di un rilievo va controllato contro il codice, non concesso per cortesia
([[verify-by-running-not-by-agent-verdict]], [[last-speaker-is-not-a-criterion]]). Su Triton vale
ancora che ogni push azzera la coda di approvazione della CI, quindi si accumula e si pusha una volta
sola ([[triton-ci-needs-maintainer-approval]]); su LLVM la CI parte da sola e il vincolo non esiste.
