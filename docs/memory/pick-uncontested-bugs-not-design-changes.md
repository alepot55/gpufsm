---
name: pick-uncontested-bugs-not-design-changes
description: Il criterio che separa una PR upstream che passa da una che muore in review - il verificatore deve dare ragione a te, non un maintainer
metadata:
  type: feedback
---

Su 9 PR aperte su Triton, 4 sono state respinte e **nessuna** per errori tecnici. Le motivazioni
testuali: *"block pointer is deprecated"*, *"niche pattern matching"*, *"are there practical use
cases?"*, *"I don't think this is the right fix"*. Latenza mediana di risposta: **1 giorno**.

**Why:** il collo di bottiglia upstream non e' l'attenzione dei maintainer, e' il **filtro di
valore**. Per mesi ho creduto il contrario ("dipende da un maintainer che clicchi") e ho ottimizzato
la cosa sbagliata: la perseveranza invece della selezione del bersaglio. Il dato lo smentisce.

**How to apply — il criterio di scelta, in ordine:**

1. **Preferire i bug a cui risponde una macchina.** Un miscompile o una violazione di dominanza la
   rifiuta il *verificatore*: nessuno puo' chiedere "a cosa serve". Una modifica che migliora un
   predicato, tocca un contratto di disegno o aggiunge una canonicalizzazione richiede che un umano
   sia d'accordo, ed e' li' che si muore.
2. **Scartare esplicitamente i bersagli contesi**, anche quando il match di competenza e' migliore.
   Esempio concreto: llvm#216542 (`control-flow-sink` vs validita' affine) era piu' vicino alla mia
   esperienza di llvm#216545, ma nessun pass generico di MLIR conosce le regole affini, quindi "chi
   deve cedere" e' una discussione aperta. Ho preso #216545 e la PR e' uscita in mezza giornata.
3. **Verificare l'eta' al contrario di come sembra.** Una issue di 15 mesi "confermata dal
   maintainer" non e' un'opportunita': e' un bug che nessuno con commit rights ha tempo di seguire.
   Una di 0-3 giorni senza opinioni si chiude piu' facilmente.
4. **Controllare che il residuo esista davvero** prima di innamorarsi della descrizione: su IREE
   #20602 due terzi del lavoro erano gia' fatti upstream e il resto si mergiava in un altro repo.

**Controprova del 18 ago 2026, nello stesso giorno e sugli stessi due progetti.** Su LLVM abbiamo
6 PR aperte, tutte da crash/miscompile/violazioni del verificatore: **14 rilievi ricevuti, ZERO
sulla sostanza**, una gia' con 2 approvazioni e CI verde. Su Triton, nello stesso giorno, la PR
#11323 e' stata chiusa da un maintainer con *"discussed with @jeffniu-openai and this seems like a
micro optimization"* — era un'**ottimizzazione**, cioe' proprio la categoria in cui serve che un
umano sia d'accordo. Il criterio non e' una teoria: e' la differenza fra le due colonne.

Corollario sul CV: il conteggio delle PR e' la metrica sbagliata. Una cosa sostanziale verificabile
vale piu' di dieci aperte. Vedi [[upstream-contribution-method]] per il metodo di verifica, e
[[triton-rejects-trivial-prs]] per la soglia di questi maintainer.
