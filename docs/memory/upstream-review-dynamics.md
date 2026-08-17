---
name: upstream-review-dynamics
description: Come si comporta un maintainer Triton in review e cosa fa la differenza fra una PR ferma e una che avanza
metadata:
  type: project
---

Ricavato dal 17 ago 2026, il giorno in cui Jokeren e' passato da due giorni di silenzio a **cinque
interventi in tre ore** su due PR.

**Why:** per due giorni la lettura era "ci ignorano". Era sbagliata: il collo di bottiglia non era il
loro interesse, era che le PR non erano ancora nella forma che si puo' revisionare in due minuti.

**How to apply:**

1. **Rispondere DENTRO il thread inline, non con un commento nuovo.** Un thread risolto o ignorato non
   riemerge se rispondi in cima alla PR. E' cosi' che #10766 e' rimasta ferma sei settimane: il thread
   di peterbell10 era stato **chiuso da noi** senza mai rispondergli, quindi dal suo lato risultava
   sistemato.
2. **Ogni sua richiesta si esegue, anche quelle cosmetiche** ("Why adding an empty new line?"). Costano
   un minuto e ogni giro non fatto e' una settimana di attesa.
3. **Ma quando ha torto, si mostra l'output, non si argomenta.** Ha scritto "your test checks nothing";
   la risposta e' stata incollare l'IR prima/dopo dove la barriera sparisce. Dichiarando comunque che
   la decisione resta sua. Lasciar passare un'affermazione sbagliata indebolisce meta' della PR.
4. **Verificare la sua proposta prima di eseguirla, e dirlo se e' migliore della nostra.** Ha chiesto di
   spostare due op in `containsLocalBarrier`: verificato nel codice che `WarpReturnOp` emette davvero la
   stessa barriera → la nostra patch era **incompleta**, la sua versione **raddoppia** l'effetto.
5. **La direzione della PR e' verso il piccolo.** #11324: da +81/-4 iniziali a **+2/-1**. Ogni giro di
   review ha tolto qualcosa: il caso speciale, i commenti, un test quasi-duplicato. La forma che
   mergiano e' una riga in un elenco che esiste gia'.
6. **La CI approvata e' il segnale vero.** Non approvano la CI di una PR che intendono chiudere. Su
   #11324 e' stata approvata **due volte** in un pomeriggio. Vedi [[triton-ci-needs-maintainer-approval]].

**Sul contraddire:** e' andata bene una volta su due. Sul test `warp_return` aveva torto e l'output lo
dimostrava. Sul mio guardiano per identita' del valore aveva ragione lui e la mia versione perdeva
precisione. La regola che ha funzionato: **verificare sempre, poi dire com'e' andata**, in entrambe le
direzioni. Vedi [[upstream-contribution-method]].
