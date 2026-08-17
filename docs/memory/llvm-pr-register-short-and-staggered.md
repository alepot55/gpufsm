---
name: llvm-pr-register-short-and-staggered
description: La prima parola di un maintainer MLIR sulla nostra prima PR revisionata e' stata "slop" - le descrizioni lunghe e le PR aperte in blocco lavorano contro di noi
metadata:
  type: feedback
---

17 ago 2026, 21:50: apriamo 4 PR su LLVM in **29 secondi**. Alle 22:09, cioe' **19 minuti dopo**,
Mehdi Amini (`joker-eph`, responsabile MLIR) risponde su [#216852](https://github.com/llvm/llvm-project/pull/216852):

> Thanks for the fix. Can we start by please pruning the description from the slop?
> Second can the test be included in the existing canonicalize.mlir file?

Le nostre descrizioni erano tra **2.300 e 4.500 caratteri**: sezioni `### Verified by execution`,
`**The decisive control:**`, esegesi sulle modalita' di build. Tagliate a **544-1.119**.

**Why:** su LLVM la descrizione serve a far capire il bug in trenta secondi, non a dimostrare quanto
abbiamo lavorato. Tutto il resto e' materiale da **risposta in revisione, quando viene chiesto**.
E "slop" non e' un rilievo di stile: e' la parola con cui si etichetta il contributo generato in
serie. Su un progetto che ha una [AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html) e da
cui dipendiamo, e' a un passo dal problema serio. Vedi [[llvm-requires-ai-disclosure]].

**How to apply:**

1. **Corpo PR: 500-1.200 caratteri.** Sintomo (con il repro minimo), una riga di causa, la
   correzione, e le modifiche a test altrui con il motivo. Piu' la dichiarazione AI, obbligatoria.
   Niente titoli di sezione, niente elenchi di verifiche fatte: quelle si tirano fuori se chiedono.
2. **Mai aprire PR in blocco.** Quattro in 29 secondi, sul cruscotto di un revisore, e' la firma di
   chi scarica volume. Distanziarle, e non aprirne di nuove finche' le aperte non sono pulite e
   risposte: il collo di bottiglia e' la capacita' di revisione, non la nostra produzione.
3. **Concedere sulle obiezioni di disegno.** Su quella stessa PR `Hardcode84` ha obiettato che la
   canonicalizzazione non dovrebbe creare `cf` dal nulla. Risposta data: "puo' avere ragione, non
   difendo il pattern; se lo rimuovono questa PR diventa inutile e la chiudo". Impuntarsi e' il modo
   classico di morire in revisione, e non era il nostro punto.
4. **Rispondere con fatti misurati.** Alla domanda sul file di test la risposta e' stata: eseguita
   la pipeline di `canonicalize.mlir` sul nostro caso, l'operazione esce intatta perche' quel file
   ancora tutto a `func.func`. Fatto, non difesa d'ufficio.
5. **Le risposte ai maintainer le scrive l'utente.** E' quello che la policy chiede. Preparargliele
   va bene; mandarle a suo nome solo se lo chiede esplicitamente, e dichiarandolo.

Correzione a una nostra convinzione precedente: [[pick-uncontested-bugs-not-design-changes]] diceva
che il collo di bottiglia e' il filtro di valore, non l'attenzione. Resta vero sulla *selezione*,
ma qui l'attenzione e' arrivata in 19 minuti e il primo attrito e' stato sulla **forma**.
