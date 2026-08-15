---
name: upstream-contribution-method
description: Come si porta a casa un contributo upstream su Triton — il metodo che ha prodotto le PR #11323 e #11324
metadata:
  type: project
---

Ricavato dalla notte del 2026-08-16, in cui sono state aperte due PR upstream partendo da zero.

**Why:** il collo di bottiglia upstream non è scrivere la patch, è (a) scegliere un bersaglio che
esista davvero e non sia già preso, e (b) arrivare con l'evidenza che il maintainer chiederebbe.

**How to apply — nell'ordine:**

1. **Leggere i commenti dell'issue PRIMA di investigare.** Su #11111 la causa (bug di ptxas 12.9) era
   già nel thread dal 3 agosto: 40 minuti di riproduzione per riscoprirla. Costo evitabile.
2. **Verificare che il bersaglio non sia preso**: `gh api "search/issues?q=repo:triton-lang/triton+is:pr+<N>"`.
   Le issue piccole e ben scritte vengono raccolte in giorni; chi arriva secondo butta il lavoro
   (è successo a `#11228`, chiusa con "Solved by #10891").
3. **Misurare prima di rivendicare.** Il primo bersaglio (`WaitBarrier ↔ TMA`) sembrava buono e aveva
   delta **zero**; la regola su `warp_yield`, tenuta fuori per disciplina di scope, si è rivelata
   quella che toglie **30 barriere** dal PTX generato. Senza misura si sarebbe scelto il contrario.
4. **Falsificare la propria ipotesi con un esperimento, non con un ragionamento.** L'ipotesi "barriera
   insufficiente" su #11111 è caduta in una build.
5. **Far attaccare la patch da agenti avversariali prima di pubblicarla.** Hanno trovato: un puntatore
   al codice sbagliato nel commento, un secondo effetto non dichiarato (e non testato) e un test che
   dipendeva dal piazzamento dell'allocatore. Tutte cose che in review costano un giro.
6. **Riempire la new-contributor checklist** di `.github/PULL_REQUEST_TEMPLATE.md` (obbligatoria sotto
   le 3 PR mergiate) e girare davvero `pre-commit run --from-ref origin/main --to-ref HEAD`.
7. **Un push solo, poi congelare il branch**: vedi [[triton-ci-needs-maintainer-approval]].

Vedi anche [[triton-build-harness-on-modal]].
