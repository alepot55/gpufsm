---
name: cite-your-prior-work-in-third-person
description: sotto double-blind il lavoro precedente proprio si cita normalmente, in terza persona; una voce {{Anonymous}} "under review" è il difetto, non la tutela
metadata:
  type: reference
---

# Il proprio lavoro precedente si cita, in terza persona

Il 21 ago 2026 il paper ASPLOS ("The Lock-Step Tax") non citava da nessuna parte il paper HPEC 2026
accettato ("The Two Faces of Abstraction Regret"), che va su IEEE Xplore a fine settembre, cioè
**prima** della notifica ASPLOS del 21 dicembre. Il CFP esclude solo i workshop senza proceedings
archiviati, quindi HPEC conta come pubblicazione anteriore e la sovrapposizione va dichiarata.

La convenzione è: **citazione piena e normale, prosa in terza persona.** Non "our prior work", non
"reference removed", non un autore sbianchettato. Un revisore non può distinguere una self-citation
in terza persona da una citazione qualsiasi, mentre una voce anonimizzata **segnala** che è tua.

In `paper2/refs.bib` c'era già una voce orfana, mai citata da nessun `\cite`:

```bibtex
@misc{paper1, author = {{Anonymous}}, note = {Manuscript under review, 2026} }
```

sbagliata due volte: `{{Anonymous}}` è esattamente la formula che tradisce, e "under review" era
falso da quando il paper è stato accettato. Sostituita con `@inproceedings{twofaces2026}`, autore
reale, venue reale.

**Effetto collaterale da mettere in conto:** aggiungere una voce **rinumera la bibliografia**. Con
`ACM-Reference-Format` (ordine alfabetico) l'inserimento a metà sposta tutti i numeri successivi, e
i numeri cambiano anche nelle pagine che non hai toccato. Se il vincolo è "non toccare le prime due
pagine" perché il rapid review legge solo quelle, va verificato con un diff pagina per pagina che
**cambino solo i numeri di citazione** e non il testo:

```
diff <(pdftotext -layout old.pdf - | awk -v RS='\f' 'NR==1') \
     <(pdftotext -layout new.pdf - | awk -v RS='\f' 'NR==1')
```

Un termine tecnico coniato nel paper precedente (lì "regret") va **definito alla prima occorrenza**
nel nuovo, attaccando la citazione: è il punto in cui la sovrapposizione si dichiara da sé. Vedi
[[verify-citations-on-the-source]].
