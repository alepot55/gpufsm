# Le citazioni si verificano sulla fonte, non sulle note di uno sweep

`paper2/RELATED_WORK.md` apre dicendo che ogni ID era "verified on publisher/arXiv pages". Il
21 ago 2026, controllando le 29 voci di `refs.bib` una per una, `hopps2025` aveva **Yufeng Du**
come primo autore. È **Xingran Du** (ACM DL, ASPLOS'25). Venue e anno erano giusti, il che è il
motivo per cui era sopravvissuto: sembrava plausibile.

Nello stesso passaggio: `subwarp2022` aveva `author = {Damani, Sana and others}`, che
renderizza **"et al."** in una bibliografia dove `docs/SUBMISSION_ASPLOS.md` dichiarava
"full author names throughout (no et al.)".

## Cosa fare

Prima di sottomettere, per ogni voce: **aprire la pagina dell'editore o di arXiv** e confrontare
autori, venue, anno. Le note di un sweep precedente non sono una fonte, sono un ricordo.

Due controlli meccanici che costano nulla:

```bash
grep -c "and others" refs.bib                     # deve dare 0
pdftotext paper.pdf - | grep -c 'et al\.'          # deve dare 0
```

## Il rovescio: cercare anche cosa manca

Lo stesso sweep ha trovato che mancava **"Why GPUs are Slow at Executing NFAs and How to Make
them Faster"** (Liu, Pai, Jog, ASPLOS 2020) — il titolo più vicino al nostro, alla conferenza a
cui stiamo sottomettendo, degli stessi primi autori di AsyncAP che già citavamo. Un revisore
ASPLOS ci pensa in tre secondi.

Corollario: cercare per **titolo del proprio paper**, non solo per parole chiave dell'area.
Chi ha già scritto la frase che stai scrivendo è il revisore che ti tocca.
