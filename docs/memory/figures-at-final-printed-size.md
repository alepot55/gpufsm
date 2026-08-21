# Le figure vanno emesse alla dimensione finale, non riscalate

Il 21 ago 2026, preparando il paper ASPLOS, ho scoperto che tutte le figure di `paper2` erano
generate a `figsize=(6,4)` e incluse con `\includegraphics[width=\columnwidth]`. In una colonna
`acmart[sigplan]` (3.33 in) quello è un **riscalamento a 0.55×**: una label impostata a 9pt
arriva sulla pagina a **~5pt**, e il CFP ASPLOS vieta qualunque cosa sotto gli **8pt**.

Il punto interessante non è l'errore, è **perché il compliance check non l'aveva visto**:
cercava `\resizebox`. Il restringimento arrivava dal `width=`, che nessuno legge come una
riscalatura.

## La regola

Emettere ogni figura **alla dimensione a cui verrà stampata** e includerla a scala 1.0:

```python
COL = 3.33   # una colonna acmart[sigplan] su letter
WIDE = 7.0   # \textwidth, per figure*
```

Così un 8pt nel sorgente è un 8pt sulla pagina, e la verifica diventa banale. `paper2/figures.py`
lo dichiara in cima come "sizing contract" e va tenuto.

## Come verificarlo davvero

Non fidarsi della checklist: **leggere il log di build**. `acmart` avvisa da sé sulle figure
senza descrizione (`A possible image without description`), e quell'avviso era presente a ogni
build mentre `docs/SUBMISSION_ASPLOS.md` continuava a dichiarare che i `\Description` c'erano
tutti. La riscrittura li aveva persi tutti e otto in silenzio.

```bash
grep -ci 'without description' build.log   # deve dare 0
```

Stessa famiglia di [[verify-by-running-not-by-agent-verdict]]: il timbro non è la verifica.
