# Barriera mancante: offset di subslice confrontati tra forme diverse (PR #11325)

**Stato:** [PR #11325](https://github.com/triton-lang/triton/pull/11325) aperta il 16 ago 2026,
+57/-0 su 2 file, reviewer Jokeren + ptillet.

## Il difetto

`AllocationSlice::subsliceOffsets` (`lib/Analysis/Membar.cpp:23-30`) registra gli offset di un
`memdesc_subslice` **nelle coordinate della forma da cui il subslice e' preso**. `intersects`
(`Membar.cpp:62-71`) poi confronta gli offset di due slice direttamente, senza verificare che siano
espressi nella stessa forma.

Un `memdesc_reinterpret` in mezzo cambia quanti byte vale una riga. Quindi:

- riga 16 di un buffer `32x16xf16` = byte [512, 1024)
- riga 8 dello stesso buffer visto come `16x32xf16` = byte [512, 1024)

Stessi byte, ma il confronto vede `[16,32)` contro `[8,16)` e conclude "disgiunti" → **nessuna
barriera** tra una `local_store` e una `local_load` sugli stessi byte → corsa tra warp.

## Perche' e' pubblicabile (a differenza di #11323/#11324)

E' una barriera **mancante**, cioe' un errore di correttezza silenzioso, non una barriera di troppo.
Per il criterio dei maintainer ([[triton-rejects-trivial-prs]]) questo non e' mai "trivial".
E si raggiunge dall'**API pubblica di Gluon**, non da MLIR scritto a mano — cosa che conta, perche'
Raoux aveva chiuso la nostra #10785 con "either this path should never happen or we should make it work".

## Come e' stato trovato

Workflow di audit (`wf_97378bab-2c6`): 1 agente mappa come l'analisi scopre gli effetti di memoria,
5 agenti cercano in parallelo su famiglie diverse di operazioni, poi ogni candidato passa a un
revisore avversariale con mandato di **refutarlo**. 7 verdetti, 3 non refutati, 1 verificato a mano
e risultato vero. Gli altri 2 (multicast TMA cross-CTA, confine di chiamata di funzione) restano da
verificare: vedi il journal del workflow.

## Verifica fatta (non dedotta)

1. **Riproduzione MLIR** con controllo negativo accanto: stesso accesso *senza* reinterpret → la
   barriera c'e'; *con* reinterpret → non c'e'. Isola la causa.
2. **Riproduzione end-to-end da Gluon**: kernel Python di 5 righe → TTGIR → membar → nessuna barriera.
3. **Il test nuovo fallisce su `main` e passa con la patch** (verificato eseguendo i due binari).
4. **Nessuna regressione**: `test/Analysis`, `test/TritonGPU`, `test/Conversion`,
   `test/TritonNvidiaGPU`, `test/NVWS` → insieme dei falliti identico prima e dopo.
5. `pre-commit run --from-ref HEAD~1 --to-ref HEAD` → tutto verde (clang-format incluso).

## La correzione

Stesso guardiano che gia' protegge la prova sugli indici di buffer (`Membar.cpp:40`): confrontare gli
offset solo se `bufferId` coincide **e** `accessTy.getAllocShape()` coincide. E' conservativa: puo'
solo aggiungere barriere, mai toglierne.

⚠️ La meta' `bufferId` del guardiano **non e' coperta da test**: servirebbe che l'allocatore
sovrapponga due buffer con vite disgiunte, e un test cosi' dipende dal piazzamento dell'allocatore
(fragilita' gia' segnalata in review la notte prima). Dichiarato apertamente nella descrizione della
PR, con offerta di separarla.

## Alberi Modal

`--tree ws4` = questa patch. `--tree main` = baseline. Confronto senza rebuild, vedi
[[triton-build-harness-on-modal]].
