# Barriera mancante: offset di subslice confrontati tra frame diversi (PR #11325)

**Stato 18 ago 2026:** [PR #11325](https://github.com/triton-lang/triton/pull/11325) aperta il 16 ago,
**+115/-3** su 3 file, head `06bf3850`, reviewer Jokeren + ptillet. Il criterio e' stato **riscritto
tre volte** durante la review: nessuna richiesta pendente da parte nostra.

## Il difetto

`AllocationSlice::subsliceOffsets` (`lib/Analysis/Membar.cpp:23-30`) registra gli offset di un
`memdesc_subslice` **nelle coordinate della forma e del tipo elemento da cui il subslice e' preso**.
`intersects` (`Membar.cpp:62-71`) poi confronta gli offset di due slice direttamente, senza
verificare che siano espressi nello stesso frame.

Un `memdesc_reinterpret` in mezzo cambia quanti byte vale una riga. Quindi:

- riga 16 di un buffer `32x16xf16` = byte [512, 1024)
- riga 8 dello stesso buffer visto come `16x32xf16` = byte [512, 1024)

Stessi byte, ma il confronto vede `[16,32)` contro `[8,16)` e conclude "disgiunti" → **nessuna
barriera** tra una `local_store` e una `local_load` sugli stessi byte → corsa tra warp.

## Perche' e' pubblicabile (a differenza di #11323/#11324)

E' una barriera **mancante**, cioe' un errore di correttezza silenzioso, non una barriera di troppo.
Per il criterio dei maintainer ([[triton-rejects-trivial-prs]]) questo non e' mai "trivial".
E si raggiunge dall'**API pubblica di Gluon** (`allocate_shared_memory` + `.reinterpret` + `.slice`,
5 righe), non da MLIR scritto a mano — cosa che conta, perche' Raoux aveva chiuso la nostra #10785
con "either this path should never happen or we should make it work".

⚠️ Onesta' sul rischio: la corsa e' **latente**. 2000 lanci non hanno mostrato corruzione, e ConSan
non modella le corse fra warp dentro un CTA (`ConcurrencySanitizer.cpp:86-93`) — per progetto.
L'argomento che regge e' che **l'analisi e' incoerente con se stessa**: lo stesso accesso senza
reinterpret in mezzo riceve la barriera.

## Come e' stato trovato

Workflow di audit (`wf_97378bab-2c6`): 1 agente mappa come l'analisi scopre gli effetti di memoria,
5 agenti cercano in parallelo su famiglie diverse di operazioni, poi ogni candidato passa a un
revisore avversariale con mandato di **refutarlo**. 7 verdetti, 3 non refutati, 1 verificato a mano
e risultato vero. Gli altri 2 sono diventati le issue #11326 (confine di chiamata) e #11328
(multicast di cluster).

## La correzione, e le due che sono state scartate

| ver | criterio | perche' e' caduta |
|---|---|---|
| v1 | `accessTy.getAllocShape()` coincide | **mezza correzione**: un reinterpret che cambia solo il *tipo elemento* lascia la forma invariata. Verificato: 0 barriere prima e dopo. |
| v2 | la *Value* sorgente del subslice coincide | **falso positivo**: due reinterpret identici sono Value diverse, e la prova di disgiunzione si perdeva. Jokeren: "I don't think this is the right fix". Verificato su ws7: 1 barriera dove 0 e' corretto. |
| v3 ✅ | il *Type* della sorgente del subslice coincide | il tipo porta forma **e** tipo elemento; due viste separate ma identiche dello stesso buffer restano confrontabili. |

In pratica, accanto al guardiano che gia' protegge la prova sugli indici di buffer
(`Membar.cpp:40`): confrontare gli offset solo se `bufferId` coincide **e** `subsliceSrcTy`
coincide. E' conservativa: puo' solo aggiungere barriere, mai toglierne.

⚠️ La meta' `bufferId` del guardiano **non e' coperta da test**: servirebbe che l'allocatore
sovrapponga due buffer con vite disgiunte, e un test cosi' dipende dal piazzamento dell'allocatore
(fragilita' gia' segnalata in review la notte prima). Dichiarato apertamente nella descrizione della
PR, con offerta di separarla.

## Verifica fatta (non dedotta)

1. **Riproduzione MLIR** con controllo negativo accanto: stesso accesso *senza* reinterpret → la
   barriera c'e'; *con* reinterpret → non c'e'. Isola la causa.
2. **Riproduzione end-to-end da Gluon**: kernel Python di 5 righe → TTGIR → membar → nessuna barriera.
3. **I test nuovi falliscono su `main` e passano con la patch** (verificato eseguendo i due binari).
   Sono quattro, e due sono controlli negativi (`CHECK-NOT`) che pinnano il fatto che la patch **non**
   aggiunge barriere dove non servono:
   `@subslice_offsets_after_reinterpret` (forma), `@subslice_offsets_after_reinterpret_element_type`
   (f32→f16), `@subslice_offsets_same_shape_stay_disjoint`, `@subslice_offsets_through_identical_views`
   (il caso di Jokeren).
4. **Costo zero misurato**: 2203 `bar.sync` su main, 2203 con la patch, su 107 kernel.
5. **Nessuna regressione**: `test/Analysis`, `test/TritonGPU`, `test/Conversion`,
   `test/TritonNvidiaGPU`, `test/NVWS` → insieme dei falliti identico prima e dopo.
6. `pre-commit run --from-ref HEAD~1 --to-ref HEAD` → tutto verde (clang-format incluso).

## Alberi Modal

`--tree ws4` = questa patch. `--tree ws7` = il controllo che ha smascherato il falso positivo della
v2. `--tree main` = baseline. Confronto senza rebuild, vedi [[triton-build-harness-on-modal]].
