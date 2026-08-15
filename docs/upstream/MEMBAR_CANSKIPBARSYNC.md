# Upstream target: quanto costa ancora `canSkipBarSync` (Triton, membar NVIDIA)

Stato: **indagine aperta, 2026-08-15.** Obiettivo: un contributo upstream di sostanza (non docs) su
un problema che i maintainer hanno lasciato aperto e che sanno di avere.

## Perché questo e non un bug qualsiasi

Le issue piccole e ben scritte upstream vengono raccolte in giorni (verificato: `#10890` è stata
chiusa dal suo stesso autore con `#10891` in tre settimane; `#11228`, arrivata seconda sullo stesso
bug, è stata chiusa con "Solved by #10891"). Quello che manca upstream non è chi scrive fix piccoli:
è chi chiude i problemi **lasciati a metà**. Questo è uno di quelli, e sta esattamente sul nostro
terreno (barriere, memoria condivisa, misura e attribuzione).

## La storia, verificata alla fonte

| Data | Evento |
|------|--------|
| 2025-05-28 | Issue **#6960**: `NVIDIA::canSkipBarSync` esiste ma non è passato alla membar analysis. Raoux: *"looks like an oversight"*, poi *"feel free to send a PR"*. |
| 2025-08-13 | Raoux apre **#7846** (2 righe, collega il filtro). Un giorno dopo commenta: *"this somehow regress perf :("*. La PR è **ancora aperta e ferma da un anno**. |
| 2025-10-05 | lijinpei apre **#8374** (+400/-42) con benchmark su Hopper: le barriere ridondanti costano **~10%** su matmul warp-specialized. Mai revisionata. |
| 2026-01-22 | peterbell10 **#9246** "Resurrect canSkipBarSync": versione conservativa, collegata alla compilazione vera. Merged. Riporta +18 GB/s su MoE persistente bf16×mxfp4. |
| 2026-02-09 | peterbell10 **#9281** "Fully restore canSkipBarSync": versione aggressiva, speedup misurati su **GB200 e H100**. Merged. |
| 2026-02-13 | Raoux **#9459**: *"This seems to be causing some very sporadic hang, reverting while it gets figured out"*. **Revert.** Da allora nessuno l'ha ripreso. |

## La diagnosi (perché quella versione impiccava)

`#9281` non era sbagliata solo nel filtro: trattava `ttng.ArriveBarrierOp` come se **contenesse** una
barriera locale, cioè come un punto di sincronizzazione della CTA. Un `mbarrier.arrive` non
sincronizza la CTA: viene emesso da una **warp leader**. Se il leader segnala il ring buffer prima
che le altre warp della partizione abbiano finito la loro `wait`, il produttore riusa il buffer e le
fasi degli mbarrier divergono → **deadlock sporadico**, esattamente il sintomo del revert.

Questa diagnosi non è nostra congettura: è ciò che **Mogball ha poi implementato** in **#10035**
(15 apr 2026, *"[nvidia] Always insert bar sync before all mbarrier arrives"*), con nel corpo della PR
lo schema producer/consumer del deadlock. Su `main` oggi `containsLocalBarrier` include
`ArriveBarrierOp`, `BarrierExpectOp` e `TCGen5CommitOp` → **prima di ogni arrive c'è una barriera CTA**.

Nel frattempo sono rientrati altri pezzi del lavoro di febbraio:
- **#10675** (19 giu): niente barriere ridondanti tra wait di memoria back-to-back (`MemWaitOpTrait`)
  — copre `async_wait`/`async_tma_store_wait`, **non** `WaitBarrierOp`.
- **#11056** (28 lug): semantica di `WarpGroupDotWaitOp` con `num_warps > 4`, con l'attributo
  `warpGroupLocal` assegnato dalla membar analysis.

## Il buco che resta, in una riga

`NVIDIA::canSkipBarSync` su `main` ha **tre** regole (coppie di op mbarrier single-thread;
TMA-load → `WaitBarrier`; atomici commutativi identici). Mancano le rilassature di `#9281` sulle
coppie **`WaitBarrier` ↔ `AsyncTMACopyGlobalToLocal`/`AsyncTMAGather`**, che toccano la stessa
allocazione mbarrier ma il cui ordine tra thread è irrilevante perché è l'mbarrier stesso a ordinarle.

**La condizione che le rendeva insicure non esiste più**: il pericolo era che una warp corresse avanti
fino a un arrive; da #10035 ogni arrive è preceduto da una barriera CTA. Il caso `wait → wait` resta
escluso di proposito (una warp può correre avanti in una catena di wait) — ed è anche l'unico che
#9281 escludeva esplicitamente.

## Come si decide se vale una PR (criterio di kill dichiarato prima di misurare)

Il rischio onesto è che il guadagno sia **zero**: dato che `BarrierExpectOp` è già un punto di
barriera, nel loop TMA tipico una `bar.sync` cade comunque poco prima della copia, e la barriera che
vorremmo togliere potrebbe essere già assente. Quindi si misura prima e si scrive dopo:

1. **Baseline**: compilare un insieme di kernel rappresentativi (tutorial gluon persistence /
   tcgen05-mma-scaled, fused-attention, matmul warp-specialized) su `main` e contare le `bar.sync`
   nel PTX/SASS oltre al tempo di kernel.
2. **Patch**: aggiungere la sola coppia `WaitBarrier ↔ TMA copy/gather`, ricompilare, ricontare.
3. **Kill criterion**: se il delta di `bar.sync` è ~0 su tutti i kernel, **niente PR**; si scrive il
   risultato negativo nel thread di #6960/#7846 (che è comunque informazione che oggi manca a tutti).
4. Se il delta è reale: PR con lit test in `test/Analysis/test-membar-ttng.mlir`, run pulita del
   **concurrency sanitizer** (GSan) sui tutorial, e i numeri di prima/dopo su H100.

## Infrastruttura (fatta)

`scripts/modal_triton.py`: build di Triton da sorgente dentro un **Modal Volume** (build su CPU, dove
i minuti costano poco; test su GPU affittata), con ccache e checkout persistenti → la prima build si
paga una volta, le successive sono incrementali.

```bash
python scripts/modal_triton.py build --ref <sha> [--patch p.patch]
python scripts/modal_triton.py run --cmd "pytest -q python/test/unit/..." 
```

## Chi decide, se si arriva alla PR

`peterbell10` (autore di #9246/#9281/#11056) e `ThomasRaoux` (autore del revert e di #7846).
`Mogball` per #10035. Nessuno di questi è `ptillet`: non pingare lui.
