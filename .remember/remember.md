# Handoff — 2026-08-18 mattina (portatile)

## In una riga

La review upstream si e' aperta: **Jokeren ha lasciato 5 interventi in 3 ore il 17 ago**, tutti evasi.
Nessuna azione pendente da parte nostra su nulla. Il merge dipende da loro.

## Stato upstream (verificato con `gh api` il 18 ago 08:00, non a memoria)

| | diff | stato | in attesa di |
|---|---|---|---|
| [#11324](https://github.com/triton-lang/triton/pull/11324) | +22/-1 | aperta, 4 richieste evase | approvazione CI + merge |
| [#11325](https://github.com/triton-lang/triton/pull/11325) | +115/-3 | aperta, riscritta 3 volte | risposta di Jokeren |
| [#10766](https://github.com/triton-lang/triton/pull/10766) | +75/-1 | **sbloccata** il 18 ago | review di peterbell10 |
| [#11326](https://github.com/triton-lang/triton/issues/11326) · [#11328](https://github.com/triton-lang/triton/issues/11328) | — | issue con riproduttori | triage |
| #11311 | — | **mergiata** 15 ago | — |
| #11323 | — | **chiusa**: "micro optimization" | — |

Dettaglio completo dei quattro giri di review: `docs/PR_LEDGER.md` (sezione 17-18 ago).

## Cosa ha sbloccato la situazione (riusabile)

`docs/memory/upstream-review-dynamics.md`. In sintesi:

1. **Rispondere DENTRO il thread inline.** Un commento nuovo in cima non fa riemergere un thread
   risolto. #10766 e' rimasta ferma sei settimane perche' avevamo chiuso noi il thread di peterbell10.
2. **Eseguire ogni richiesta, anche cosmetica** ("Why adding an empty new line?").
3. **Ma quando ha torto, mostrare l'output, non argomentare.** Su "your test checks nothing" ho
   incollato l'IR prima/dopo → ha chiesto di **rimettere** il test.
4. **Verificare la sua proposta**: su #11324 era migliore della nostra, e la nostra era incompleta.
5. **La direzione e' verso il piccolo**: da +81/-4 a +2/-1 di codice.

## Trappole pagate (non ripagarle)

- ⚠️ **Un output vuoto non e' un risultato negativo.** `2>/dev/null` mi ha nascosto un nome SSA
  invalido e ho concluso pubblicamente "non si puo' fare". Era falso.
  → `docs/memory/empty-output-is-not-a-result.md`
- ⚠️ **Ri-misurare TUTTE le build con lo stesso comando nella stessa run.** I numeri assoluti
  pubblicati (2356/2326) erano incoerenti perche' confrontavo conteggi di script diversi.
- ⚠️ **Benchmark: scegliere il kernel dove l'effetto puo' esistere.** Il primo era
  `matmul_tma_ws_kernel`, dove la warp specialization gira una volta per programma.
- ⚠️ Confrontare la **lista** dei test falliti, non il numero.
- ⚠️ I worktree git vanno in `~/.cache/tritonwt/`, non in `/tmp` (tmpfs da 2 GB).

## Alberi Modal

`main`=baseline · `ws3`=#11324 v1 · `ws4`=#11325 v1 · `ws6`=#11324 v2 · `ws7`=#11325 v2 ·
`ws8`=fold-split-join · `ws9`/`ws11`=#11324 con test · `ws10`=#11325 v3 (tipo sorgente).

## Monitor

`~/.cache/watch_triton.sh` — commenti inline, review, stato merge, CI; ogni 2 min; ignora i nostri
stessi commenti e i transitori `unknown`. Va riavviato a ogni sessione (`Monitor` persistente).

## Aperto

- **Niente da fare upstream**: rispondere in fretta se commentano. Non pushare senza una richiesta.
- ASPLOS 2027: pacchetto pronto, bloccante esterno (HotCRP di settembre non aperto).
  Deadline **9 set 2026 AoE**. Premere davvero SUBMIT.
