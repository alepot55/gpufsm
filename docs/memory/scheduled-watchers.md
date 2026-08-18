# Tre watcher schedulati, fuori dalla repo

Dal 18 ago 2026 girano tre task schedulati di Claude Code. **Non stanno nella repo**: vivono in
`~/.claude/scheduled-tasks/<taskId>/SKILL.md` (per-macchina, come tutto ciò che sta sotto `~/.claude` —
vedi [[memory-lives-in-the-repo]]). Questa nota esiste perché altrimenti in una sessione futura sono
invisibili: si scopre che esistono solo quando arriva una notifica.

| taskId | quando | cosa fa |
|---|---|---|
| `triton-review-watch` | 8:17 e 18:17 | nuovi commenti/review/CI su #11324, #11325, #10766, #11326, #11328 |
| `asplos-2027-submission-watch` | 9:23 | se il sito HotCRP di settembre è aperto + giorni alla deadline (9 set 2026 AoE) |
| `gpufsm-nightly-gate` | 2:13 | `ruff format --check`, `ruff check`, `mypy`, `pytest -m "not gpu"`; riporta solo se rosso |

Tre vincoli che ne determinano il valore:

1. **Sono tutti report-only.** Non postano upstream, non committano, non sottomettono. È deliberato:
   una risposta a un maintainer e la pressione del bottone SUBMIT sono decisioni, non passi operativi,
   e [[be-autonomous-no-confirmations]] copre i secondi, non le prime.
2. **Girano solo mentre l'app è aperta.** Se è chiusa quando il task scade, parte al lancio successivo.
   Un digest che manca non vuol dire "nessuna novità": vuol dire che il portatile era spento.
3. **Il watcher Triton tiene lo stato in `~/.claude/gpufsm-watch/triton-last-seen.json`.** Cancellarlo
   fa ri-segnalare tutto lo storico come nuovo. Il dedup è sugli id dei commenti, e ignora quelli di
   `alepot55`.

Il watcher Triton è quello che paga: il maintainer risponde in ore, non in giorni
([[upstream-review-dynamics]]), e ogni push azzera l'approvazione della CI
([[triton-ci-needs-maintainer-approval]]), quindi sapere *subito* cosa è arrivato decide se si pusha
una volta sola o due.

## Il livello reattivo: `~/.cache/watch_triton.sh`

12 ore di latenza sono troppe per una review che si muove in ore, quindi sopra il cron c'è un
secondo livello: lo script qui sopra (`~/.claude/gpufsm-watch/` tiene invece lo stato del
cron), armato con il tool `Monitor` (`persistent: true`). Fa polling
ogni 120s e **ogni riga di stdout diventa una notifica dentro la sessione in corso** — non un digest
da leggere dopo, ma un risveglio con tutto il contesto della conversazione ancora caricato.

I due livelli non sono un doppione:

| | latenza | sopravvive alla sessione | contesto |
|---|---|---|---|
| `Monitor` + `watch-triton.sh` | ~2 min | **no**, muore con la sessione | tutta la conversazione |
| `triton-review-watch` (cron) | fino a 12h | sì | nessuno, riparte da zero |

Un **webhook non è possibile**: su `triton-lang/triton` i permessi sono `pull` only, quindi non si
può registrare. "Reattivo" qui significa polling stretto, punto — non inseguire l'idea del push.

Dettagli che si pagano se si dimenticano:

- Lo stato è in `since`, `open-prs`, `ci-seen` nella stessa cartella. Cancellare `since` fa
  ri-sparare tutto lo storico; cancellare `ci-seen` fa ri-annunciare ogni run CI già visto.
- Le review `COMMENTED` col body vuoto sono filtrate via: sono solo il contenitore dei commenti
  inline, che arrivano per conto loro, e tenerle raddoppiava ogni evento (5 righe vuote su 12 nel
  test contro l'attività del 17-18 ago).
- Lo script emette una riga `[WATCHER DEGRADATO]` dopo 5 poll falliti di fila. Serve: senza,
  "nessuna notifica" e "l'API non risponde più" sono indistinguibili — vedi
  [[empty-output-is-not-a-result]].
- Attenzione agli apostrofi italiani nei commenti dentro i filtri jq: sono in single quote bash e
  un `gia'` chiude il quoting. Costato un syntax error.

Per elencarli o cambiarli, dall'app: sezione "Scheduled" in sidebar, oppure i tool MCP
`scheduled-tasks` (`list_scheduled_tasks`, `update_scheduled_task`, `delete_scheduled_task`).
