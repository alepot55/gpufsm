# I watcher upstream, e perché stanno fuori dalla repo

Dal 18 ago 2026 la sorveglianza delle PR upstream è automatica, su **due livelli**. Niente di tutto
questo sta nella repo: vive sotto `~/.claude/`, quindi è **per-macchina** e su un altro PC non
esiste (come tutto ciò che sta lì — vedi [[memory-lives-in-the-repo]]). Questa nota esiste perché
altrimenti in una sessione futura i watcher sono invisibili: te ne accorgi solo quando arriva una
notifica, o peggio quando *non* arriva.

## Livello 1 — reattivo: `~/.claude/gpufsm-watch/watch-upstream.sh`

Un unico script che fa polling ogni 120s su **11 thread, due repo**:

- `triton-lang/triton`: PR #11324, #11325, #10766 + issue #11326, #11328
- `llvm/llvm-project`: PR #216947, #216853, #216605, #216851, #216854, #216852

Si arma col tool `Monitor` (`persistent: true`), e **ogni riga di stdout diventa una notifica dentro
la sessione in corso**: non un digest da leggere dopo, ma un risveglio con il contesto della
conversazione ancora caricato. Ha due modalità:

| modalità | chi la usa | comportamento |
|---|---|---|
| loop (default) | `Monitor` | gira all'infinito, una riga per evento |
| `GPUFSM_WATCH_ONESHOT=1` | il cron | un giro solo, esce con `rc=0` (~32s) |

Oltre ai commenti sorveglia **il gate di `docs/PR_LEDGER.md`**: *"non aprire la settima finché una
delle sei non atterra"*. Quando una PR LLVM atterra, lo script lo dice esplicitamente, perché quello
è il momento in cui la riserva #203858 diventa apribile.

## Livello 2 — durevole: i tre cron

| taskId | quando | cosa fa |
|---|---|---|
| `triton-review-watch` | 8:17 e 18:17 | stesso giro dello script, in one-shot, su tutti e 11 i thread |
| `asplos-2027-submission-watch` | 9:23 | se l'HotCRP di settembre è aperto + giorni alla deadline (9 set 2026 AoE) |
| `gpufsm-nightly-gate` | 2:13 | `ruff format --check`, `ruff check`, `mypy`, `pytest -m "not gpu"`; parla solo se rosso |

Stanno in `~/.claude/scheduled-tasks/<taskId>/SKILL.md`. Il nome `triton-review-watch` è ormai
bugiardo: copre anche LLVM.

## Perché due livelli e non uno

| | latenza | sopravvive alla sessione | contesto |
|---|---|---|---|
| `Monitor` + lo script | ~2 min | **no** | tutta la conversazione |
| cron | fino a 12h | sì | nessuno, riparte da zero |

**Il Monitor muore con la sessione, e muore spesso**: il 18 ago è successo due volte in un
pomeriggio. Quindi il cron non è un doppione, è l'unica cosa che regge quando l'app è chiusa — e a
sua volta, se l'app è chiusa alla scadenza, il cron parte al lancio successivo. Corollario da
ricordare: **un digest che manca non significa "nessuna novità", significa "non stava girando
niente"** ([[empty-output-is-not-a-result]]).

Un **webhook non è possibile**: su `triton-lang/triton` e `llvm/llvm-project` i permessi sono `pull`
only, quindi non si può registrare. "Reattivo" qui vuol dire polling stretto, punto — non inseguire
l'idea del push.

## Tutto report-only, e non per pigrizia

Nessun livello posta, pusha, apre PR o sottomette. È deliberato: rispondere a un revisore e premere
SUBMIT sono **decisioni**, non passi operativi, e [[be-autonomous-no-confirmations]] copre i secondi,
non le prime.

## Dettagli che si pagano se si dimenticano

- Lo stato è in `since`, `open-prs-v2`, `ci-seen` nella stessa cartella. Cancellare `since` fa
  ri-sparare tutto lo storico come nuovo; cancellare `ci-seen` fa ri-annunciare ogni run CI già visto.
- Gli autori che finiscono in `[bot]` sono filtrati, e così i messaggi di `alepot55`.
- Le review `COMMENTED` col body vuoto sono scartate: sono il contenitore dei commenti inline, che
  arrivano per conto loro, e tenerle raddoppiava ogni evento (5 righe vuote su 12 in un test).
- Dopo 5 poll falliti di fila esce `[WATCHER DEGRADATO]`. Senza, "nessuna notifica" e "l'API non
  risponde più" sono indistinguibili.
- Attenzione agli apostrofi italiani dentro i filtri jq: stanno in single quote bash, e un `gia'`
  chiude il quoting. Costato un syntax error.
- Per elencarli o cambiarli: sezione "Scheduled" nella sidebar dell'app, o i tool MCP
  `scheduled-tasks` (`list_scheduled_tasks`, `update_scheduled_task`, `delete_scheduled_task`).
