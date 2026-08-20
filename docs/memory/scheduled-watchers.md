# I watcher upstream, e perché stanno fuori dalla repo

Dal 18 ago 2026 la sorveglianza delle PR upstream è automatica, su **due livelli**. Niente di tutto
questo sta nella repo: vive sotto `~/.claude/`, quindi è **per-macchina** e su un altro PC non
esiste (come tutto ciò che sta lì — vedi [[memory-lives-in-the-repo]]). Questa nota esiste perché
altrimenti in una sessione futura i watcher sono invisibili: te ne accorgi solo quando arriva una
notifica, o peggio quando *non* arriva.

## Il lock: un solo poller alla volta

`Monitor` e cron eseguono lo **stesso** script e condividono `since`. Se pollano insieme se lo
sovrascrivono a vicenda e gli eventi in mezzo spariscono senza traccia. Un `flock` non bloccante su
`~/.claude/gpufsm-watch/.poller.lock` risolve: chi arriva secondo stampa `[SKIP]` ed esce con
`rc=0`, che e' la cosa giusta perche' significa che qualcun altro sta gia' guardando.

Provato anche un **servizio systemd utente** che facesse da poller permanente (18 ago 2026):
funzionava, ma e' stato smontato per scelta esplicita di tenere Claude Desktop aperto e restare su
`Monitor` semplici. Se un giorno il buco fra le sessioni tornasse a dare fastidio, la strada era
quella: unit con `Restart=always`, `ExecStart=/bin/bash -lc` (una shell di login eredita
`GITHUB_TOKEN` senza copiarlo in un `EnvironmentFile`) e `loginctl enable-linger`.

## Livello 1 — lo script: `tools/watch/upstream.sh`

Sta **in repo**, non più solo sotto `~/.claude/`: è l'unica copia, e il daemon la esegue da lì. In
`~/.claude/gpufsm-watch/` restano soltanto lo stato (`since`, `open-prs-v2`, `ci-seen`) e
`events.log`, che sono stato di macchina.

Un unico script che fa polling ogni 120s su **11 thread, due repo**:

- `triton-lang/triton`: PR #11324, #11325, #10766 + issue #11326, #11328
- `llvm/llvm-project`: PR #216947, #216853, #216605, #216851, #216854, #216852

Ogni riga di stdout è un evento: il daemon la accoda al log, e il `Monitor` la trasforma in **una
notifica dentro la sessione in corso**: non un digest da leggere dopo, ma un risveglio con il contesto della
conversazione ancora caricato. Ha due modalità:

| modalità | chi la usa | comportamento |
|---|---|---|
| loop (default) | il daemon | gira all'infinito, una riga per evento |
| `GPUFSM_WATCH_ONESHOT=1` | il cron | un giro solo, esce con `rc=0` (~32s) |

Oltre ai commenti sorveglia **il gate di `docs/PR_LEDGER.md`**: *"non aprire la settima finché una
delle sei non atterra"*. Quando una PR LLVM atterra, lo script lo dice esplicitamente, perché quello
è il momento in cui la riserva #203858 diventa apribile.

## Livello 2 — rete di sicurezza: i tre cron

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
| daemon + `Monitor` che ne segue il log | ~2 min | **sì**, il daemon; il Monitor no | tutta la conversazione |
| cron | fino a 12h | sì | nessuno, riparte da zero |

Da quando il polling sta nel daemon, il buco cieco fra due sessioni è chiuso. Il cron resta come
terzo livello per il caso in cui il **daemon** sia giù (macchina spenta, unit fallita): non è un
doppione. Regola che resta valida a ogni livello: **un digest che manca non significa "nessuna
novità", significa "non stava girando niente"** — controllare
`systemctl --user is-active gpufsm-watch` prima di concludere che è tutto tranquillo
([[empty-output-is-not-a-result]]).

Un **webhook non è possibile**: su `triton-lang/triton` e `llvm/llvm-project` i permessi sono `pull`
only, quindi non si può registrare. "Reattivo" qui vuol dire polling stretto, punto — non inseguire
l'idea del push.

## Tutto report-only, e non per pigrizia

Nessun livello posta, pusha, apre PR o sottomette. È deliberato: rispondere a un revisore e premere
SUBMIT sono **decisioni**, non passi operativi, e [[be-autonomous-no-confirmations]] copre i secondi,
non le prime.

## ⚠️ Esiste una seconda famiglia di watcher, in `~/.cache/`

`watch_asplos.sh`, `watch_triton.sh`, `watch_llvm.sh`, `watch_upstream.sh`: scritti da **sessioni
parallele**, non da questa. `~/.cache/watch_upstream.sh` è nato alle 11:42 del 18 ago, sei minuti
prima di quello in `tools/watch/`, e fa in gran parte la stessa cosa. Nessuno di loro era vivo al
controllo delle 15:30 (log a 0 byte, nessun processo, nessun crontab): erano processi di sessione,
morti con la sessione che li aveva lanciati.

Prima di scrivere un watcher nuovo, **guardare lì**. `watch_asplos.sh` in particolare è migliore di
quanto scriverei da zero: sorveglia la pagina del CFP invece di indovinare l'URL HotCRP, tiene un
elenco dei link già visti, ha un secondo segnale testuale indipendente dai link, e un file di
heartbeat perché "il silenzio del log è indistinguibile dalla morte dello script" — un guasto che
era già costato una serata. È quello armato oggi per ASPLOS.

Da consolidare: o si portano in `tools/watch/` come si è fatto con `upstream.sh`, o si cancellano.
Due stack che sorvegliano le stesse cose sono il modo di credersi coperti mentre nessuno guarda.

## La lista dei bersagli non si aggiorna da sola

I numeri di PR sorvegliati sono **scritti a mano** in `tools/watch/upstream.sh`. Il 19 ago ho aperto
llvm#217392 e non l'ho aggiunta: l'approvazione di `matthias-springer` del mattino dopo **non ha
prodotto nessuna notifica**, e l'ho scoperta solo perché l'utente ha chiesto di ricontrollare. Nello
stesso momento la lista conteneva ancora #216851, già mergiata, cioè sprecava una chiamata API per
sorvegliare una PR morta.

Regola: **aprire o chiudere una PR upstream include modificare `TARGETS`.** Il watcher tace sia
quando non succede niente sia quando guarda dalla parte sbagliata, e i due silenzi sono identici
([[empty-output-is-not-a-result]]).

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
