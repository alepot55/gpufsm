# tools/watch — watcher delle PR upstream

`upstream.sh` fa polling sulle PR/issue aperte su `triton-lang/triton` e `llvm/llvm-project` e stampa
**una riga per evento nuovo**. Sola lettura: non posta, non pusha, non apre e non chiude nulla.

Sorveglia anche il gate di `docs/PR_LEDGER.md` ("non aprire la settima finché una delle sei non
atterra"): quando una PR LLVM atterra lo dice esplicitamente.

## Due modi di eseguirlo

| | comando | uso |
|---|---|---|
| loop | `bash tools/watch/upstream.sh` | il daemon systemd; un giro ogni 120s |
| one-shot | `GPUFSM_WATCH_ONESHOT=1 bash tools/watch/upstream.sh` | il cron; un giro solo, ~32s, `rc=0` |

## Come viene eseguito

Dal tool `Monitor` di Claude Code, che tiene lo script in loop per tutta la durata della sessione, e
dal cron `triton-review-watch` in one-shot come rete di sicurezza. Il `Monitor` muore con la
sessione: se la sessione e' chiusa, l'unica copertura e' il cron.

Lo stato (`since`, `open-prs-v2`, `ci-seen`) resta in `~/.claude/gpufsm-watch/`: e' stato di
macchina, non codice.

Un **lock** (`~/.claude/gpufsm-watch/.poller.lock`) garantisce un solo poller alla volta. Se il cron
scatta mentre il `Monitor` sta girando, vede il lock e salta il giro con `rc=0`: e' corretto, quel
giro lo sta gia' facendo il `Monitor`. Per la stessa ragione lanciarlo a mano durante una sessione
non fa danni, semplicemente non fa nulla.

Il token non sta in nessun file: l'unit usa una shell di login, che legge `GITHUB_TOKEN`
dall'ambiente dell'utente. Contesto in `docs/memory/scheduled-watchers.md`.
