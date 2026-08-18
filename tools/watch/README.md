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

## Installazione del daemon

```
cp tools/watch/gpufsm-watch.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now gpufsm-watch
loginctl enable-linger "$USER"     # sopravvive al logout
```

Lo stato (`since`, `open-prs-v2`, `ci-seen`) e il log (`events.log`) restano in
`~/.claude/gpufsm-watch/`: sono stato di macchina, non codice.

⚠️ **Non lanciarlo a mano mentre il servizio gira**: condividono `since` e due poller se lo
sovrascrivono a vicenda, perdendo eventi. Prima `systemctl --user stop gpufsm-watch`.

Il token non sta in nessun file: l'unit usa una shell di login, che legge `GITHUB_TOKEN`
dall'ambiente dell'utente. Contesto in `docs/memory/scheduled-watchers.md`.
