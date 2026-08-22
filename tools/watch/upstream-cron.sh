#!/bin/bash
# Rete di sicurezza per il watcher upstream: un giro one-shot, pensato per crontab.
#
# Il Monitor in sessione polla ogni 120s ma muore con la sessione, e il 22 ago 2026
# e' successo: sette ore scoperte. Questo giro sopravvive alla sessione. Il flock
# dentro upstream.sh fa si' che quando il Monitor e' vivo questo esca subito con
# [SKIP], che e' la risposta giusta: qualcun altro sta gia' guardando.
#
# Le righe [SKIP] restano nel log di proposito: sono il battito. Un log fermo
# significa cron morto, non "nessuna novita'" -- vedi docs/memory/empty-output-is-not-a-result.md.
set -uo pipefail

REPO=/home/alepot55/Desktop/projects/gpufsm
LOG="$HOME/.claude/gpufsm-watch/events.log"

out=$(cd "$REPO" && GPUFSM_WATCH_ONESHOT=1 bash tools/watch/upstream.sh 2>&1)
[ -n "$out" ] && printf '%s %s\n' "$(date -Is)" "$out" >>"$LOG"

# Notifica solo sugli eventi veri: [SKIP] e le diagnostiche non svegliano nessuno.
real=$(grep -vE '^\[SKIP\]|^\[WATCHER' <<<"$out" | grep -E 'triton|llvm-project' | head -3)
if [ -n "$real" ] && command -v notify-send >/dev/null; then
  notify-send -u normal "upstream: movimento" "$real"
fi
exit 0
