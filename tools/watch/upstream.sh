#!/usr/bin/env bash
# Watcher reattivo su TUTTE le PR upstream aperte di alepot55: Triton + LLVM.
# Una riga di stdout = un evento. Sola lettura: non posta, non pusha, non apre, non chiude nulla.
#
# Oltre ai commenti sorveglia il gate di PR_LEDGER.md:243 -- "non aprire la settima finche' una
# delle sei non atterra". Quando una PR LLVM atterra, lo dice esplicitamente.

set -uo pipefail

STATE="$HOME/.claude/gpufsm-watch"
mkdir -p "$STATE"
SINCE_FILE="$STATE/since"
OPEN_FILE="$STATE/open-prs-v2"
CI_FILE="$STATE/ci-seen"

ME="alepot55"
INTERVAL="${GPUFSM_WATCH_INTERVAL:-120}"

# "repo:numero" -- il registro sta in docs/PR_LEDGER.md, questo e' solo il bersaglio del polling.
TARGETS="
triton-lang/triton:11325
triton-lang/triton:10766
triton-lang/triton:11326
triton-lang/triton:11328
triton-lang/triton:11393
triton-lang/triton:11396
llvm/llvm-project:216947
llvm/llvm-project:216853
llvm/llvm-project:216605
llvm/llvm-project:217392
llvm/llvm-project:216854
llvm/llvm-project:216852
"

# Un solo poller alla volta. Monitor e cron condividono "since": se pollano insieme se lo
# sovrascrivono a vicenda e gli eventi in mezzo spariscono. Chi arriva secondo non fa niente --
# e va bene cosi', perche' vuol dire che qualcun altro sta gia' guardando.
LOCK="$STATE/.poller.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[SKIP] un altro poller ha gia' il lock: questo giro non serve" >&2
  exit 0
fi

[ -f "$SINCE_FILE" ] || date -u +%Y-%m-%dT%H:%M:%SZ >"$SINCE_FILE"
[ -f "$CI_FILE" ] || : >"$CI_FILE"
if [ ! -f "$OPEN_FILE" ]; then
  echo "$TARGETS" | tr -d ' ' | grep -v '^$' | sort >"$OPEN_FILE"
fi

fails=0

while true; do
  since=$(cat "$SINCE_FILE")
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ok=1
  still_open=""

  for target in $TARGETS; do
    repo=${target%%:*}
    n=${target##*:}
    short=${repo##*/}

    # commenti nel thread principale
    if body=$(gh api "repos/$repo/issues/$n/comments?since=$since" 2>/dev/null); then
      jq -r --arg me "$ME" --arg n "$n" --arg r "$short" '
        .[] | select(.user.login != $me)
        | select(.user.login | test("\\[bot\\]$") | not)
        | "[\($r) #\($n) commento] \(.user.login): \(.body | gsub("\\s+"; " ") | .[0:280])  \(.html_url)"
      ' <<<"$body" 2>/dev/null
    else ok=0; fi

    # le issue non hanno commenti inline ne' review: salta il resto
    kind=$(gh api "repos/$repo/issues/$n" --jq 'if .pull_request then "pr" else "issue" end' 2>/dev/null) || { ok=0; continue; }
    if [ "$kind" = "issue" ]; then
      still_open="$still_open$repo:$n
"
      continue
    fi

    # commenti inline: e' qui che si chiedono le modifiche
    if body=$(gh api "repos/$repo/pulls/$n/comments?since=$since" 2>/dev/null); then
      jq -r --arg me "$ME" --arg n "$n" --arg r "$short" '
        .[] | select(.user.login != $me)
        | select(.user.login | test("\\[bot\\]$") | not)
        | "[\($r) #\($n) INLINE \(.path):\(.line // .original_line // 0)] \(.user.login): \(.body | gsub("\\s+"; " ") | .[0:280])  \(.html_url)"
      ' <<<"$body" 2>/dev/null
    else ok=0; fi

    # review formali; scarta le COMMENTED col body vuoto (sono il contenitore degli inline)
    if body=$(gh api "repos/$repo/pulls/$n/reviews" 2>/dev/null); then
      jq -r --arg me "$ME" --arg n "$n" --arg r "$short" --arg since "$since" '
        .[] | select(.user.login != $me) | select((.submitted_at // "") > $since)
        | select(.state != "COMMENTED" or ((.body // "") | gsub("\\s+"; "") | length) > 0)
        | "[\($r) #\($n) REVIEW \(.state)] \(.user.login): \(.body // "" | gsub("\\s+"; " ") | .[0:280])  \(.html_url)"
      ' <<<"$body" 2>/dev/null
    else ok=0; fi

    # atterraggio: e' il gate che sblocca la PR di riserva
    st=$(gh api "repos/$repo/pulls/$n" --jq 'if .merged then "MERGIATA" elif .state=="closed" then "CHIUSA" else "open" end' 2>/dev/null) || { ok=0; continue; }
    if [ "$st" = "open" ]; then
      still_open="$still_open$repo:$n
"
    elif grep -q "^$repo:$n\$" "$OPEN_FILE"; then
      echo "[$short #$n $st] https://github.com/$repo/pull/$n -- aggiornare docs/PR_LEDGER.md"
      if [ "$repo" = "llvm/llvm-project" ] && [ "$st" = "MERGIATA" ]; then
        echo "[GATE SBLOCCATO] una PR LLVM e' atterrata: la riserva #203858 (scf::loopUnrollByFactor, gia' implementata e verificata) si puo' aprire -- vedi docs/PR_LEDGER.md"
      fi
    fi
  done

  # CI in attesa di approvazione umana (solo Triton: su LLVM gira da sola)
  if runs=$(gh api "repos/triton-lang/triton/actions/runs?status=action_required" \
      --jq '.workflow_runs[] | select(.head_repository.owner.login=="'"$ME"'") | "\(.id)|\(.name)|\(.head_branch)"' 2>/dev/null); then
    while IFS='|' read -r id name branch; do
      [ -z "$id" ] && continue
      grep -qx "$id" "$CI_FILE" && continue
      echo "$id" >>"$CI_FILE"
      echo "[triton CI action_required] $name su $branch -- serve un maintainer  https://github.com/triton-lang/triton/actions/runs/$id"
    done <<<"$runs"
  else ok=0; fi

  if [ "$ok" = 1 ]; then
    echo "$now" >"$SINCE_FILE"
    printf '%s' "$still_open" | grep -v '^$' | sort >"$OPEN_FILE"
    fails=0
  else
    fails=$((fails + 1))
    # il silenzio non deve somigliare a "nessuna novita'"
    if [ "$fails" -eq 5 ]; then
      echo "[WATCHER DEGRADATO] 5 poll falliti di fila sull'API GitHub -- non sto piu' vedendo niente"
    fi
  fi

  # Il cron esegue un giro solo e legge lo stdout; il Monitor gira all'infinito.
  if [ -n "${GPUFSM_WATCH_ONESHOT:-}" ]; then
    [ "$ok" = 1 ] || { echo "[WATCHER DEGRADATO] giro one-shot con almeno una chiamata fallita" >&2; exit 1; }
    exit 0
  fi

  sleep "$INTERVAL"
done
