# Memoria di progetto — indice

La memoria di questo progetto sta **nella repo**, non nello store locale di Claude
(`~/.claude/projects/*/memory/`, che è per-macchina e sparisce con i worktree). Un file = un fatto.
Aggiungere qui una riga per ogni file nuovo.

- [Autonomia: non chiedere conferme](be-autonomous-no-confirmations.md) — PR, merge, install, infra: si fanno, non si propongono.
- [La memoria sta nella repo](memory-lives-in-the-repo.md) — perché `docs/memory/` e non lo store di Claude.
- [Token già in env sul portatile](laptop-tokens-in-env.md) — GitHub + Modal id/secret esportati; non stamparli.
- [git push e il credential helper](git-push-credential-helper.md) — il 403 su push era una credenziale vecchia in `store`, non il token.
- [Da locale l'API GitHub funziona](github-api-works-from-local.md) — il 403 upstream era il proxy delle sessioni cloud.
- [La CI di Triton va approvata a mano](triton-ci-needs-maintainer-approval.md) — ogni push azzera il verde.
- [Metodo per i contributi upstream](upstream-contribution-method.md) — cosa ha prodotto #11323 e #11324, e cosa è costato non farlo.
- [Harness di build Triton su Modal](triton-build-harness-on-modal.md) — alberi paralleli, FileCheck vero, trappole già pagate.
- [La config del linter deve seguire i rename](lint-config-must-track-renames.md) — `ruff --fix` ha riscritto `int(0)` nei kernel Warp perché le per-file-ignores puntavano ai path vecchi.
- [Trappole dell'harness Modal](modal-gpu-harness-gotchas.md) — rc=0 non vuol dire che abbia eseguito i tuoi comandi.
- [Triton chiude le PR "trivial"](triton-rejects-trivial-prs.md) — conta l'impatto misurato, non la dimensione; la coda issue è satura.
- [Scegliere bug non contesi](pick-uncontested-bugs-not-design-changes.md) — il collo di bottiglia upstream e' il filtro di valore, non l'attenzione: preferire i bug a cui risponde il verificatore.
- [LLVM obbliga a dichiarare l'AI](llvm-requires-ai-disclosure.md) — policy di ammissibilita': dichiarazione nella PR + un umano che sappia difendere la patch.
- [Mai aspettare un lavoro che non vedi](never-poll-a-job-you-cant-see.md) — git su volume di rete striscia, e il ciclo di attesa su un file mai scritto brucia ore in silenzio.
- [Verificare eseguendo, non col timbro dell'agente](verify-by-running-not-by-agent-verdict.md) — il revisore avversariale ne ha confermati 10 su 11; eseguendoli, 2 su 7 erano falsi.
- [PR LLVM: corte e distanziate](llvm-pr-register-short-and-staggered.md) — la prima parola di un maintainer sulla nostra prima PR revisionata è stata "slop"; 4 PR in 29 secondi è la firma di chi scarica volume.
- [Progetti OSS misurati (ago 2026)](oss-targets-measured-2026-08.md) — sette progetti sugli stessi numeri: restare su LLVM, riserva wasmtime, e il numero che squalifica ciascun altro.
- [Rispondere ai revisori subito](answer-reviewers-immediately.md) — tenere ferme le repliche in attesa di un ok e' costato 7 ore; la policy chiede un umano responsabile, non che digiti lui.
- [Dinamica di review upstream](upstream-review-dynamics.md) — cosa sblocca una PR ferma, e quando contraddire un maintainer.
- [Un output vuoto non e' un risultato](empty-output-is-not-a-result.md) — controllare stderr prima di concludere "non si puo' fare".
- [Tre watcher schedulati, fuori dalla repo](scheduled-watchers.md) — review upstream, deadline ASPLOS, gate notturno: dove vivono e perché sono report-only.

