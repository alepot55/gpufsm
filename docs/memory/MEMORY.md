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
