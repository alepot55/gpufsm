---
name: memory-lives-in-the-repo
description: La memoria di progetto va scritta in docs/memory/ dentro la repo, non nello store locale di Claude
metadata:
  type: feedback
---

Richiesta esplicita del 2026-08-15: la memoria di progetto sta **in `docs/memory/`**, versionata con
il codice. Lo store per-macchina (`~/.claude/projects/*/memory/`) tiene solo un puntatore.

**Why:** il progetto si lavora da tre posti (PC, portatile, sessioni cloud) e i worktree hanno store
separati che spariscono. Solo la repo è la stessa ovunque, ed è già dove vivono `CLAUDE.md` e i
`docs/`. Memoria fuori dalla repo = memoria che si perde e che l'utente non può leggere né correggere.

**How to apply:** un file = un fatto, con lo stesso frontmatter di prima, più una riga in
`docs/memory/MEMORY.md`. I fatti grossi di stato progetto restano dove sono già (`CLAUDE.md`,
`docs/PR_LEDGER.md`): `docs/memory/` è per i fatti trasversali e per le preferenze di lavoro.
Vedi [[be-autonomous-no-confirmations]].
