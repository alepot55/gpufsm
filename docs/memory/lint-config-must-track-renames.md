---
name: lint-config-must-track-renames
description: Le per-file-ignores di ruff che puntano a nomi vecchi lasciano che --fix rompa i kernel Warp in silenzio
metadata:
  type: project
---

Durante il refactoring del 2026-08-16 i backend GPU sono stati spostati
(`backends/warp_backend.py` → `backends/warp/{nfa,dfa}.py`), ma le `per-file-ignores` in
`pyproject.toml` continuavano a nominare i path vecchi. Al primo `ruff check --fix` la regola
`UP018` ha riscritto `int(0)` in `0` dentro i `@wp.kernel`.

**Why:** in Warp `int(0)` non è una conversione ridondante, **dichiara un locale mutabile
`wp.int32`**. Con un letterale nudo, Warp miscompila le riassegnazioni condizionali successive:
il kernel produce risultati sbagliati senza errori di compilazione. È invisibile su qualunque
macchina senza GPU, quindi né la CI né i test locali lo avrebbero preso — solo un confronto con
l'oracolo su hardware.

**How to apply:**
- Quando sposti o rinomini un file, **cerca il suo path in `pyproject.toml`** (`per-file-ignores`,
  `overrides` di mypy, `files`) prima di committare. Usa glob di directory
  (`src/gpufsm/backends/warp/*.py`) invece di nomi singoli: sopravvivono a un file nuovo.
- Dopo un `--fix` su codice DSL (Warp, Triton, kernel), **diffare l'output** invece di fidarsi
  del "All checks passed". Un linter che non conosce la semantica del DSL è un editor automatico.
- La stessa classe di trappola vale per i commenti che spiegano perché qualcosa di apparentemente
  ridondante serve: `int(0)` nei kernel Warp, le architetture `-real` in `CMakeLists.txt`,
  l'ordine delle estrazioni RNG in `bench/generators.py`. Sono elencati in `docs/CONTRIBUTING.md`.

Vedi anche [[memory-lives-in-the-repo]].
