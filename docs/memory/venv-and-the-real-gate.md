# Il gate gira solo dal venv, e da un worktree mente

Il 18 ago 2026, provando a far girare il gate di `CLAUDE.md` sul portatile, `ruff` non era nel PATH e
**`mypy` e `pytest` non erano installati da nessuna parte**. Il `.venv/` del progetto conteneva solo
utility (`ruff`, `modal`, `pre-commit`, `clang-format`): l'ambiente di sviluppo vero non era mai stato
preparato su questa macchina.

Due conseguenze da tenere a mente.

## 1. I comandi del gate vanno chiamati col path del venv

```
V=/home/alepot55/Desktop/projects/gpufsm/.venv/bin
$V/ruff format --check ...   $V/ruff check ...   $V/mypy src/gpufsm   $V/python -m pytest -m "not gpu" -q
```

Chiamare `ruff` nudo dà `command not found`, che **non è un gate rosso** — è un ambiente assente. Se
i due casi si confondono, ogni notte arriva un falso allarme e in una settimana il watcher notturno
diventa rumore che si ignora. Il prompt del cron `gpufsm-nightly-gate` distingue esplicitamente le
due cose per questo motivo.

Riparazione dell'ambiente, dal checkout principale: `$V/pip install -e ".[dev]"` — è quello che fa
anche la CI (`.github/workflows/ci.yml`). Serve `TMPDIR` su disco, perché `/tmp` è un tmpfs da 2 GB.

Stato verde di riferimento al 18 ago 2026: 118 file formattati, ruff pulito, mypy pulito su 40 file,
**942 test passati** e 24 deselezionati (marker `gpu`). Un numero di test molto più basso senza che
ne siano stati cancellati è un errore di collection travestito da successo
([[empty-output-is-not-a-result]]).

## 2. `pip install -e` punta al checkout principale, sempre

L'install editable è fatto da `/home/alepot55/Desktop/projects/gpufsm`, quindi **da dentro un
worktree `import gpufsm` risolve comunque su `src/` del principale**, non su quello del worktree.
Verificato:

```
$ cd .claude/worktrees/<qualsiasi>/ && python -c "import gpufsm; print(gpufsm.__file__)"
/home/alepot55/Desktop/projects/gpufsm/src/gpufsm/__init__.py
```

Vuol dire che **lanciare `pytest` dentro un worktree testa il codice di `main`**, non le modifiche che
hai davanti: verde ingannevole se stai cambiando `src/`. Per testare davvero il worktree serve
`PYTHONPATH=src`. Il cron notturno non è esposto al problema perché gira nel principale, ed è scritto
lì apposta.
