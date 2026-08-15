---
name: triton-build-harness-on-modal
description: scripts/modal_triton.py compila Triton da sorgente su Modal in alberi paralleli, così baseline e patch si confrontano senza rebuild
metadata:
  type: project
---

`scripts/modal_triton.py` tiene **più checkout persistenti** in un Modal Volume (`--tree main`,
`--tree ws`, `--tree ws2`, …), ognuno col suo venv. Build su CPU (32 core), GPU solo per i test.

**Why:** il primo loop rifaceva la build a ogni confronto baseline↔patch, ~5 minuti a giro, due volte
per misura. Con due alberi il confronto è una singola chiamata e i due binari coesistono.

**How to apply:**

```bash
python scripts/modal_triton.py build --tree ws --ref <sha> --patch p.patch
python scripts/modal_triton.py run --cpu --tree ws --cmd "..."     # triton-opt, lit: niente GPU
python scripts/modal_triton.py run --tree ws --cmd "..."           # H100
```

Trappole già pagate, da non ripagare:
- Le dipendenze di build vanno nel **venv** (l'install gira con `--no-build-isolation`), e sono quelle
  di `python/requirements.txt`: **nanobind pinnato**, non pybind11.
- **`FileCheck` c'è già**: sta in `python/triton/FileCheck` dentro il checkout, non nel pacchetto LLVM
  scaricato. Il sostituto `filecheck` di PyPI fa fallire 29 test anche su un albero pulito, quindi con
  quello si confronta la *lista dei falliti* prima/dopo, non il numero assoluto.
- Serve `PYTHONPATH=<tree>/python`: il finder dell'editable install non regge l'import dei backend da
  una cwd qualsiasi (l'harness lo imposta da sé).
- Il container ha `/tmp` piccolo: per `pre-commit` in locale serve `TMPDIR` su disco.

Costo reale della notte del 16 ago: qualche decina di minuti CPU e ~25 minuti di H100, dentro il free
tier di $30/mese. Vedi [[upstream-contribution-method]].
