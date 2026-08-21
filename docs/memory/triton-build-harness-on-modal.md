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

## Cambiare workspace non è gratis (21 ago 2026)

`ResourceExhaustedError: Workspace ... has exceeded its spend limit` ferma **build e run insieme**.
Passare a un secondo profilo (`modal token set --profile=X`) risolve il limite ma **non porta i
volumi**: sono per-workspace, quindi `triton-upstream` non esiste nel nuovo e si riparte da clone +
build a freddo. Con ccache freddo il primo build è dell'ordine dell'ora; quelli dopo, che toccano un
file, sono minuti.

Due trappole nell'ordine in cui le ho pagate:

- `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in env **sovrascrivono il profilo**, silenziosamente: il CLI
  lo dice in un warning che scorre via. Ogni chiamata va fatta con
  `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=<profilo> ...`.
- `timeout` non può eseguire una funzione di shell: `timeout 600 M build ...` dà
  `timeout: failed to run command 'M'`. Il timeout va **dentro** la funzione.

## `git apply --3way` implica `--index`

Il build applica la patch con `git apply --3way`, che **mette le modifiche in staging**. Quindi
`git diff` esce vuoto e l'albero sembra pulito anche quando la patch c'è: per vedere cosa sta
davvero compilando serve **`git diff --cached`**. Confuso una volta per "la patch non è stata
applicata".

## `FileCheck`: la nota sopra era già scritta, e l'ho ignorata lo stesso

Il 21 ago ho cercato `FileCheck` sotto `/root/.triton/llvm`, non l'ho trovato, `$FC` è rimasto vuoto
e il controllo di regressione ha stampato **ROSSO**. Il path giusto (`python/triton/FileCheck`) era
già tre righe più su in questo stesso file. Vedi [[a-missing-binary-is-not-a-red-test]]: quando un
gate diventa rosso, **stampare il path del binario prima di credere al rosso**.
