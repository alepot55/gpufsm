---
name: modal-gpu-harness-gotchas
description: Tre trappole di scripts/modal_gpu.py pagate il 2026-08-16 — comandi ignorati, g++ mancante, log dentro l'upload
metadata:
  type: project
---

**Why:** `modal_gpu.py` esce con rc=0 anche quando non ha eseguito ciò che gli hai chiesto. Tre
modi in cui è successo, tutti e tre ora chiusi nel codice, ma vale sapere che forma hanno.

**How to apply:**

1. **I globali del modulo non arrivano nel container.** Modal re-importa il file dentro il
   container, dove le `GPUFSM_MODAL_*` non esistono: `CMDS`/`FETCH` letti a import-time
   ricadevano sui default, quindi ogni run eseguiva la probe di default e riportava rc=0.
   Ora sono argomenti di `_run_remote`. **Regola generale: ciò che serve al runtime remoto passa
   per argomenti, non per l'ambiente.** Se un giorno una run "riesce" ma l'output non
   corrisponde ai comandi, è questo.

2. **`--apt g++` serve per costruire l'estensione CUDA.** L'immagine `nvidia/cuda:*-devel` ha
   `nvcc` ma nessun compilatore host, e CMake si rifiuta di configurare: la build fallisce, i
   backend CUDA restano "non disponibili" e i test GPU passano lo stesso perché sono skippati.
   4 test verdi invece di 24 è il sintomo.

3. **Non scrivere il log dentro la directory che Modal carica.** `> .tmp/x.log` nel repo fa
   fallire il job con `... was modified during build process`. Il log va nello scratchpad, e
   `.tmp` è nella IGNORE list.

Extra: costruire per una sola architettura
(`--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=80-real` su A100) taglia minuti di
GPU pagata rispetto alle quattro di default.

Vedi anche [[triton-build-harness-on-modal]] (per la build di Triton da sorgente, che è un
harness diverso: `scripts/modal_triton.py`, con Volume e alberi paralleli).
