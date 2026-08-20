# CLAUDE.md — gpufsm

Memoria di progetto, caricata a ogni sessione in questa repo. **Tienila corta**: qui sta solo
ciò che serve *adesso*. Lo storico delle sessioni sta in `docs/HISTORY.md`, i fatti
trasversali in `docs/memory/` (indice: `docs/memory/MEMORY.md`).

> ⚡ **Autonomia: non chiedere conferme per i passi operativi** (aprire/mergiare PR interne,
> installare dipendenze, configurare infra). Si fanno e si riporta.
> Vedi `docs/memory/be-autonomous-no-confirmations.md`.

## 1. Cos'è

Studio + framework sull'elaborazione di **automi (NFA/DFA) su GPU**, che confronta i DSL
(**Triton**, **Warp**, **Gluon**) contro **CUDA** scritto a mano sui workload **irregolari**.

**Tesi:** l'*abstraction regret* — la performance preclusa dal DSL perché non esprime il
layout o il control-flow necessario, **ad algoritmo fisso** — segue il **paradigma di
esecuzione**, non l'altezza dell'astrazione. Nel 2×2 (altezza × paradigma) il costo segue la
colonna: CUDA e Warp (thread-SIMT) ≈ 1×, Triton (tile/SPMD) 6–8×, Gluon non esprime il kernel.
Due facce: **NFA** control-flow-bound, **DFA** memory-bound.

## 2. Decisioni prese (NON rimetterle in discussione senza l'utente)

1. **Scope = solo automi.** MatMul/MLP fuori dal core.
2. **Oracolo di correttezza = simulatore CPU** (`gpufsm.reference`). Ogni backend deve
   produrre output identico (accepted + match_len). Semantica: **latch-first-match**.
   Nessun numero si riporta prima che l'oracolo sia verde (`gpufsm.bench.oracle`).
3. **Packaging:** `pyproject.toml` + `scikit-build-core` (CUDA via CMake, build graceful).
4. **Repo `gpufsm`, privata** fino al preprint arXiv.
5. **Multi-stream non è novel** (standard dal ~2015): è la baseline onesta dell'ablation.

## 3. Architettura

```
src/gpufsm/
├─ core/       nfa dfa result registry bitmap packing   (numpy-only, backend-agnostic)
├─ reference.py  gli oracoli: simulate() [NFA] + simulate_dfa() [DFA]
├─ api.py        run / run_batch / benchmark — unica porta, NFA e DFA
├─ backends/  cpu.py + triton/ cuda/ warp/  (un modulo per tecnica, guardia nel __init__)
│  └─ cuda/native/  una translation unit per famiglia di kernel + include/
├─ bench/     generators timing oracle nvcc csvio sweep  (l'harness condiviso)
├─ io/ costmodel.py cli.py
experiments/cure/{milestones,landmarks,passes,validation}/   provenienza dei numeri
scripts/      entry point sottili (misure + driver Modal)
```

Due regole che reggono tutto:

- **Un solo punto di estensione:** `@register(Kind, Backend, technique)`. NFA e DFA passano
  dallo stesso registry; `run()` ricava il `Kind` dall'automa.
- **`gpufsm.bench` è l'harness condiviso.** Un solo `random_nfa`, un solo builder nvcc, un
  solo timer, una sola scrittura CSV. Non reintrodurre copie locali: le famiglie di NFA sono
  riprodotte bit-per-bit e pinnate da `tests/test_generators.py` — se cambia l'ordine delle
  estrazioni RNG, i CSV committati non descrivono più gli automi che il codice costruisce.

## 4. Convenzioni

> 🗣️ **Risposte BREVI e CONCISE.** Risultato e punti critici; il dettaglio sta nei commit e
> nei doc. Vale anche per la prosa dei paper: **mai em dash**.

Prima di **ogni** commit (è esattamente ciò che gira in CI):

```bash
ruff format --check src tests scripts experiments paper/figures.py paper2/figures.py && \
ruff check src tests scripts experiments paper/figures.py paper2/figures.py && \
mypy src/gpufsm && pytest -m "not gpu" -q
```

- **Python**: src-layout, type hints, ruff + mypy. Niente codice morto.
- **Test**: pytest. Marker `gpu` per ciò che richiede GPU. `tests/test_golden.py` pinna i
  verdetti dell'oracolo: se fallisce è cambiata la *semantica*, non la struttura.
- **Build graceful**: senza CUDA toolkit/GPU l'estensione non compila e i backend si
  registrano "non disponibili"; il core CPU/Triton resta installabile.
- **Riproducibilità**: le figure si rigenerano SOLO da CSV versionati.
- **Git**: commit chiari e atomici, messaggi in inglese, mai attribuzione AI.

## 5. Dove si lavora

**In locale.** Il cloud è un ripiego (CPU-only + GitHub scoped a una repo sola).

- **PC** = RTX 4070 diretta: build CUDA, Nsight (sudo passwordless), sweep.
- **Portatile** = niente GPU → **Modal**:
  ```bash
  python scripts/modal_gpu.py --preflight
  python scripts/modal_gpu.py --gpu A100 --apt g++ \
      --cmd 'pip install -e "." --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON' \
      --cmd 'pytest -m gpu -q'
  ```
  `--apt g++` serve: l'immagine CUDA ha nvcc ma non un compilatore host, e senza CMake non
  configura. Token `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`/`GITHUB_TOKEN` già in env sul
  portatile — non stamparli mai.
- **Upstream Triton da locale con `gh`** (da cloud l'API dà 403: è il proxy della sessione).
  Registro PR: `docs/PR_LEDGER.md`. ⚠️ Ogni push azzera l'approvazione della CI upstream:
  fare tutti i push in una volta, poi congelare il branch.
- ⚠️ `/tmp` è un tmpfs da 2 GB e si riempie: usare `TMPDIR` su disco per pip/build.

## 6. Stato corrente

### ✅ HPEC 2026: ACCETTATO (orale + Xplore) — camera-ready entro il 4 SET

Paper 133 `Accept-Oral-Xplore`, talk mercoledi 16 set, 20:15 ora italiana. Le due review sono
gia' state evase nel `.tex` (NFA/DFA espansi, giustificazione del protocollo line-for-line,
predittivita' della mappa). Restano **camera-ready, Copyright Form e registrazione**: sono azioni
a nome dell'utente. Dettagli e scadenze in `docs/HPEC_CAMERA_READY.md`.

### ⚠️ AZIONE APERTA: sottomettere ad **ASPLOS 2027 entro il 9 SET 2026 AoE**

PPoPP 2027 è stata persa creando l'account e non premendo submit. **Premere davvero SUBMIT**:
un draft salvato non conta.

- Pronto e committato: `paper2/gpufsm_asplos.{tex,pdf}` (anonimo, da caricare),
  `paper2/gpufsm_asplos_named.*` (NON caricare), `paper2/ASPLOS_ABSTRACT.txt`,
  `paper2/resubmission_note.pdf`, compliance in `docs/SUBMISSION_ASPLOS.md`.
- 🚫 Bloccante esterno: il sito HotCRP di settembre non è ancora aperto. **Prendere l'URL dal
  CFP, non indovinarlo.**
- Toolchain LaTeX installata; build: `pdflatex → bibtex → pdflatex ×2`.

### Upstream Triton

**Stato 18 ago 2026:** 1 mergiata (#11311), 3 aperte, 1 chiusa (#11323, "micro optimization"),
2 issue. **La review si e' aperta il 17 ago**: Jokeren ha lasciato 4 richieste su #11324, tutte
evase, e una obiezione su #11325 che ha portato a riscrivere il criterio.

| | cosa | stato |
|---|---|---|
| #11324 | i terminatori di `warp_specialize` come sync point | +22/-1, 4 richieste evase, CI verde sulla sostanza |
| #11325 | barriera **mancante**: offset di subslice fra frame diversi | riscritta 3 volte, ultima sul **tipo** della sorgente |
| #10766 | fold split/join, da luglio | **sbloccata il 18 ago**: il test svuotato e' stato ripristinato |
| #11326, #11328 | barriere mancanti (chiamata, cluster multicast) | issue con riproduttori, nessun triage |

⚠️ **Non pushare senza motivo** (azzera la coda di approvazione CI), ma **una richiesta del
maintainer e' un motivo**: risponde in ore, non in giorni. Rispondere **dentro** il thread inline.
Metodo: `docs/memory/upstream-review-dynamics.md`, `docs/PR_LEDGER.md`.

### Storico

Tutto il resto — cosa è stato misurato, cosa è stato confutato, i numeri e le loro date — sta
in **`docs/HISTORY.md`**. Leggerlo quando serve la provenienza di un numero o per non ripetere
un esperimento già fallito, non a ogni sessione.
