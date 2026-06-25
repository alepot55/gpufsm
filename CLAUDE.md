# CLAUDE.md — gpufsm

Memoria di progetto per Claude Code. Auto-caricata a ogni sessione in questa repo. Tienila aggiornata
quando cambiano decisioni o convenzioni.

## 1. Cos'è questo progetto (contesto & tesi)
`gpufsm` è il refactoring publication-grade di `triton_vs_cuda_fsm`: uno studio + framework sulla
**elaborazione di automi a stati finiti (NFA/FSM) su GPU**, che confronta **OpenAI Triton** (DSL block-based)
vs **NVIDIA CUDA** (basso livello) sui workload **irregolari**.

**Tesi centrale (memory-centric):** per i workload irregolari su GPU è l'**organizzazione della memoria** —
non la complessità algoritmica — il determinante primario della performance; e l'astrazione di un DSL conta
solo nella misura in cui **vincola il layout di memoria esprimibile** ("abstraction regret"). Mostriamo
quanta parte del gap Triton↔CUDA (10–30×) si chiude riorganizzando *solo la memoria*, a parità di algoritmo.

## 2. Decisioni prese (NON rimetterle in discussione senza l'utente)
1. **Scope = solo FSM/FSA.** MatMul/MLP fuori dal core (al più contrasto minimale in appendice).
2. **Trim aggressivo + moduli opzionali.** BitGen, ngAP avanzato, parser ANML C++ = opzionali, isolati
   dietro interfaccia. Il core installa e gira senza di essi.
3. **Packaging moderno:** `pyproject.toml` + `scikit-build-core` (compila CUDA via CMake). Docker/conda opzionali.
4. **Nuova repo GitHub:** nome **`gpufsm`**, **privata** per ora (pubblica al preprint arXiv).
5. **Oracolo di correttezza = simulatore NFA su CPU** (`reference.py`). Tutti i backend devono produrre
   output identico (accepted + match_len). Semantica: **latch-first-match** (report al primo stato accettante).

## 3. Architettura target
- **Una sola API** (`api.py`): `run(nfa, input, backend, technique) -> Result`;
  `benchmark(...) -> BenchmarkStats` (warmup + N ripetizioni → mean/std/CI95).
- **Registry estensibile** (`registry.py`): `@register(Backend, Technique)` → un backend/tecnica = un file +
  una riga. Backend: `CPU` (reference), `TRITON`, `CUDA`.
- **NFA in CSR** (`nfa.py`): transizioni simboliche + epsilon in CSR, accept states, alphabet map, simbolo ANY.
- **src-layout**: `src/gpufsm/{nfa,reference,result,registry,api,cli}.py`, `io/`, `backends/{triton,cuda}/`.

## 4. Findings chiave da riusare (dal vecchio codice)
- **Bit-packing è il difetto #1 e il fix è già in-repo:** i kernel Triton default usavano int32 (4 B/stato);
  il path BitGen impacchetta a 1 bit → 500 stati = 2000 B vs 64 B = **31× di spreco**. Unificare su bitmap
  packed 1-bit + bitwise. Riusare la logica `(num_states+31)//32` di `fsa_engine.py`.
- **CUDA**: bitmap on-stack in local memory, limite ~256 stati → per ≤64 stati un `unsigned long long`
  register-resident. CSR read-only condivisa, mai in shared memory → candidarla a shared/blocco.
- **Transfer PCIe**: lato CUDA era `cudaMemcpy` sincrono (≈99.9% del tempo su 1 MB); Triton async. Riportare
  *kernel time* e *transfer time* separati. Pinned + buffer persistenti + `cudaMemcpyAsync`.
- **Multi-stream**: CUDA lo faceva già (`grid = num_strings`), Triton era single-program (`grid=(1,)`).
  ⚠️ Il multi-stream da solo **non è novel** (standard dal ~2015): non venderlo come contributo.
- **Suite**: 14 automi ANMLZoo/AutomataZoo/Regex (Brill, ClamAV, Snort, Protomata, Yara, Bro217, Fermi,
  Hamming, SPM, EntityResolution…), 80–500 stati.
- Riuso utile: input-prep vettorizzato (lookup table numpy 256 voci), rappresentazione CSR, paper LaTeX.

## 5. Strategia di pubblicazione (sopra il SOTA)
- **Contributo = diagnosi + cura:** (A) cost model / ablation memory-centric dell'abstraction regret +
  (B) engine automi **portabile in Triton** (bit-parallelo, multi-stream coalescizzato) che recupera il gap.
  (C) framework+artifact come complemento (badge Artifact Evaluation).
- **Caveat novelty:** (B) è forte SOLO se batte la baseline multi-stream banale e si avvicina (~2–3×) a
  ngAP/CUDA; altrimenti il paper poggia su (A)+(C) onesti.
- **SOTA da citare/battere:** ngAP (ASPLOS'24), BitGen (MICRO'25), AsyncAP (HPCA), iNFAnt/iNFAnt2, DFAGE,
  Hyperscan (NSDI'19, baseline CPU). Benchmark: ANMLZoo (IISWC'16), AutomataZoo (IISWC'18). Metriche:
  throughput (Gbps) + latency, con mean/std/CI su ≥2 GPU; profilare banda/occupancy/L2 (Nsight).
- **Venue:** IISWC/PACT/PPoPP (per A) → MICRO/ASPLOS (se B regge). **Preprint arXiv presto** per priorità;
  artifact su Zenodo (DOI). Lavoro autosufficiente e indipendente dall'affiliazione.
- Esiste un report `/deep-research` (verifica citazioni/numeri esatti) — integrare quando disponibile.

## 6. Convenzioni di sviluppo
- **Python**: src-layout, type hints, `ruff` (lint+format) + `mypy`. Niente codice morto.
- **Test**: `pytest`. Marker `gpu` per i test che richiedono GPU (`pytest -m "not gpu"` deve passare in CI
  CPU-only). L'oracolo è `reference.py`: ogni backend testato per output identico su tutta la suite.
- **Build graceful**: se manca CUDA toolkit/GPU, l'estensione non compila e i backend CUDA si registrano
  "non disponibili" → core solo-Triton/CPU resta installabile e testabile.
- **Dati**: fixtures piccole versionate in `data/`; suite grande scaricata on-demand con **checksum**
  (mai link privati/SharePoint).
- **Branch di lavoro**: `claude/repo-refactor-optimize-snflie`. Commit chiari e atomici.
- **Riproducibilità**: figure del paper rigenerate SOLO da CSV versionati; `gpufsm env` cattura versioni/GPU.

## 7. Stato corrente (handoff sessione 1)

### Fatto e verde (CPU)
- Fondazione completa: `src/gpufsm` (nfa, reference, bitmap, result, registry, api, cli, examples,
  io/{anml,datasets}), backend CPU (`reference`, `bitmap`).
- Packaging `pyproject`+`scikit-build-core` (build CUDA graceful), CI GitHub Actions (ruff+mypy+pytest CPU).
- Test: **20 verdi** (`pytest -m "not gpu"`), incl. fuzz 300 NFA bitmap==reference. ruff+mypy puliti.
  `pip install -e .` funziona. CLI `env/list/verify/bench/sweep` funzionano.
- Dataset con checksum (`io/datasets`), docs (METHODOLOGY/REPRODUCIBILITY/CONTRIBUTING), paper migrato in `paper/`.
- Trim legacy completato: working tree ~90M → 17M.

### Scritto ma NON validato (serve GPU — l'ambiente di sviluppo non ne aveva)
- `backends/triton_backend.py`: kernel Triton `dense` (single-program, mirror del reference). Si registra
  solo se torch+triton+GPU presenti.
- `backends/cuda_backend.py` + `backends/cuda/nfa_kernel.cu`: kernel CUDA CSR `dense` + binding pybind11,
  build con `GPUFSM_BUILD_CUDA=ON`. Si registra solo se l'estensione `_cuda` è importabile.
- Test `tests/test_gpu_backends.py` (marker `gpu`, ora SKIPPED): confronto Triton/CUDA vs reference su
  esempi + fuzz. **Primo compito su GPU: farli passare** (`pytest -m gpu`).

### TODO prossima sessione (priorità)
1. **Validare GPU**: `pip install -e ".[dev,triton]"` + `GPUFSM_BUILD_CUDA=ON`; far passare `pytest -m gpu`.
   Probabili fix ai kernel `dense` (loop dinamici Triton, dtype, compile pybind11/CUDA).
2. **Tecniche bit-packed/multi-stream GPU** (il contributo): versione packed-1-bit + multi-stream
   coalescizzato, con `gpufsm.bitmap` come specifica eseguibile. Poi l'**ablation memory**
   (byte→bit, global→shared CSR, sync→async, single→multi-stream).
3. **ANML loader** (`io/anml.py` è uno stub): parser Python per ANMLZoo/AutomataZoo + benchmark suite.
4. **Figure paper**: riscrivere `paper/generate_figures.py` sullo schema CSV di `gpufsm sweep`.
5. **§13.2 SOTA**: integrare citazioni/numeri dal `/deep-research` (run `wf_b1efa63a-655`).

### Dove sta il codice
Repo nuova **`gpufsm`** (privata). Storia pulita: contenuto = commit iniziale del branch `gpufsm-main`
(orphan) della vecchia repo `triton_vs_cuda_fsm`. La provenienza/legacy (kernel ngAP v2, BitGen, anml) resta
nella history di `triton_vs_cuda_fsm` per riferimento durante il port GPU.
Piano completo: `/root/.claude/plans/voglio-un-refactoring-completo-jolly-teapot.md`.
