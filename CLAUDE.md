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
- **SOTA da citare/battere (verificato 2026-06-25, vedi `docs/LITERATURE_REVIEW.md`):** ngAP (ASPLOS'24
  Best Paper, 10.1145/3617232.3624848; ext. TOCS 10.1145/3748646), **HybridSA** (OOPSLA'24,
  10.1145/3689771 — NFA bit-parallela GPU, la prior art più vicina al nostro bit-thesis: citare e
  differenziare), **BitGen** (MICRO'25, 10.1145/3725843.3756052 — è "Interleaved Bitstream Execution",
  bitstream Parabix per regex, **NON** packing 1-bit di stati NFA), **AsyncAP** (SIGMETRICS/POMACS 2023,
  10.1145/3579453 — ⚠️ NON è HPCA), **AutomataBLAS** (TACO'25, 10.1145/3774656 — AP-as-SpMV memory-efficient),
  iNFAnt (SIGCOMM CCR'10), Hyperscan (NSDI'19, baseline CPU). Benchmark: ANMLZoo (IISWC'16),
  AutomataZoo (IISWC'18). Metriche: throughput (Gbps) + latency, mediana+CI95 (timing non-gaussiani,
  Hoefler&Belli SC'15) su ≥2 GPU; roofline + Nsight (DRAM bytes, L2 hit, occupancy, sectors/req).
- **Novità "abstraction regret":** termine inedito (0 hit in letteratura) MA il fenomeno no — va
  operazionalizzato (cost model predittivo) e difeso da perf-portability (Pennycook 2016) e dal
  counter-thesis autotuning (arXiv:2505.03780). Core difendibile = automi irregolari × layout-memoria
  vincolato-dal-DSL × ablation/cost-model quantificato.
- **Multi-DSL (stato):** **Warp** backend FATTO e verde (thread-SIMT Python esprime gli automi, ≤64 stati).
  **Gluon**: provato e **NON esprime il kernel** — `gl.load` ritorna sempre un tensore con layout (niente
  scalar load), quindi il loop CSR data-dependent `for k in range(lo,hi)` è inesprimibile. È un *finding*
  più forte di un kernel: la abstraction regret sugli automi è prima di tutto un limite di **control-flow**,
  non solo di layout (Gluon dà controllo del layout ma non serve). Vedi `docs/DSL_EXPRESSIVENESS.md`.
  **Mojo** = breadth cross-vendor futura. Trap (solo-tensor, NON benchmarkare): cuTile, CuTe DSL,
  ThunderKittens, Pallas, TileLang.
- **Venue/timeline (oggi 2026-06-25; IISWC/PACT/MICRO/ASPLOS-Spring SCADUTI):** arXiv ora → **PMBS@SC26
  (paper 5 ago 2026)** target realistico → **ASPLOS 2027 Fall (9 set 2026)** anchor conferenza. Stretch:
  HPCA 2027 (31 lug), PPoPP 2027 (3 ago). In 1 settimana è realistico un **preprint arXiv A+C**, non un full conference.
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

## 7. Stato corrente (handoff sessione 2)

### Fatto e verde (GPU) — sessione 2, RTX 4070 (sm_89), CUDA toolkit 13.3 / driver 580 (max CUDA 13.0)
- **Backend GPU validati + 2 tecniche memory-centric.** `pytest` → **23 verdi** (20 CPU + 3 GPU).
  Tecniche per backend GPU: `dense`, `bitpacked`, `multistream` (`gpufsm list`).
- **Tecnica `bitpacked`** (asse byte→bit): working-set = bitmask packed (1 bit/stato, parole 64-bit) invece
  di un int8/stato; stesso algoritmo CSR del `dense`, solo il layout cambia (apples-to-apples). Triton: kernel
  con accept-test word-parallel; ⚠️ le maschere bit DEVONO essere int64 (`one << x`) — i literal Python li
  tronca Triton a int32 perdendo i bit ≥32 (rompeva NFA >64 stati, ora coperto da stress 65..500 stati).
  CUDA: kernel `template<int NWORDS>` → per ≤64 stati (NWORDS=1) il working-set è un `unsigned long long`
  **register-resident** (byte→bit + global→register); dispatch fino a 512 stati. Evidenza (~4 KB, no-match
  full-scan): triton 2.69→2.14 ms (1.26×); cuda 4.11→2.00 ms (2.05×).
- **Tecnica `multistream`** (asse single→multi-stream) + **API `run_batch`** (esportata): un batch di stringhe
  in un solo lancio, un program/block per stringa (Triton grid=(N,) con slice cur/nxt per-program; CUDA un
  block/stringa, input concatenati + offset). `run_batch` ha fallback a loop di `run`, quindi ogni tecnica è
  batchabile. ⚠️ Il multi-stream **non è novel** (CLAUDE.md §4): tenuto come baseline onesta dell'ablation.
  Evidenza (1024×256 B vs loop per-stringa): triton 242→3.8 ms (63×); cuda 324→8.3 ms (39×) — in gran parte
  ammortamento overhead-launch + parallelismo tra SM.
- Commit: `fix(gpu): validate ...`, `feat(gpu): add bit-packed ...`, `feat(gpu): add multi-stream ...`.
- Fix iniziale di validazione (commit `fix(gpu): validate Triton + CUDA dense backends on hardware`):
  - **Triton**: il kernel `dense` aveva `return` dentro il `for` per-posizione (vietato da Triton →
    `UnsupportedLanguageConstruct`). Riscritto con flag `done` che congela il primo match (latch-first-match)
    e lascia girare il loop fino in fondo.
  - **CUDA**: aggiunto `CUDA_CHECK` su launch/sync (gli errori erano silenziati e mascheravano il guasto reale).
  - **CMakeLists**: default `CMAKE_CUDA_ARCHITECTURES` = `75-real;80-real;86-real;89-real` (solo SASS, **niente
    PTX**), impostato **prima** di `enable_language(CUDA)`. Il toolkit (13.3) è più recente del max CUDA del
    driver (13.0): qualsiasi PTX incorporato viene rifiutato al load ("PTX compiled with an unsupported
    toolchain"); le cubin real-arch caricano grazie alla minor-version compatibility. Evitare numeri nudi
    (`89`) e `native` (incorporano PTX) su toolkit/driver disallineati.
- **Setup ambiente** (l'host ha `externally-managed-environment` PEP 668): venv `.venv` con
  `--system-site-packages` (riusa torch 2.9.1+cu128 + triton 3.5.1 da `~/.local`). Install:
  `.venv/bin/pip install -e ".[dev,triton]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON`.
  ⚠️ `GPUFSM_BUILD_CUDA=ON` come env var NON basta: scikit-build-core legge il define dal pyproject → va
  passato via `--config-settings`.
- **Assi ablation FATTI**: byte→bit (`bitpacked`), single→multi-stream (`multistream`), global→shared CSR
  (`multistream_shared`), sync→async (`multistream_async`). CUDA limitato a ≤512 stati (BITPACKED_MAX_WORDS=8).
- **Multi-DSL FATTO**: backend **Warp** (thread-SIMT, ≤64 stati). **Gluon** provato → non esprime il kernel
  (no scalar load) — `docs/DSL_EXPRESSIVENESS.md`.
- **Cost model FATTO** (`gpufsm.costmodel` + `scripts/calibrate_costmodel.py` + `paper/data/costmodel_rtx4070.csv`).

### ⚠️ FINDING CHIAVE che riformula la roadmap (vedi `docs/RESULTS_COSTMODEL.md`)
1. **I kernel attuali sono COMPUTE-bound, non memory-bound.** L'eps-closure è O(n²)/simbolo (n passi × n
   stati) + scan O(n) → throughput ∝ 1/n². Prova: `multistream_shared` (traffic CSR = 0) **pareggia**
   `multistream` (traffic > 0) a ogni dimensione. ⇒ In questo regime **il layout di memoria non conta**.
   Gli assi memory (byte→bit, shared-CSR, async) mordono SOLO con un kernel **work-efficient**
   (active-set/worklist, stile ngAP) che porti il kernel nel regime memory-bound.
2. **L'abstraction regret è quantificata e NON è l'altezza dell'astrazione, è il PARADIGMA di esecuzione.**
   Costo compute vs CUDA (stesso algoritmo): **Triton (tile/SPMD) 15.7×, CUDA 1.0×, Warp (thread-SIMT) 0.62×**
   (batte la CUDA scritta a mano). Due DSL Python di pari livello agli estremi → conta tile/SPMD vs thread-SIMT.

### TODO prossima sessione (riformulato dai finding)
- ✅ **Kernel WORK-EFFICIENT FATTO** (CUDA `worklist`): itera solo gli stati attivi (bit set) + eps-closure
  frontier-based, elimina l'O(n²). **1000–7000× più veloce del full-scan**, speedup crescente con n (n=64→1148×,
  n=500→7147×). Validato vs reference (30 batch ≤500 stati, 0 mismatch). È la base del contributo (B).
  TODO: versione Triton worklist; verificare con Nsight se ora è memory-bound (→ gli assi memory contano).
- ✅ **Sweep rigoroso FATTO** (task #7): `paper/data/sweep_techniques.csv` (median+CI95). worklist 15–132 Gbps
  vs full-scan ~0.5; multistream/shared/async identici → compute-bound confermato.
- ✅ **Figure FATTE** (task #9): `paper/figures.py` (4 figure dai CSV versionati; supera la legacy generate_figures.py).
- ✅ **Draft paper FATTO** (task #10): `paper/DRAFT.md` (arXiv-style, tutti i finding + figure + related work verificato).
- ⛔ **Nsight (task #6) BLOCCATO**: `ERR_NVGPUCTRPERM` (counter admin-gated, serve sudo/`NVreg_RestrictProfilingToAdminUsers=0`+reboot).
  Fix + alternativa in `docs/PROFILING.md`. La tesi compute-bound è già provata dall'ablation (shared-CSR pareggia) → ncu è confermativo.
- ✅ **ANML loader FATTO** (task #8, parser): `io/anml.py` parsa il sottoinsieme ANML (homogeneous→edge-labelled,
  symbol-set classes/ranges/negation/wildcard) + exporter `to_anml`; validato con fixture + round-trip (4 test).
  ⚠️ Manca solo il **download dei dati ANMLZoo reali** (DATASETS vuoto, serve SHA pinnato da mirror fidato — non
  bypassare la safety). Con i dati → numeri su automi reali (forte per i reviewer).
- ✅ **Worklist Triton FATTO** (≤64 stati): Triton **PUÒ** esprimere il kernel work-efficient via `libdevice.ffs`
  + while-loop data-dependent (a differenza di Gluon che non ha scalar load). MA paga **~9× di regret vs CUDA**
  sul kernel work-efficient (cuda 221–286 Gbps, triton 26–29 Gbps), vs 15.7× sul full-scan → **espressività ≠
  efficienza**: anche esprimendo l'algoritmo giusto, il modello tile/SPMD impone un penalty costante grosso sul
  lavoro scalare data-dependent. (Finding forte per il paper.)
- TODO rimasti: **worklist warp/block-parallel** (per avvicinare ngAP, contributo B forte); **2ª GPU** (generalità);
  migrare DRAFT.md → LaTeX; pin SHA ANMLZoo + run suite; aggiungere worklist al sweep/figure.
- **Contributo (A)+(C) è già forte e difendibile ORA**: caratterizzazione + cost model + regret quantificata
  + abstraction-spectrum (CUDA/Warp esprimono, Triton stride 15.7×, Gluon non esprime) + worklist 15–132 Gbps. Preprint pronto in bozza.

### Fatto e verde (CPU) — sessione 1
- Fondazione completa: `src/gpufsm` (nfa, reference, bitmap, result, registry, api, cli, examples,
  io/{anml,datasets}), backend CPU (`reference`, `bitmap`).
- Packaging `pyproject`+`scikit-build-core` (build CUDA graceful), CI GitHub Actions (ruff+mypy+pytest CPU).
- Test: **20 verdi** (`pytest -m "not gpu"`), incl. fuzz 300 NFA bitmap==reference. ruff+mypy puliti.
  `pip install -e .` funziona. CLI `env/list/verify/bench/sweep` funzionano.
- Dataset con checksum (`io/datasets`), docs (METHODOLOGY/REPRODUCIBILITY/CONTRIBUTING), paper migrato in `paper/`.
- Trim legacy completato: working tree ~90M → 17M.

### TODO prossima sessione (priorità)
1. **Completare l'ablation memory** (il contributo A): mancano gli assi **global→shared CSR** (CSR read-only
   in shared memory per blocco) e **sync→async transfer** (pinned + buffer persistenti + `cudaMemcpyAsync`,
   riportando kernel-time e transfer-time separati — `Result` già li distingue). Poi la versione
   **bit-parallela coalescizzata** (thread cooperanti per parola di stato, stile iNFAnt) per (B): deve battere
   il multi-stream banale e avvicinarsi (~2–3×) a ngAP/CUDA.
2. **Sweep/CSV multi-tecnica**: estendere `gpufsm sweep` per coprire tutte le tecniche×backend e produrre il
   CSV per l'ablation (lo schema alimenta le figure). Aggiungere un comando/bench batch per il multi-stream.
3. **ANML loader** (`io/anml.py` è uno stub): parser Python per ANMLZoo/AutomataZoo + benchmark suite.
4. **Figure paper**: riscrivere `paper/generate_figures.py` sullo schema CSV di `gpufsm sweep`.
5. **§13.2 SOTA**: integrare citazioni/numeri dal `/deep-research` (run `wf_b1efa63a-655`).

### Dove sta il codice
Repo nuova **`gpufsm`** (privata). Storia pulita: contenuto = commit iniziale del branch `gpufsm-main`
(orphan) della vecchia repo `triton_vs_cuda_fsm`. La provenienza/legacy (kernel ngAP v2, BitGen, anml) resta
nella history di `triton_vs_cuda_fsm` per riferimento durante il port GPU.
Piano completo: `/root/.claude/plans/voglio-un-refactoring-completo-jolly-teapot.md`.
