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
> ⚠️ **CI parity (la CI falliva sempre):** la CI esegue **`ruff format --check`** oltre a `ruff check`.
> Prima di OGNI commit lanciare: `ruff format src tests scripts paper/figures.py && ruff check src tests && mypy && pytest -m "not gpu"`.
> Non basta `ruff check`: serve anche il **format**.

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
- **[Iter più recente] #2 DEEP CONCLUSO (shared-mem worklist) — finding onesto.** `worklist_shared`:
  working set in shared memory dinamica (warps/block adattivo per stare in 48KB; ≤1536 stati), vs il global
  di `worklist_warp`. Validato == warp bit-for-bit (9 test verdi). **FINDING:** shared **pareggia** warp
  (0.99–1.10×, `paper/data/worklist_shared_rtx4070.csv`) → una volta che il kernel è work-efficient il
  **layout del working-set NON è più il collo di bottiglia** (specchia il risultato compute-bound
  `multistream_shared`); il gap residuo verso SOTA assoluto (ngAP-class) è **algoritmico** (memoization/
  non-blocking), non residency. Documentato in Implementation+Limitations. ⇒ #2 chiuso onestamente (warp 3-9× @batch saturante
  è la vera vincita; ngAP-style memoization sarebbe "competere con SOTA", fuori scope = positioning non benchmark).
  **STATO: #2,#3,#4,#5 TUTTI FATTI. Resta solo #1 (2ª GPU) che richiede hardware dall'utente.**
- **[Iter -1] #5 (AE packaging) + #4 (SOTA table) FATTI.** #5: `docs/ARTIFACT_APPENDIX.md`
  (SIGPLAN-style check-list/install/claims→commands→expected + piano Zenodo-DOI-al-release), `CITATION.cff`
  arricchito (titolo two-faces, abstract, keywords), `REPRODUCIBILITY.md` aggiornato (6 famiglie non 2, .tex
  canonico non "migration pending", +righe DFA-sweep/warp-speedup), Artifact Availability nel .tex punta
  all'appendix. #4: tabella positioning SOTA (iNFAnt/AsyncAP/ngAP/HybridSA/BitGen) in related work — ESPLICITO
  che le cifre sono speedup sul LORO baseline/hardware (NON comparabili in Gbps assoluti), tutti algoritmi
  CUDA-only, il nostro asse (espressività DSL, algoritmo fisso) è ortogonale; match ngAP-class = future work.
  6 famiglie reali già bastano → non aggiunte altre (SPM 100k/24M troppo lento all'oracolo, marginale). Paper
  5pp pulito. RESTA: #2 deep (shared-mem block-cooperative per throughput assoluto SOTA), #1 (2ª GPU) dopo.
- **[Iter -1] #2 KERNEL BLOCK-PARALLEL FATTO (warp-per-string worklist).** `worklist_warp`:
  un warp (32 lane) per stringa; le lane partizionano le parole di stato e scatterano transizioni/eps-closure
  via `atomicOr` nel next-set globale condiviso, con `__any_sync` per frontier-empty/accept. Risolve la
  sotto-utilizzazione del worklist 1-thread su automi grandi. Validato bit-for-bit vs oracle (1252 stringhe,
  0 mismatch) + == worklist_global su NFA >64 stati (`tests/test_worklist_warp.py`, 5 verdi). Bench
  (`scripts/bench_worklist_warp.py`): ⚠️ **speedup BATCH-DEPENDENTE (audit 26 giu).** A batch saturante (4096
  stringhe, GPU piena): **real automi 3-9×** (levenshtein 3.7×, fermi 3.1×, brill 8.9×), sintetici densi ~12×.
  A batch piccolo (256) saliva a 12-180× perché global 1-thread non riempie la GPU → il 12-17× iniziale era un
  artefatto di batch. Il numero ONESTO/conservativo = 3-9× @batch saturante. CSV: `worklist_warp_rtx4070.csv`
  (+ `worklist_warp_batch_rtx4070.csv` documenta la sensibilità al batch). Driver = densità active-set × words,
  non size. Paper (Implementation+Limitations) corretto a 3-9×. **NSIGHT (26 giu):** worklist_warp fixa
  l'occupancy (17→57%) ma è **latency-bound NON memory-bound** — DRAM ≤2.25%, L2 hit ≥97.6% anche su brill
  (CSR 17MB ≫ L2 6MB), perché tutte le stringhe condividono la CSR e solo un hot-subset di righe è toccato →
  resta L2-resident. Spiega perché worklist_shared è inerte (working set già in L2) e perché il gap SOTA è
  **algoritmico** (worklist compatto active-ID, meno atomicOr/syncwarp, ngAP non-blocking), non memory.
  Dati in `paper/data/nsight_rtx4070.csv` + `docs/PROFILING.md`. PROSSIMO: prototipo worklist compatto (array
  di ID attivi) vs bitmap-scan O(nwords) — l'esperimento che potrebbe alzare il throughput assoluto.
  **FATTO + RISULTATO (26 giu): IPOTESI CONFUTATA.** `worklist_compact` (1-thread, frontier compatto O(active)
  + visited bitmap, clear O(active)) validato vs oracle (1200 stringhe, 0 mismatch). MA: vs global 1-thread
  appena **0.8–1.5×** (levenshtein 0.8× PEGGIO, fermi 1.0-1.2×, brill 1.5×), e vs warp **0.1–0.4×** (molto
  più lento). Perché: saltare una parola vuota nel bitmap-scan è 1 load+branch (economico) + accesso coalescizzato,
  mentre il frontier compatto ha load sparsi + dedup test-and-set per transizione → il word-scan NON era il collo
  di bottiglia. Tenuto come ablation validata (come worklist_shared). Premessa: automi reali sparsissimi (brill
  mean 1.5 attivi/667 words). ⇒ Né layout (shared) né work-reduction (compact) è la leva; lo è il **parallelismo**
  (warp, già fatto). Gap SOTA = **ridondanza algoritmica cross-string/symbol** (memoization, non-blocking ngAP).
  Knowledge base: `docs/KERNEL_EXPERIMENTS.md` (log vivo degli esperimenti kernel + esiti). Paper corretto. ⚠️ Rebuild ext: `pip install -e ".[dev,triton]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON`
  (NON `--no-build-isolation`: manca scikit_build_core nel venv). RESTA per SOTA assoluto: block-cooperative
  + shared-mem frontier privatization (prossimo passo #2). User (26 giu): fare #2-#5, #1 (2ª GPU) dopo.
- **[Iter -1] DFA sweep fine — knee L2 visibile** (vedi findings two-faces sotto).
- **[Iter -2] AUDIT COST-MODEL + typografia + artifact statement.** (a) Claim "<1% error at
  large n" era sovrastimato: errore reale predicted-vs-measured = **<1% solo a n=256** (CUDA 0.3%, Triton
  0.6%), ~2%(CUDA)/~13%(Triton) a n=128, 20–60% a n=32/64 (launch overhead). Warp fit esatto = 2pt/2par
  (non è segnale di qualità). Prosa riconciliata in .tex/DRAFT/RESULTS_COSTMODEL/PROFILING. (b) 3 tabelle
  (throughput/nsight/capability) wrappate in `\resizebox`+`\tabcolsep` → **0 overfull \hbox** (era 3), 5pp,
  no undefined refs. (c) Sezione `\section*{Artifact availability}` (AE-friendly: regen a un comando, suite CPU
  no-GPU, Zenodo DOI al release). Abstract riletto: tight e coerente coi numeri canonici, nessuna modifica.
- **[Iter -1] PROBE GLUON FALSIFICABILE + claim sharpening.** (a) `scripts/gluon_probe.py`:
  artefatto runnabile (non snippet) che riproduce l'errore esatto `Value argument cannot be block type
  if pointer argument is not a block` su Triton 3.5.1 — exit 0 sul fallimento atteso, exit 1 se un Gluon
  futuro lo compila → claim falsificabile per costruzione. Citato da gpufsm.tex/DSL_EXPRESSIVENESS/REPRODUCIBILITY.
  ⚠️ Gluon `@jit` DEVE stare in un file .py (no REPL/-c). (b) Contributo (A) ora front-loada le 2 facce + il
  controllo Triton↔Gluon + capability→cost table come novelty di testa. (c) Limitations: piano 2ª-GPU concreto
  (claim qualitativi = proprietà del compilatore arch-independent; il knee DFA L2 e i fattori di regret assoluti
  = run camera-ready su A100/H100 ≥40MB L2, framed come predizione falsificabile + re-run a un comando). Paper
  ricompila pulito (5pp, no undefined refs).
- **[Iter precedente] RIGORE NUMERI + suite reale allargata.** (a) DRAFT.md riconciliato col .tex (two faces,
  DFA second face §6.5, capability table §6.6, Hexcute/LMS/Tawa/Descend in related work). (b) **Audit numeri**:
  tutte le cifre citate ora tracciano ai `paper/data/*.csv` — corretti stale: regret NFA 15.7×→6–8× misurato /
  10.1× fit; Warp 0.62×→0.6–0.9×; worklist speedup 250×/1148×/7147×→332×@32..≈10⁴×@500; worklist regret 9×/142 Gbps→
  6.5×/164 Gbps; DFA 496@4096/207@200k→443@4096(4MB)/213@50k(50MB). Propagati a docs/ + CLAUDE.md. (c) **Real-suite
  3→6 famiglie**: +Fermi(40.8k)/RandomForest(33.2k,6.27M tr)/CoreRings(48k), tutti pure-STE, SHA pinnati, GPU
  worklist_global==reference bit-for-bit (test_anmlzoo_gpu 6 verdi).
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

### 🚀 SCOPE v2 (26 giu, mandato utente "contributo più forte, rivoluziona il paper")
**Tesi rivoluzionata: "le due facce dell'abstraction regret".** L'NFA si è rivelato control-flow/compute-bound;
manca la faccia memory-bound. La aggiungo con un **2° workload: simulazione DFA** (lookup tabella densa
states×256, 1 accesso random/simbolo → memory-bound, regime opposto). Tesi generale: *il regret di un DSL
dipende dalla capacità che il workload stressa* — NFA = faccia control-flow (Triton non esprime active-set →
9–15×), DFA = faccia memory-layout (layout/cache tabella domina). Eleva da "studio automi" a **framework
capability-vs-costo** su 2 workload × spettro DSL (CUDA/Triton/Gluon/Warp). Piano: (1) `dfa.py` core + oracolo;
(2) kernel DFA CUDA/Triton/Warp; (3) misurare regret DFA (atteso memory-bound, Nsight DRAM% alto); (4) riscrivere
il paper attorno alle due facce + tabella capability. Aumentare gradualmente, tutto a discrezione.
- **Progresso v2:** ✅ DFA core + oracolo + `dfa_api` (cpu/cuda/triton/warp); ✅ kernel DFA **CUDA/Triton/Warp**
  tutti validati vs oracolo; ✅ regret memory-bound misurato + figura (`paper/data/dfa_regret_rtx4070.csv`,
  `fig_dfa_memory_bound`). **Risultati two-faces (sweep fine 1–100 MB, `scripts/sweep_dfa.py`):** DFA memory-bound —
  cuda **picco 345 Gbps esattamente a 6 MB (= L2) → crolla 2.4× a plateau DRAM ~150–175 Gbps**, warp stessa forma
  a metà (160→97), **triton piatto 29–32 Gbps** su tutto il range (non raggiunge mai il regime memory-bound).
  Regret DFA: triton 5–12× (max dove cuda picca a L2), warp 1.5–2.2×. Quindi Triton paga regret su ENTRAMBE le
  facce (NFA control-flow 6–8×/10× fit + DFA memory 5–12×) → **è il modello tile/SPMD, non il workload**; Warp
  (thread) vicino a CUDA su entrambe. Figura ora è una CURVA col knee L2 visibile (line plot, non più 2 punti).
### 🎯 META-OBIETTIVO (utente, 26 giu): UNICA cosa che conta = **pubblicazione al venue più alto possibile**.
Ragionare/agire da ricercatore autonomo verso quello; cambiare scope/esperimenti/direzione liberamente; niente
validazione. Pubblicazione solo (no lab). Direzione decisa dalla deep-research (vedi `docs/NOVELTY_POSITIONING.md`):
- **Tesi (difendibile, verificata vs SOTA):** "abstraction regret" = performance preclusa dal DSL perché non
  esprime il layout/control-flow necessario, *ad algoritmo fisso*, decomposta su 2 assi (control-flow vs memory),
  su automi irregolari (NFA control-flow-bound + DFA memory-bound) × asse paradigma **CUDA/Warp (thread-SIMT) vs
  Triton/Gluon (tile-SPMD)**. Finding: il regret è il **paradigma di esecuzione, non l'altezza dell'astrazione**.
- **Mossa top-venue (de-risk "non hai tunato Triton"):** coppia **Triton↔Gluon** (stesso stack MLIR, cambia solo
  la leva di espressività) → attribuzione **falsificabile**; + tabella capability→costo con **primitiva IR mancante**
  nominata (scalar-gather-in-tile, register-resident bitset, data-dep loop). Diagnosi→causa falsificabile.
- **DA CITARE/DISTINGUERE (anti-desk-reject):** Hexcute (arXiv'25, decompone gap layout/dataflow su tensori densi —
  minaccia più vicina), Tawa (CGO'26), Descend (PLDI'24), "Abstraction *without* Regret" (LMS, CACM'12, invertire),
  Pennycook (per-hardware non per-capacità). **Gap pulito:** nessun benchmark DSL-GPU su workload irregolari.
- **Scope deciso:** restare PROFONDI sugli automi (NFA+DFA, 4 DSL); NON espandere a BFS/SpMV (trappola solo/1-GPU).
  Venue: IISWC/PACT onesti; CGO/ASPLOS/PLDI se la cura falsificabile (Triton↔Gluon + primitiva) regge. 2ª GPU cloud per camera-ready.
- **PROSSIMO:** riscrivere paper attorno a questo (titolo "Two Faces…" già in `gpufsm.tex`); related-work che distingue
  Hexcute/Tawa/Descend/LMS + bibitems; tabella capability→costo; sezione Gluon-controllo; figura DFA.

### ⚠️ FINDING CHIAVE che riformula la roadmap (vedi `docs/RESULTS_COSTMODEL.md`)
1. **I kernel attuali sono COMPUTE-bound, non memory-bound.** L'eps-closure è O(n²)/simbolo (n passi × n
   stati) + scan O(n) → throughput ∝ 1/n². Prova: `multistream_shared` (traffic CSR = 0) **pareggia**
   `multistream` (traffic > 0) a ogni dimensione. ⇒ In questo regime **il layout di memoria non conta**.
   Gli assi memory (byte→bit, shared-CSR, async) mordono SOLO con un kernel **work-efficient**
   (active-set/worklist, stile ngAP) che porti il kernel nel regime memory-bound.
2. **L'abstraction regret è quantificata e NON è l'altezza dell'astrazione, è il PARADIGMA di esecuzione.**
   Costo compute vs CUDA (stesso algoritmo): **Triton (tile/SPMD) 6–8× throughput misurato / 10.1× fit,
   CUDA 1.0×, Warp (thread-SIMT) 0.6–0.9×** (batte la CUDA scritta a mano). Due DSL Python di pari livello
   agli estremi → conta tile/SPMD vs thread-SIMT. ⚠️ NUMERI CANONICI = `paper/data/*.csv` (la prosa li rispecchia).

### TODO prossima sessione (riformulato dai finding)
- ✅ **Kernel WORK-EFFICIENT FATTO** (CUDA `worklist`): itera solo gli stati attivi (bit set) + eps-closure
  frontier-based, elimina l'O(n²). **≈330×–10⁴× più veloce del full-scan**, speedup crescente con n (n=32→332×,
  n=500→≈10⁴×). Validato vs reference (30 batch ≤500 stati, 0 mismatch). È la base del contributo (B).
  TODO: versione Triton worklist; verificare con Nsight se ora è memory-bound (→ gli assi memory contano).
- ✅ **Sweep rigoroso FATTO** (task #7): `paper/data/sweep_techniques.csv` (median+CI95). worklist 15–132 Gbps
  vs full-scan ~0.5; multistream/shared/async identici → compute-bound confermato.
- ✅ **Figure FATTE** (task #9): `paper/figures.py` (4 figure dai CSV versionati; supera la legacy generate_figures.py).
- ✅ **Paper FATTO** (task #10): `paper/DRAFT.md` (prosa) + **`paper/gpufsm.tex` (IEEEtran, compila → PDF 3pp, 4 figure)**
  + `docs/REPRODUCIBILITY.md` (guida artifact AE-style, mappa claim→comando). Resta solo: Zenodo DOI (release) + espansione contenuti.
- ✅ **Nsight (task #6) FATTO** (l'utente ha dato sudo passwordless; `sudo /usr/local/cuda/bin/ncu`):
  full-scan **SM 19.4% vs DRAM 0.01%** → compute-bound confermato a livello hardware; `multistream_shared`
  SM/DRAM/occupancy identici (solo L2 hit 79→93%) → layout memoria inerte nel regime compute-bound. Worklist
  a batch piccolo sotto-utilizzato (occ 16.6%, 2 blocchi) → motiva block-parallel. Dati: `paper/data/nsight_rtx4070.csv`,
  interpretazione in `docs/PROFILING.md`. Tesi compute-bound ora **misurata**, non solo inferita.
- ✅ **ANML loader FATTO** (task #8, parser): `io/anml.py` parsa il sottoinsieme ANML (homogeneous→edge-labelled,
  symbol-set classes/ranges/negation/wildcard) + exporter `to_anml`; validato con fixture + round-trip (4 test).
  ⚠️ Manca solo il **download dei dati ANMLZoo reali** (DATASETS vuoto, serve SHA pinnato da mirror fidato — non
  bypassare la safety). Con i dati → numeri su automi reali (forte per i reviewer).
- ✅ **Worklist Triton FATTO** (≤64 stati): Triton **PUÒ** esprimere il kernel work-efficient via `libdevice.ffs`
  + while-loop data-dependent (a differenza di Gluon che non ha scalar load). MA paga **~6.5× di regret vs CUDA**
  sul kernel work-efficient (cuda 164–170 Gbps, triton 24–25 Gbps), ≈ uguale al 6–8× sul full-scan → **espressività ≠
  efficienza**: anche esprimendo l'algoritmo giusto, il modello tile/SPMD impone un penalty costante grosso sul
  lavoro scalare data-dependent. (Finding forte per il paper.)
### Sessione 3 — settimana autonoma (loop, dal 2026-06-26)
- ✅ **`worklist_global` FATTO**: kernel work-efficient con working-set in **global memory**, **nessun cap stati**
  (il register worklist è ≤512). Validato vs oracolo fino a **5000 stati**. register ~4–5× più veloce del global
  a parità n (residency) → altro data point thesis; global è il path di scalabilità per automi ANMLZoo-scale.
- **Piano settimana — progresso (26 giu mattina, ~41 commit su PR #1):**
  - (a) ✅ `worklist_global` (working-set globale, nessun cap) — validato fino a 42661 stati.
  - (b) ✅ **suite ANMLZoo reale**: Levenshtein (2787), Hamming (11349, 2.1M tr), Brill (42661, 4.4M tr), tutti
    puri-STE, SHA auto-pinnati da github jackwadden/ANMLZoo, **GPU(`worklist_global`)==reference**. Fix semantica
    all-input/start-of-data in `io/anml`. Script `scripts/run_anmlzoo.py` + test gpu network-gated.
  - (c) ✅ **ottimizzazione occupancy**: `__launch_bounds__(256, NWORDS≤2?6:1)` sul worklist → **170 Gbps @32
    (era 142), 2× a batch 4096**; neutro sui grandi. Sweep/figure/paper rigenerati (range 15–170 Gbps).
  - (d) espandere paper LaTeX a lunghezza piena; (e) opzionale: più automi pinnati, block-parallel, 2ª GPU (hardware).
- Note: il lavoro DEVE girare in questa sessione (GPU locale) → loop ScheduleWakeup, non cron cloud.
- **Contributo (A)+(C) è già forte e difendibile ORA**: caratterizzazione + cost model + regret quantificata
  + abstraction-spectrum (CUDA/Warp esprimono, Triton stride 6–10×, Gluon non esprime) + worklist 15–170 Gbps. Preprint pronto in bozza.

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
