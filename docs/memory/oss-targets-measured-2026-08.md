---
name: oss-targets-measured-2026-08
description: Sette progetti OSS misurati sugli stessi numeri il 18 ago 2026 - restare su LLVM, riserva wasmtime, e perche' gli altri cinque sono fuori
metadata:
  type: reference
---

Workflow `wf_9b4f08fa-f51` (8 agenti, ~950k token). Misurato con `gh`, non stimato. Rifarlo costa
caro: leggere qui prima di riaprire la domanda "dove conviene contribuire".

Metriche: latenza mediana apertura → **prima revisione umana di un esterno**; % di PR di esterni
chiuse che risultano fuse; PR di esterni aperte da **oltre 30 giorni senza uno sguardo**; offerta di
bug giudicati da una macchina (crash/verificatore/miscompile) aperti e non assegnati negli ultimi
90 giorni; e **se riusciamo a compilarlo**, che e' un requisito duro: quello che non sappiamo
costruire non lo sappiamo verificare.

| progetto | 1a revisione | fusi | marci >30gg | bug | build |
|---|---|---|---|---|---|
| llvm/llvm-project | 0.52 g | 67% | **942** | 463 | SI |
| openxla/xla | 0.55 g | 77% | 0 | 113 | **non misurata** |
| pytorch/pytorch | 0.65 g | 48% | 186 | 154 | parziale |
| iree-org/iree | 1.74 g | 88% | 28 | 15 | parziale |
| bytecodealliance/wasmtime | **0.28 g** | 81% | 6 | 8 | SI |
| ggml-org/llama.cpp | 0.22 g | 57% | **354** | 142 | SI (4 min) |
| apache/tvm | 0.53 g | 76% | 12 | 28 | SI |

**Decisione: restare su LLVM, nessun secondo fronte.** L'impianto di build e' gia' pagato
(`scripts/modal_llvm.py`), abbiamo 5 PR aperte e la prima revisione e' arrivata in 19 minuti.

**Cosa ha squalificato gli altri, un numero ciascuno:**

- **openxla/xla** — richiede un **CLA di Google che l'utente deve firmare di persona**, e il tempo
  di build a freddo non e' misurabile da noi perche' tutta la loro CI gira su RBE interna. Nota
  metodologica utile: li' `is:merged` non significa niente, le PR degli esterni atterrano via
  copybara e risultano *chiuse*; il tasso vero di atterraggio e' 77%.
- **pytorch/pytorch** — 48% di merge e tre correzioni di correttezza chiuse in poche ore con testo
  di circostanza e zero revisione tecnica.
- **iree-org/iree** — 9 mergiatori in 30 giorni, tutti della stessa azienda, e una persona sola fa
  26 merge su 60. Fattore-bus di uno su un'offerta di 15 bug.
- **apache/tvm** — 8 mergiatori, e `tqchen` da solo ha chiuso 12 PR di esterni, 4 senza spiegazione.
- **ggml-org/llama.cpp** — 354 PR di esterni aperte da oltre un mese senza uno sguardo, e nessun bot
  che le chiuda. In piu' i test dei kernel ggml, cioe' la parte che darebbe valore al nome,
  richiedono due backend: proprio quella non la possiamo verificare.
- **bytecodealliance/wasmtime** — **non squalificato, e' la riserva.** Batte LLVM su tutto (0.28 g,
  81%, nessun CLA, nessun DCO, nessuna barriera CI per i nuovi) ma ha solo 8 bug del tipo giusto e
  i manutentori li chiudono in 3.5 giorni mediani: correremmo contro di loro. ⚠️ La loro policy
  **vieta l'uso di AI sulle issue etichettate "good first issue"**.

**Trigger deciso ora per non rimetterlo in discussione:** se entro il **1 set 2026** meno di 2 delle
5 PR LLVM sono atterrate, aggiungere wasmtime (Cranelift e' un backend di compilazione, il lavoro
su SSA/codegen si somma invece di biforcarsi).

**Il numero scomodo su LLVM:** 942 PR di esterni aperte da oltre 30 giorni senza revisione umana,
la piu' vecchia da 976 giorni. La distribuzione e' bimodale: o ti guardano subito o non ti guardano
mai. Noi siamo finiti nel ramo buono, ma e' esposizione a un progetto solo.

Vedi [[llvm-pr-register-short-and-staggered]] per come scrivere le PR una volta scelto il bersaglio
e [[pick-uncontested-bugs-not-design-changes]] per quali bug prendere.
