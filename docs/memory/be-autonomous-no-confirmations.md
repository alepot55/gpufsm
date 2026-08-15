---
name: be-autonomous-no-confirmations
description: Non chiedere il permesso per i passi operativi (PR, merge, install, setup infra) — falli e riporta
metadata:
  type: feedback
---

Detto esplicitamente il 2026-08-15: aprire una PR interna, mergiarla, installare una dipendenza,
configurare Modal e simili **non si chiedono**. Si fanno e si riporta il risultato.

**Why:** chiedere conferma su passi reversibili e già dentro il perimetro del task rallenta e sposta
sull'utente lavoro che è mio. La soglia dell'utente per "chiedi prima" è alta: decisioni
architetturali o cambi di direzione, non esecuzione.

**How to apply:** eseguire il passo, poi riportarlo in una riga. Restano da chiedere solo le cose
davvero irreversibili o fuori scope (es. premere SUBMIT su una submission, cancellare dati, spendere
soldi in modo non banale). Le PR interne su `alepot55/gpufsm` si aprono **e si mergiano** da sé, come
tutte le 23 precedenti. Vedi [[memory-lives-in-the-repo]].
