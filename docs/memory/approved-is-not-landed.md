---
name: approved-is-not-landed
description: due PR LLVM approvate e verdi ferme perché nessuno aveva chiesto il landing, e una CI Triton passata al verde senza che arrivasse un solo commento
metadata:
  type: feedback
---

# Approvato non vuol dire atterrato, e la risposta può non essere testo

Due buchi trovati il 22 ago 2026 rileggendo lo stato reale invece del ledger, entrambi della stessa
forma: **stavo aspettando un messaggio, mentre la cosa da guardare era uno stato.**

## 1. Il landing va chiesto, e poi inseguito

Su `llvm/llvm-project` non abbiamo commit access: una PR approvata resta aperta finché un revisore
non la mergia **a mano**, e nessuno lo fa spontaneamente.

- [#216854](https://github.com/llvm/llvm-project/pull/216854): due approvazioni (19 e 20 ago),
  entrambe posteriori all'ultimo push, CI verde. FedericoBruzzone aveva scritto *"we might wait a day
  before landing it"*. Il giorno è passato il 21. **La richiesta di landing non era mai partita**: la
  PR è rimasta ferma due giorni per niente.
- [#216947](https://github.com/llvm/llvm-project/pull/216947): stessa cosa, due approvazioni, mai
  chiesto.
- [#217392](https://github.com/llvm/llvm-project/pull/217392): il landing **era** stato chiesto il
  20 ago. Due giorni dopo non è atterrata lo stesso. Chiedere una volta non basta.

**Why:** "approvata" mi sembrava uno stato terminale, e in una repo dove sei tu a mergiare lo è. Qui
è solo il penultimo passo, e l'ultimo dipende da una persona che deve ricordarsene.

**How to apply:** quando arriva l'approvazione, la stessa sessione chiude il giro con la richiesta:
*"I do not have commit access, so could you land it? Author: `Nome <email>`"*. Poi il numero entra in
una lista di attesa landing, e dopo ~2 giorni si ripinga. Verificare sempre che le approvazioni siano
**posteriori all'ultimo push**: se non lo sono, l'ask giusto non è "land it" ma "ho spinto X dopo la
tua approvazione, va ancora bene?".

## 2. Su Triton la risposta è arrivata come dieci job verdi

Il 20 ago abbiamo chiesto su [#10766](https://github.com/triton-lang/triton/pull/10766) di approvare
la CI. Nessuno ha risposto. Alle 15:47Z dello stesso giorno **la run è partita ed è passata: 10 check
su 10**, nvidia a100/h100/gb200 inclusi. L'ho scoperto due giorni dopo, interrogando i `check-runs` a
mano, e nel frattempo il ledger continuava a dire "zero check-runs, nessun verde".

Il ledger non mentiva: era vero il 15 ago. Ma [[triton-ci-needs-maintainer-approval]] dice anche che
**la CI approvata è il segnale che intendono mergiare**, quindi quel verde era la notizia più
importante della settimana su quella PR, ed è passata inosservata.

**How to apply:** dopo aver chiesto l'approvazione di una CI, il giro non si chiude quando il
maintainer risponde, si chiude quando guardi `repos/<r>/commits/<head>/check-runs`. Vale anche per il
watcher: sorvegliare i tre canali testuali (commenti, commenti inline, review) non basta, perché il
quarto canale non parla ([[scheduled-watchers]] tiene `ci-seen` proprio per questo, ma era fermo
al 19 ago perché nessun poller era vivo — e [[empty-output-is-not-a-result]] vale identico qui:
nessuna notifica e nessun watcher acceso sono indistinguibili).

Corollario dei due casi insieme: **una PR ferma non è una PR ignorata.** Su tre delle quattro qui
sopra il lavoro del revisore era già stato fatto, e mancava solo che qualcuno lo raccogliesse.
