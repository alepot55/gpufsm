---
name: triton-rejects-trivial-prs
description: Su triton-lang/triton i fix piccoli vengono chiusi come "trivial"; passa solo ciò che ha impatto misurato o un bug riproducibile
metadata:
  type: project
---

Jokeren, che è CODEOWNER dell'area membar, chiude le PR piccole con parole sue:
"It doesn't seem necessary to me. Note that we don't typically accept a trivial PR like this."
(#11226) e "This is trivial" (#11075). Stesso esito su #11180 e #11214.

**Why:** la tattica "fix piccolo e indiscutibile" sembrava confermata da #11311 (mergiata in <24h),
ma #11311 non è passata perché era piccola: è passata perché **riparava un esempio rotto**, cioè un
difetto reale. Piccolo e *inutile* viene rifiutato; piccolo e *necessario* viene mergiato.

**How to apply:**
- Il criterio di selezione non è la dimensione della patch, è: *esiste un comportamento sbagliato o un
  costo misurabile che sparisce con questa patch?* Se la risposta è no, non aprirla.
- Corollario sulle nostre due PR: [#11324](https://github.com/triton-lang/triton/pull/11324) toglie 30
  barriere misurate ⇒ regge il criterio. [#11323](https://github.com/triton-lang/triton/pull/11323) non
  cambia il codice generato ⇒ **rischia di essere chiusa come "not necessary"**, ed è giusto saperlo
  invece di scoprirlo in review.
- La coda delle issue è **satura**: al 16 ago ogni issue aperta non banale (#10924, #11146, #10987,
  #11013, #10977) aveva già una PR collegata entro giorni. Cercare lì un bersaglio libero è tempo perso;
  meglio partire da un difetto trovato da noi leggendo il codice, come è stato per le due membar.

Vedi [[upstream-contribution-method]] e [[triton-ci-needs-maintainer-approval]].
