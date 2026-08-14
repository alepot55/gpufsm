# Upstream Triton, stato al 2026-08-14 (i due commenti sono stati POSTATI)

I draft che stavano qui sono stati rivisti e pubblicati da sessione locale con `gh`. Questo file
ora registra cosa e' stato fatto e cosa resta.

## Cosa blocca davvero le PR: la CI non parte da sola

Verificato oggi: `triton-lang/triton` gira con "require approval for all outside collaborators".
Ogni push di un contributor esterno mette la run in `action_required` finche' un maintainer non
clicca **Approve and run workflows**. Non e' la variante "primo contributo": alepot55 era gia'
stato approvato tre volte e viene comunque gated.

Conseguenze operative:

- **Ogni push cancella il verde**, incluso il merge di `main` per rinfrescare il branch. Il refresh
  del 14 ago ha azzerato la CI verde che #10766 aveva dal 2 luglio. Fare tutti i push in una volta,
  poi congelare il branch e pingare. lezcano su #10875: "ping if necessary please, but don't update
  the branch as otherwise we'll need to rerun all tests again".
- Le run pendenti si trovano con
  `gh api 'repos/triton-lang/triton/actions/runs?status=action_required'` e vanno citate per URL nel
  commento, cosi' il maintainer non deve cercarle.
- Latenza reale dell'approvazione: una run su `fold-split-join` creata il 2 lug 21:50Z e' partita il
  6 lug 04:31Z.

`.github/CODEOWNERS` non ha regole per `lib/Dialect/Triton/IR/`, `test/` o `examples/`: cadono tutte
su `*  @ptillet`, che e' auto-assegnato ma non ha review ne' merge dal 17 giugno. Mai indirizzare il
ping al reviewer auto-assegnato. Chi mergia davvero (ultime 5,5 settimane): Jokeren 15, ThomasRaoux 9,
peterbell10 9, lezcano 7; canonicalizzazioni di dialect a ThomasRaoux, `examples/plugins/` a Jokeren,
semantica split/join a neildhar dopo #10749.

## PR #10766 (fold split/join)

Descrizione riscritta e commento di ping postato a @ThomasRaoux (cc peterbell10, neildhar). Cosa e'
cambiato rispetto al draft:

- **Use case misurato**, che e' la cosa che Raoux aveva chiesto il 1 luglio e che mancava: un helper
  `@triton.jit` che ritorna `tl.join(cheap, tl.dot(x, w))` e un chiamante che fa `tl.split` e usa
  solo la meta' economica. Senza il fold il round-trip tiene vivo il `tt.dot` morto e il suo
  staging in shared memory: 16 KB smem, 78 vs 40 registri, 224 vs 72 istruzioni SASS, 31,5 vs
  23,2 us (1,35x, 3 run entro l'1%). Col fold il kernel compila identico alla versione scritta a mano.
- **Caveat onesto tenuto nel testo**: se la meta' morta e' solo una load, ptxas la toglie da solo e il
  SASS e' identico. Vale solo il caso che trascina roba nell'allocatore di shared memory.
- Isolamento del meccanismo su un `triton-opt` senza fold (`dce.mlir`, due funzioni identiche a meno
  del round-trip): in `@before` la load morta sopravvive a `--canonicalize`, in `@after` no.
- Interazione con **#10749** (rilassa `JoinOp::verify` / `SplitOp::isCompatibleReturnTypes` a
  `ignoreRegBroadcast=true`): i fold confrontano i tipi per uguaglianza esatta, quindi sono
  strettamente piu' stretti del verifier nuovo e declinano sui round-trip che differiscono solo per
  register broadcast. Stesso schema di `TransOp::fold`, che lascia il caso layout-differente a
  `CanonicalizeConvertFromTranspose`.
- Ritirata l'offerta di chiudere (era l'ultima parola sul thread e su una PR ferma da 45 giorni
  suonava come consenso).
- Thread di review di peterbell10 (ormai outdated) risolto.
- Offerte nel commento, non pushate: test FileCheck con encoding che fissano il guard di tipo, e la
  variante di `ws_data_partition.mlir` con due reshape fra join e split che conserva le 4 CHECK
  originali invece di sostituirle con CHECK-NOT.

Run da approvare: https://github.com/triton-lang/triton/actions/runs/31785463900

## PR #10780 (docs plugin examples) e PR #11311 (nuova)

Il difetto vero e' stato **estratto in una PR separata**: il blocco python dell'**Example 4** di
`examples/plugins/README.md` non parsa su main (`IndentationError`, guardie di early-return a 2 spazi
in corpi a 4), difetto entrato con #8401 e mai corretto; gli stessi due blocchi usano `pathlib` e
`hashlib` senza importarli. `#11311` = solo quello, un file, +9/-5, verificato con `compile()` su
tutti e 4 i blocchi e con l'esecuzione del blocco Example 2 (RC=0 prima e dopo).

⚠️ Correzione di merito rispetto ai draft e ai commenti vecchi: e' l'**Example 4**, non l'Example 3.
Il commento postato apre correggendo il record.

#10780 resta aperta con la sola parte contestata (precompute di key/hash), descrizione ripulita: il
numero "38,9 -> 12,4 us" e' stato rimosso perche' veniva da un runtime patchato in locale, non da
Triton stock. Offerto a Jokeren/CRobeck di chiuderla se non vogliono il precompute.

Run da approvare: https://github.com/triton-lang/triton/actions/runs/31785470864

## Regola per la prossima sessione

Non pushare piu' su `fold-split-join` ne' su `hook-key-cache` senza un motivo forte: ogni push
richiede un nuovo click di approvazione della CI. Monitorare le risposte e rispondere, non ritoccare
i branch.
