---
name: empty-output-is-not-a-result
description: Un output vuoto non e' un risultato negativo — va sempre verificato stderr prima di concludere "non si puo' fare"
metadata:
  type: feedback
---

Il 17 ago ho scritto a un maintainer Triton che **non era possibile** salvare un test dal fold di
`split(join(x))`: avevo provato tre varianti e il pass non produceva nulla. Il 18 ago ho scoperto che
avevo scritto `%0b` come nome SSA, che in MLIR e' invalido (dopo `%0` si aspetta `=`). `triton-opt`
rifiutava l'intero chunk e stampava l'errore su **stderr**, che avevo soppresso con `2>/dev/null`.
Le varianti funzionavano tutte.

**Why:** ho trasformato un mio errore di battitura in una conclusione tecnica pubblicata, su una PR
gia' ferma da sei settimane. Il costo non e' stato il typo: e' stato aver chiesto al maintainer di
scegliere fra due opzioni quando non c'era nessuna scelta da fare.

**How to apply:**

- Un comando che **non stampa nulla** non ha "risposto no". Ha risposto *niente*. Prima di dedurre
  qualcosa, rilanciare **senza** `2>/dev/null` e leggere stderr.
- Vale in particolare quando si conta (`grep -c` che da' `0`): zero occorrenze e input rifiutato
  producono lo stesso numero.
- Prima di scrivere "non si puo' fare" a qualcuno, verificare che il caso **positivo** noto funzioni
  ancora nello stesso harness. Se il controllo che dovrebbe passare non passa, il problema e' il banco
  di prova, non l'ipotesi.
- Vedi anche [[upstream-review-dynamics]]: correggere in pubblico costa una riga, lasciare in piedi
  una conclusione sbagliata costa settimane.
