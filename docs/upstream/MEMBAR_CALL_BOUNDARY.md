# Barriera mancante al confine di chiamata (issue #11326)

**Stato:** [issue #11326](https://github.com/triton-lang/triton/issues/11326) aperta il 16 ago 2026.
Aperta come **segnalazione, non PR**, di proposito: vedi "perche' non una PR" in fondo.

## Il difetto

L'analisi membar modella una `tt.call` **solo con lo stato di uscita** della funzione chiamata
(`Membar.cpp:250-262`, `funcMap` contiene solo il summary di uscita, `Function.h:131-154`). Quindi
cio' che il chiamante ha lasciato in sospeso non viene mai confrontato con il **primo** accesso della
funzione chiamata. Se si sovrappongono, nessuna barriera.

In piu' `getScratchBufferId` esclude esplicitamente `triton::CallOp` (`Membar.cpp:131-137`), quindi la
chiamata non e' modellata nemmeno come operazione con scratch.

## Verifica (eseguita, non dedotta)

`docs/upstream/repro_membar_call_boundary.mlir` su `main` c346e50c7b:
- scratch del chiamante: `allocation.offset = 0`, `allocation.size = 8192`
- frame della `tt.call`: `allocation.offset = 0` → **stessi byte**
- barriere nel chiamante: **0**

La direzione opposta e' gia' coperta in-tree (`test/Analysis/test-membar.mlir:1497-1500`): uno scratch
op *dopo* la call che aliasa il frame ottiene la barriera. Spostarlo *prima* scopre il buco.
Lo stesso pericolo dentro una singola funzione e' catturato (`test-membar.mlir:91-105`) ⇒ **l'inlining
cambia la correttezza del programma**.

## Raggiungibilita'

`tt.call` sopravvive fino a TritonGPU solo con `@triton.jit(noinline=True)`, gia' esercitato in-tree
da `test_noinline[shared]`. Un chiamante che fa una conversione di layout, una riduzione o un `tl.dot`
prima di chiamare un helper `noinline` e' codice ordinario.

## Perche' NON una PR

1. La correzione tocca il **disegno** della funzione di trasferimento interprocedurale: o si modella la
   call come scrittura conservativa dell'intero frame, o si dota ogni funzione di un summary di
   *ingresso* (accessi prima del primo punto di sincronizzazione). Sono due scelte diverse con costi
   diversi, e non tocca a noi indovinare.
2. Tre PR aperte dallo stesso contributor esterno **nello stesso file** diluiscono la review
   ([[upstream-contribution-method]]). #11325 e' gia' aperta su `Membar.cpp`.
Nella issue e' offerta l'implementazione di entrambe le varianti.

## Provenienza

Terzo dei tre candidati sopravvissuti all'audit `wf_97378bab-2c6`, verificato a mano dopo
[[MEMBAR_SUBSLICE_COORDS]]. Resta non verificato il secondo (multicast TMA cross-CTA).
