# LICM affine solleva un'op che cattura un valore del ciclo (PR llvm#216605)

**Stato:** [llvm/llvm-project#216605](https://github.com/llvm/llvm-project/pull/216605), aperta il
16 ago 2026. +81/-15 su 2 file. Chiude la issue upstream #216545.

## Il difetto

`isOpLoopInvariant` (`mlir/lib/Dialect/Affine/Transforms/AffineLoopInvariantCodeMotion.cpp`) entra
nelle region di **tre** sole op (`affine.if`, `affine.for`, `affine.parallel`); per ogni altra op con
region guarda solo `op.getOperands()`. I valori che una region **cattura** non sono operandi dell'op
che la contiene, quindi non li controlla nessuno.

Risultato: un `scf.for` con bounds/step/iter_args invarianti viene sollevato fuori dall'`affine.for`
anche quando il suo corpo legge un valore definito dentro il ciclo. Il verificatore lo rifiuta:
`error: operand #0 does not dominate this use`.

## Verifica (eseguita, non dedotta)

| controllo | esito |
|---|---|
| bug riprodotto su `main` pulito (7cb5d89) | ✅ |
| i 2 test nuovi falliscono **senza** patch | ✅ |
| passano **con** patch | ✅ |
| `Dialect/Affine` + `Transforms` | ✅ 163/163 |
| **suite MLIR completa** | ✅ **liste dei falliti IDENTICHE** (660 = 660, per binari non compilati) |
| clang-format stile LLVM | ✅ |

## La correzione

Estende ai valori catturati dalle region le **stesse tre condizioni** che il controllo sugli operandi
gia' applica (induction var, iter_args, op definita nel ciclo e non sollevata), via
`visitUsedValuesDefinedAbove`. `MLIRTransformUtils` era gia' collegato: nessuna modifica di CMake.

## Perche' questo bersaglio e non l'altro

L'altra issue candidata (#216542, `control-flow-sink` e validita' affine) e' **contesa per disegno**:
nessun pass generico di MLIR conosce le regole affini, quindi "chi deve cedere" e' una discussione
aperta. E' la forma esatta che ci ha fatto respingere #10766 e #10774 su Triton. Questa invece e' una
violazione di dominanza: la rifiuta il verificatore, non un'opinione.

## Trappole ripagate (erano gia' scritte e ci sono ricascato)

- ⚠️ **Lo scratchpad di sessione sta sotto `/tmp`**, che qui e' un tmpfs da 2 GB: clonarci
  llvm-project lo ha saturato e il push e' fallito con ENOSPC (mascherato da errore generico).
  I checkout grandi vanno in `~/.cache/llvmwt/`.
- ⚠️ **`| tail` in un comando lungo nasconde l'avanzamento**: `tail` non emette nulla finche' il
  processo non termina, e una build viva sembra piantata. Usare `modal app logs`.
- ⚠️ **I test MLIR stanno in `build/tools/mlir/test`**, non in `build/test` (quelli sono di LLVM).
  Puntare lit sul percorso sbagliato da' "0 falliti" su 0 test trovati: un verde che non misura nulla.
- ⚠️ Il push al fork appena creato va in timeout: GitHub sta ancora copiando il repo.

## Harness

`scratchpad/modal_llvm.py` (stessa forma di `scripts/modal_triton.py`): alberi `main` e `fix` nel
Volume `llvm-upstream`, build CPU-only con ccache. Prima build ~45 min (clone + cmake + ninja a
freddo), le successive sono incrementali. Le stampe remote non arrivano al terminale: la funzione
`run` **restituisce** il testo invece di stamparlo.
