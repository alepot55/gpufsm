# Un binario mancante non è un test rosso (e come me ne sono accorto due volte in un'ora)

Il 21 ago 2026, verificando la fix di whutsunxu per triton#11393, ho scritto un controllo di
regressione così:

```
FC=$(find /root/.triton/llvm -name FileCheck | head -1)
$OPT $T $PIPE | $FC $T && echo VERDE || echo ROSSO
```

`find` non trovava niente, `$FC` era la stringa vuota, la pipe falliva e il gate stampava
**ROSSO**. Non era un test fallito: era un binario assente. FileCheck stava in
`python/triton/FileCheck` dentro il tree, non sotto `/root/.triton/llvm`. Col path giusto:
verde su `test/Conversion/tritonnvidiagpu_to_llvm.mlir` e `test/Analysis/test-membar.mlir`.

È **la stessa forma** dell'errore già registrato in [[il-gate-gira-solo-dal-venv]]: `command not
found` che si traveste da gate rosso. La differenza è che lì il comando mancava nel PATH, qui la
variabile che lo conteneva era vuota. Regola operativa: **se un gate diventa rosso, prima stampa
il path del binario che stai invocando.** Una riga di `echo` separa "il codice è rotto" da "il
tuo comando non esiste", e le due cose portano a decisioni opposte.

Lo stesso giorno la stessa trappola era già costata di più: tre build Modal falliti con
`exit status 127` da `cmake --build`, che ho letto come errore della patch finché non ho
guardato il codice di uscita. 127 = comando non trovato; `CMAKE_MAKE_PROGRAM` nella CMakeCache
puntava a `/usr/local/bin/ninja`, sparito quando ninja è passato nel venv. E siccome nello script
il build non gateava la probe, le misure giravano sul **binario vecchio** e sembravano risultati
veri: tre bracci sperimentali con numeri identici, che ho quasi riportato come "ipotesi refutata".

Le due difese che ho poi messo nello script e che vanno tenute:

1. **la probe non parte se il build è rosso** (`if build X; then probe X; else echo saltata; fi`);
2. **ogni probe stampa `md5sum` del binario** che sta per misurare. Se due bracci hanno lo stesso
   md5, non hai due bracci. Vedi [[empty-output-is-not-a-result]].

Corollario su `git apply --3way`, usato da `scripts/modal_triton.py`: implica `--index`, quindi
la patch finisce **staged**. `git diff --stat` esce vuoto e sembra un tree pulito; serve
`git diff --cached` per vedere cosa è stato davvero applicato.
