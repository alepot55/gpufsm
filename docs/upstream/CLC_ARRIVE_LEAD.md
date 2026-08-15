# Lead #2: l'arrive di `clusterlaunchcontrol.try_cancel` non ha una barriera propria

**Stato 16 ago: patch scritta e verificata, ma da mandare come ISSUE, non come PR** (vedi in fondo).

Trovato mentre si verificava l'invariante di #10035. **Verificato alla fonte, non dedotto.**

## Il fatto

`CLCTryCancelOpConversion` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp`) emette

```
@$2 clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [$0], [$1];
```

sotto `createElectPredicateWarp0`, cioè **un solo thread eletto della warp 0**, e non emette nessuna
`ttg::BarrierOp` prima. L'op non è in `containsLocalBarrier` (`lib/Analysis/Membar.cpp`).

È esattamente la forma che PR #10035 (*"always insert bar sync before all mbarrier arrives"*) è stata
scritta per proteggere: se il thread eletto fa avanzare l'mbarrier mentre le altre warp sono ancora
nel loop di attesa, le fasi divergono e il kernel si pianta.

## Perché oggi non esplode

I due soli creatori emettono un `BarrierExpectOp` **immediatamente prima**:

- `lib/Dialect/TritonNvidiaGPU/Transforms/CLCLowering.cpp` — `BarrierExpectOp::create(...)` poi
  `CLCTryCancelOp::create(...)`.
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp` — stessa coppia.

e `BarrierExpectOp` **sì** che porta la barriera implicita. Quindi la garanzia c'è, ma è
**incidentale all'op vicina**, non strutturale: basta che un passaggio futuro riordini, separi o
elimini l'expect perché sparisca in silenzio.

## La fix, se si fa

Rispecchiare #10035: emettere `ttg::BarrierOp(AddrSpace::Local)` in testa a `CLCTryCancelOpConversion`
e aggiungere `CLCTryCancelOp` a `containsLocalBarrier`, così l'analisi sa che quella barriera esiste
e non ne aggiunge una seconda. Costo: due righe più un lit test.

## Prima di aprirla

1. Verificare se un utente può creare `ttng.clc_try_cancel` direttamente (Gluon / MLIR a mano): se sì,
   il difetto è raggiungibile davvero e non solo teorico. Se no, si presenta come hardening, non come bug.
2. Serve SM100+ (Blackwell) per eseguirlo. La verifica statica (`triton-opt`) basta per il lit test;
   una prova a runtime no, e va detto nella PR invece di lasciarlo intendere.
3. Stesso destinatario dell'altra PR (area membar/barriere), quindi **aprirla solo dopo** che la prima
   è stata giudicata: due PR insieme dallo stesso outside contributor diluiscono l'attenzione.


---

# Nota: il bug #11111 (store fp8 trasposto) NON è un target

Indagato la notte del 16 ago, riprodotto su H100 con `main` di oggi (11 configurazioni, output
trasposto tutto Inf, contiguo pulito). Poi, leggendo il thread: **masahi il 3 ago aveva già scritto
che è un bug di `ptxas` 12.9** (che Triton usa ancora su Hopper) e che sparisce con ptxas 13.3.

Cosa resta di utile:
- Ipotesi "barriera insufficiente nella conversione di layout" (Triton usa `bar.warp.sync`, che è di
  convergenza e non di memoria, per le conversioni intra-warp — introdotta in #7810) **falsificata**
  con un esperimento: sostituita con la barriera CTA piena, la corruzione è identica.
- Il probe con valori distinguibili mostra che l'output trasposto contiene **dati estranei** (byte che
  non esistono nell'input), non una permutazione: coerente con un miscompile a valle del PTX.

**Regola da ricordare: leggere i commenti dell'issue PRIMA di investigare.** Qui la causa era già nel
thread e ho speso ~40 minuti di riproduzione per riscoprirla.


## Cosa è stato comunque contribuito su #11111

Commento postato con la triage che nel thread non c'era
(<https://github.com/triton-lang/triton/issues/11111#issuecomment-5304744938>):

| variante (XBLOCK=32, num_warps=1) | output trasposto |
|---|---|
| due store, e5m2 | corrotto (byte estranei 0x60, 0xFC) |
| **solo** lo store trasposto | pulito → servono entrambi gli store |
| due store, **e4m3** | corrotto → **non è specifico di e5m2** |
| due store, **int8** | pulito → è legato alla conversione fp8, non allo store a 8 bit |

più il fatto che la corruzione **non** è una barriera mancante (esperimento con la barriera CTA piena).


## Verificata, e poi riclassificata (16 ago, notte)

Patch pronta sul branch `membar-clc-arrive` (+9 righe, 3 file): barriera emessa nel lowering come per
`arrive_barrier`/`barrier_expect`/`tc_gen5_commit`, più `CLCTryCancelOp` in `containsLocalBarrier` e i
CHECK aggiornati (nel dialect LLVM la barriera è `nvvm.barrier`, non la forma PTX — primo tentativo
sbagliato proprio lì). Lit: **stesso identico set di fallimenti pre-esistenti**, nessuna regressione.

**Perché non diventa una PR:** la barriera aggiunta è ridondante in tutti i path attuali, perché il
`barrier_expect` adiacente la porta già. Quindi oggi costa e compra solo robustezza contro refactor
futuri — e il costo non è misurabile senza hardware SM100. Proporla come patch invita un "no"
ragionevole. Va mandata come **issue** che descrive la fragilità, offrendo la patch se la vogliono.
