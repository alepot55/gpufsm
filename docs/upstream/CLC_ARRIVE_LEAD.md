# Lead #2: l'arrive di `clusterlaunchcontrol.try_cancel` non ha una barriera propria

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
