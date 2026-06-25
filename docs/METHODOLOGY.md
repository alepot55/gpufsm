# Methodology

## Problem

We study **NFA simulation** on GPUs: given a non-deterministic finite automaton (symbolic + epsilon
transitions, accept states) and an input byte stream, decide acceptance under **latch-first-match**
semantics (report at the first accepting state, returning the matched-prefix length). This is the
irregular, latency-bound kernel underlying regex/pattern matching (Snort, ClamAV, Protomata, …).

## What we compare

The same NFA, in the same CSR representation, executed by:

- **CPU reference** (`gpufsm.reference`) — set-based simulator; the correctness oracle.
- **CPU bit-packed** (`gpufsm.bitmap`) — packed-bitmask simulator; the executable spec of the GPU
  bit-packed kernels and the concrete embodiment of the memory-centric thesis.
- **Triton** — high-level block-based DSL kernels (ported; validated on GPU).
- **CUDA** — hand-written low-level kernels (ported; validated on GPU).

Every backend must reproduce the oracle's `(accepted, match_len)` on every case (enforced by the test
suite; see [REPRODUCIBILITY](REPRODUCIBILITY.md)).

## Central thesis (memory-centric)

For irregular GPU workloads, **memory organization — not algorithmic complexity — is the primary
performance determinant**; a DSL's abstraction matters mainly insofar as it constrains the memory layout
it can express ("abstraction regret"). Concretely we quantify, at *equal algorithm*, the effect of:

1. **State-set packing**: 1 bit/state (packed bitmask) vs the legacy 4 bytes/state (≈31× at 500 states).
2. **Transition-table residency**: read-only CSR resident in shared memory / block vs global memory.
3. **Multi-stream coalescing**: `[stream × state]` bitmatrix, states-major vs streams-major.
4. **Transfer amortization**: synchronous `cudaMemcpy` vs pinned + persistent buffers + async + batching.

The key experiment is a **memory-level ablation**: starting from the naive layout, enable one memory
optimization at a time and attribute the closed fraction of the Triton↔CUDA gap to each.

## Metrics

- **Kernel time** (ms): GPU compute only — the scientific quantity.
- **Transfer time** (ms): host↔device movement — reported separately, never folded into kernel time.
- **Throughput** (Gbps) for multi-stream; **latency** for single-stream.
- Each measurement: `warmup` runs discarded, then `repeats` runs → **mean / std / 95% CI** (`gpufsm.bench`).

## Baselines and benchmarks

- Baselines to cite/beat: ngAP (ASPLOS'24), BitGen (MICRO'25), AsyncAP, iNFAnt/iNFAnt2, DFAGE, Hyperscan
  (CPU SIMD, NSDI'19).
- Benchmarks: ANMLZoo (IISWC'16), AutomataZoo (IISWC'18) — Brill, ClamAV, Snort, Protomata, Yara, etc.
  (80–500 states).
