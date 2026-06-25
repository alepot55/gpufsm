# gpufsm

**Portable GPU finite-state-machine (NFA) processing — Triton vs CUDA, done rigorously.**

`gpufsm` is a minimal, extensible framework and study of automata/regex processing on GPUs. It compares a
high-level block-based DSL (**OpenAI Triton**) against hand-written **CUDA** on *irregular* finite-state-machine
workloads, where the abstraction cost is largest.

**Thesis (memory-centric):** for irregular GPU workloads, *memory organization* — not algorithmic
complexity — is the primary performance determinant; a DSL's abstraction matters mainly insofar as it
constrains the memory layout you can express ("abstraction regret"). We quantify how much of the
Triton↔CUDA gap closes by reorganizing memory alone (bit-packing, shared-resident transition tables,
coalesced multi-stream layout), at equal algorithm.

## Install

```bash
pip install -e ".[dev]"          # core (CPU) + dev tools, no GPU needed
pip install -e ".[dev,triton]"   # add the Triton backend (needs a GPU)
# CUDA backend: build the extension explicitly (needs CUDA toolkit + GPU)
GPUFSM_BUILD_CUDA=ON pip install -e ".[dev]"
```

The build is **graceful**: with no CUDA toolkit/GPU the package still installs and runs on CPU/Triton.

## Quickstart

```python
from gpufsm import NFABuilder, Backend, run, benchmark

b = NFABuilder()
s0 = b.add_state(); s1 = b.add_state(accept=True)
b.set_start(s0); b.add_transition(s0, "a", s1)
nfa = b.build()

run(nfa, b"a", backend=Backend.CPU)        # Result(accepted=True, match_len=1, ...)
benchmark(nfa, b"a" * 4096, repeats=10)    # BenchmarkStats(mean/std/ci95)
```

CLI:

```bash
gpufsm env        # environment + available backends
gpufsm list       # backends and techniques
gpufsm verify     # check every backend agrees with the CPU reference
gpufsm bench --backend cpu --size 4096 --repeats 10
```

## Design

- **One API** (`gpufsm.api`): `run` and `benchmark`, dispatched through a `(Backend, technique)` **registry**.
- **One correctness oracle** (`gpufsm.reference`): a CPU NFA simulator (latch-first-match). Every backend
  must reproduce its `accepted` / `match_len` on every case (`pytest -m "not gpu"` enforces it on CPU; the
  GPU backends are checked under `-m gpu`).
- **One NFA representation** (`gpufsm.nfa`): CSR for symbolic + epsilon transitions, shared by all backends.
- **Extensible**: a new backend/technique is one file plus one `@register(Backend.X, "name")` line.

## Status

v0.1 foundation: CPU reference, public API, registry, CLI, packaging and CI are in place and green.
Triton and CUDA backends are being ported from the prior `triton_vs_cuda_fsm` study. See `CLAUDE.md` for the
project context and decisions, and the development plan for the full roadmap.

## License

MIT — see [LICENSE](LICENSE).
