# gpufsm

**Where does a GPU DSL's abstraction actually cost you? Measured on automata, across four DSLs.**

`gpufsm` simulates finite automata on the GPU under **CUDA**, **Triton**, **NVIDIA Warp** and
**Gluon**, at a fixed algorithm, and measures what each abstraction precludes.

**Finding — the regret follows the execution paradigm, not the height of the abstraction.**
Arrange the four DSLs on a 2x2 of *abstraction height* against *execution paradigm* and the
cost tracks the column, not the row: CUDA (low-level, thread) 1x, Warp (high-level, thread)
0.6-0.9x, Triton (high-level, tile/SPMD) 6-8x, Gluon (low-level, tile) cannot express the
kernel at all. Two workloads pin down two faces of it — NFA simulation is control-flow-bound,
DFA simulation is memory-bound — and Triton pays on both, which is what rules out "it is the
workload" and leaves "it is the model".

## Install

```bash
pip install -e ".[dev]"                                  # core (CPU) + dev tools, no GPU needed
pip install -e ".[dev,triton]"                           # + the Triton backend (needs a GPU)
pip install -e "." --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON   # + the CUDA extension
```

The build is **graceful**: with no CUDA toolkit or GPU the extension is skipped and the
package still installs and runs on CPU (and on Triton, if present). Backends that cannot load
report as unavailable rather than disappearing.

## Quickstart

```python
from gpufsm import NFABuilder, Backend, run, benchmark, random_dfa, run_batch

b = NFABuilder()
s0 = b.add_state(); s1 = b.add_state(accept=True)
b.set_start(s0); b.add_transition(s0, "a", s1)
nfa = b.build()

run(nfa, b"a", backend=Backend.CPU)          # Result(accepted=True, match_len=1, ...)
benchmark(nfa, b"a" * 4096, repeats=10)      # BenchmarkStats(mean/std/ci95)

dfa = random_dfa(4096, seed=0)               # the memory-bound face, same API
run_batch(dfa, [b"abc", b"xyz"], backend=Backend.CUDA)
```

```bash
gpufsm env        # environment + available backends, per automaton kind
gpufsm list       # every (backend, kind, technique)
gpufsm verify     # check every backend agrees with the CPU reference
gpufsm bench --backend cpu --size 4096 --repeats 10
```

## Design

- **One API** (`gpufsm.api`): `run`, `run_batch`, `benchmark`. NFAs and DFAs both go through
  it; the automaton's type selects the kind.
- **One extension point** (`gpufsm.core.registry`): a backend or technique is one module plus
  one `@register(Kind, Backend, "name")` line.
- **One oracle** (`gpufsm.reference`): a CPU simulator with latch-first-match semantics. Every
  backend must reproduce its `accepted`/`match_len`, and no measurement reports a throughput
  before that check passes.
- **One harness** (`gpufsm.bench`): the random automata, the timing statistics (median +
  bootstrap CI95 — kernel timings are not gaussian), the CSV schema and the nvcc helper that
  `scripts/` and `experiments/` share.

## Layout

| Path | What |
|---|---|
| `src/gpufsm/` | the library: `core/`, `reference.py`, `api.py`, `backends/`, `bench/` |
| `tests/` | CPU suite plus GPU-marked tests; `test_golden.py` pins the oracle verdicts |
| `scripts/` | measurement entry points and the Modal GPU drivers |
| `experiments/cure/` | the provenance of the numbers in `paper2/` |
| `paper/`, `paper2/` | the two papers, figures regenerated from versioned CSVs only |
| `docs/` | methodology, reproducibility, literature review, the PR ledger |

## Reproducing the numbers

Claims map to commands in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md); the artifact
check-list is [`docs/ARTIFACT_APPENDIX.md`](docs/ARTIFACT_APPENDIX.md). Paper figures rebuild
from committed CSVs with no GPU. Reproducing the *measurements* needs one — on a machine
without, `scripts/modal_gpu.py` rents one.

## Contributing

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). Project context and decisions are in
[`CLAUDE.md`](CLAUDE.md); the session history is in [`docs/HISTORY.md`](docs/HISTORY.md).

## License

MIT — see [LICENSE](LICENSE).
