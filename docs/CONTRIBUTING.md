# Contributing

## Dev setup

```bash
python -m pip install -e ".[dev]"
```

Before every commit, run exactly what CI runs — the format check is part of it, and skipping
it is the single most common way to get a red build:

```bash
ruff format --check src tests scripts experiments paper/figures.py paper2/figures.py && \
ruff check src tests scripts experiments paper/figures.py paper2/figures.py && \
mypy src/gpufsm && pytest -m "not gpu" -q
```

CI also compiles each CUDA translation unit with `nvcc` (no GPU needed). The GPU suite
(`pytest -m gpu`) needs real hardware; on a machine without one, `scripts/modal_gpu.py`
rents one:

```bash
python scripts/modal_gpu.py --gpu A100 --apt g++ \
    --cmd 'pip install -e "." --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON' \
    --cmd 'pytest -m gpu -q'
```

## Adding a backend or technique

The registry is the only extension point: one module, one decorator.

```python
# src/gpufsm/backends/mydsl/worklist.py
from ...core.nfa import NFA
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result

class MyExecutor:
    def __init__(self, nfa: NFA, technique: str) -> None: ...
    def run(self, input_bytes: bytes) -> Result: ...
    # optional: run_batch(list[bytes]) -> list[Result] for a single-launch batch

@register(Kind.NFA, Backend.MYDSL, "worklist")
def _make(automaton: Automaton, technique: str) -> MyExecutor:
    assert isinstance(automaton, NFA)
    return MyExecutor(automaton, technique)
```

Then list the module in the backend package's `__init__.py`, behind its availability probe:

```python
# src/gpufsm/backends/mydsl/__init__.py
def mydsl_available() -> bool: ...

if mydsl_available():
    importlib.import_module("gpufsm.backends.mydsl.worklist")

register_availability(Backend.MYDSL, mydsl_available)   # unconditionally
```

Register the probe **outside** the guard, so `gpufsm env` can report the backend as
*unavailable* instead of pretending it does not exist.

### Where code goes

- Anything a backend needs that is not a kernel — packing inputs, packing accept words,
  building batched results — belongs in `gpufsm.core`, where CPU tests can reach it. Three
  copies of `_pack` is how the accept-word dtype came to differ between backends.
- Anything a *measurement* needs — random automata, timing statistics, the nvcc helper, CSV
  writing — belongs in `gpufsm.bench`. Do not add a local `random_nfa`: there were eleven,
  and they had drifted.

## Correctness is non-negotiable

Every backend and technique must reproduce `gpufsm.reference.simulate` (NFA) or
`simulate_dfa` (DFA) exactly. Add a `@pytest.mark.gpu` test asserting agreement with the
oracle, and gate any measurement on `gpufsm.bench.oracle.require` before it reports a number.

`tests/test_golden.py` replays pinned oracle verdicts over a serialized corpus. If it fails,
the **semantics** changed — do not regenerate the golden file to make it pass unless you
understand and intend the change.

`tests/test_generators.py` pins the random-automaton families against transcriptions of the
original generators, RNG call order included. If it fails, the committed CSVs no longer
describe the automata the code builds.

## CUDA sources

`src/gpufsm/backends/cuda/native/` is one translation unit per kernel family, compiled
**without** relocatable device code. A `__device__` helper is therefore private to its `.cu`:
if two families need it, it goes in `native/include/`. Host entry points are declared in
`native/include/api.hpp`, transcribed from the definitions — spelling included, since
`uint64_t` and `unsigned long long` are different types on LP64 Linux and the mismatch only
shows up as an undefined symbol at link time.

## Style

- src-layout, type hints, `from __future__ import annotations`.
- Small, focused modules; one way to do each thing.
- No dead code, no committed build artifacts.
- Comments explain *why*, especially where the obvious thing is wrong: the `int(0)` casts in
  the Warp kernels, the `-real` CUDA architectures in `CMakeLists.txt`, and the RNG call
  order in `bench/generators.py` all look removable and are not.
