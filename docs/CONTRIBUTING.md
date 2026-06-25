# Contributing

## Dev setup

```bash
python -m pip install -e ".[dev]"
pytest -m "not gpu" -q && ruff check src/gpufsm tests && mypy src/gpufsm
```

CI runs ruff (lint + format), mypy, and the CPU test suite on every push. Keep them green.

## Adding a backend or technique

The registry is the only extension point. One file, one decorator:

```python
# src/gpufsm/backends/my_backend.py
from ..nfa import NFA
from ..registry import Backend, register, register_availability
from ..result import Result

class MyExecutor:
    def __init__(self, nfa: NFA, technique: str): ...
    def run(self, input_bytes: bytes) -> Result: ...

@register(Backend.TRITON, "my_technique")
def _make(nfa, technique): return MyExecutor(nfa, technique)

register_availability(Backend.TRITON, lambda: _probe_gpu_and_deps())
```

Then add it to the import list in `backends/__init__.py` (guarded, so a missing dependency is a no-op).

## Correctness is non-negotiable

Every backend/technique must reproduce `gpufsm.reference.simulate` (`accepted`, `match_len`) on the
example NFAs and the benchmark suite. Add a `@pytest.mark.gpu` test asserting agreement with the oracle.

## Style

- src-layout, type hints, `from __future__ import annotations`.
- Small, focused modules; one way to do each thing.
- No dead code, no committed build artifacts.
