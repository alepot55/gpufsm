"""The one public API: ``run`` (single execution) and ``benchmark`` (timed)."""

from __future__ import annotations

from . import backends as _backends  # noqa: F401  (triggers backend registration)
from .nfa import NFA
from .registry import Backend, get_factory
from .result import BenchmarkStats, Result


def run(
    nfa: NFA,
    input_bytes: bytes,
    backend: Backend | str = Backend.CPU,
    technique: str | None = None,
) -> Result:
    """Run ``nfa`` over ``input_bytes`` on the chosen backend/technique."""
    backend = Backend(backend)
    technique, factory = get_factory(backend, technique)
    return factory(nfa, technique).run(input_bytes)


def benchmark(
    nfa: NFA,
    input_bytes: bytes,
    backend: Backend | str = Backend.CPU,
    technique: str | None = None,
    repeats: int = 10,
    warmup: int = 3,
) -> BenchmarkStats:
    """Time ``repeats`` runs (after ``warmup``) and aggregate mean/std/CI95."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    backend = Backend(backend)
    technique, factory = get_factory(backend, technique)
    executor = factory(nfa, technique)

    last: Result | None = None
    for _ in range(max(0, warmup)):
        last = executor.run(input_bytes)

    raw: list[float] = []
    for _ in range(repeats):
        last = executor.run(input_bytes)
        raw.append(last.kernel_ms)

    assert last is not None
    return BenchmarkStats(
        backend=backend.value,
        technique=technique,
        accepted=last.accepted,
        match_len=last.match_len,
        n=repeats,
        raw_ms=raw,
    )
