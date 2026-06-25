"""Backend/technique registry — the single extension point.

Adding a backend or a technique is one file plus one ``@register`` line. Each
registered factory builds an :class:`Executor` (anything with ``run(bytes) ->
Result``) for a given NFA. Backends declare an availability probe so the rest of
the system degrades gracefully when CUDA/Triton are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol, runtime_checkable

from .nfa import NFA
from .result import Result


class Backend(str, Enum):
    CPU = "cpu"  # reference simulator — always available, the correctness oracle
    TRITON = "triton"
    CUDA = "cuda"


@runtime_checkable
class Executor(Protocol):
    def run(self, input_bytes: bytes) -> Result: ...


ExecutorFactory = Callable[[NFA, str], Executor]

_REGISTRY: dict[tuple[Backend, str], ExecutorFactory] = {}
_AVAILABILITY: dict[Backend, Callable[[], bool]] = {}


def register(backend: Backend, technique: str) -> Callable[[ExecutorFactory], ExecutorFactory]:
    """Decorator: register a factory ``(nfa, technique) -> Executor``."""

    def deco(factory: ExecutorFactory) -> ExecutorFactory:
        _REGISTRY[(backend, technique)] = factory
        return factory

    return deco


def register_availability(backend: Backend, probe: Callable[[], bool]) -> None:
    _AVAILABILITY[backend] = probe


def is_available(backend: Backend) -> bool:
    probe = _AVAILABILITY.get(backend)
    try:
        return bool(probe()) if probe else any(b == backend for b, _ in _REGISTRY)
    except Exception:
        return False


def get_factory(backend: Backend, technique: str | None) -> tuple[str, ExecutorFactory]:
    """Resolve a factory; if ``technique`` is None, use the backend's default."""
    techs = list_techniques(backend)
    if not techs:
        raise KeyError(f"no techniques registered for backend {backend.value!r}")
    if technique is None:
        technique = techs[0]
    if (backend, technique) not in _REGISTRY:
        raise KeyError(
            f"technique {technique!r} not registered for backend {backend.value!r}; "
            f"available: {techs}"
        )
    return technique, _REGISTRY[(backend, technique)]


def list_techniques(backend: Backend) -> list[str]:
    return [t for (b, t) in _REGISTRY if b == backend]


def available_backends() -> list[Backend]:
    return [b for b in Backend if list_techniques(b) and is_available(b)]
