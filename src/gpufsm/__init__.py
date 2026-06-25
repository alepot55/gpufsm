"""gpufsm — portable GPU finite-state-machine (NFA) processing: Triton vs CUDA.

Public API::

    from gpufsm import NFABuilder, Backend, run, benchmark

    b = NFABuilder()
    s0 = b.add_state(); s1 = b.add_state(accept=True)
    b.set_start(s0); b.add_transition(s0, "a", s1)
    nfa = b.build()

    run(nfa, b"a", backend=Backend.CPU)            # -> Result(accepted=True, match_len=1)
    benchmark(nfa, b"a" * 1000, repeats=10)        # -> BenchmarkStats
"""

from __future__ import annotations

from .api import benchmark, run, run_batch
from .nfa import ANY_SYMBOL, NFA, NFABuilder
from .reference import simulate
from .registry import Backend, available_backends, is_available, list_techniques
from .result import BenchmarkStats, Result

__version__ = "0.1.0"

__all__ = [
    "ANY_SYMBOL",
    "NFA",
    "NFABuilder",
    "Backend",
    "Result",
    "BenchmarkStats",
    "run",
    "run_batch",
    "benchmark",
    "simulate",
    "available_backends",
    "is_available",
    "list_techniques",
]
