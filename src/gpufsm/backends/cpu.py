"""CPU reference backend — always available, wraps the correctness oracle."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..bitmap import simulate_bitmap
from ..nfa import NFA
from ..reference import simulate
from ..registry import Backend, register, register_availability
from ..result import Result

# technique -> (nfa, input) -> (accepted, match_len)
_SIMULATORS: dict[str, Callable[[NFA, bytes], tuple[bool, int]]] = {
    "reference": simulate,  # plain set-based oracle
    "bitmap": simulate_bitmap,  # bit-packed; executable spec for the GPU kernels
}


class CPUExecutor:
    def __init__(self, nfa: NFA, technique: str = "reference") -> None:
        self.nfa = nfa
        self.technique = technique
        self._sim = _SIMULATORS[technique]

    def run(self, input_bytes: bytes) -> Result:
        t0 = time.perf_counter()
        accepted, match_len = self._sim(self.nfa, input_bytes)
        dt = (time.perf_counter() - t0) * 1000.0
        return Result(accepted=accepted, match_len=match_len, kernel_ms=dt, total_ms=dt)


for _tech in _SIMULATORS:
    register(Backend.CPU, _tech)(lambda nfa, technique: CPUExecutor(nfa, technique))


register_availability(Backend.CPU, lambda: True)
