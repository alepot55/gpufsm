"""Result and benchmark-statistics dataclasses (one result format for all backends)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Result:
    """Outcome of a single FSM run on one backend/technique."""

    accepted: bool
    match_len: int
    kernel_ms: float = 0.0
    total_ms: float = 0.0
    transfer_ms: float = 0.0
    mem_bytes: int = 0

    def matches(self, other: Result) -> bool:
        """Correctness equality: only the FSM verdict, not the timings."""
        return self.accepted == other.accepted and self.match_len == other.match_len


@dataclass
class BenchmarkStats:
    """Aggregated timings over ``n`` repetitions (after warmup)."""

    backend: str
    technique: str
    accepted: bool
    match_len: int
    n: int
    raw_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.raw_ms) / len(self.raw_ms) if self.raw_ms else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.raw_ms) < 2:
            return 0.0
        m = self.mean_ms
        var = sum((x - m) ** 2 for x in self.raw_ms) / (len(self.raw_ms) - 1)
        return math.sqrt(var)

    @property
    def ci95_ms(self) -> float:
        """Half-width of the 95% confidence interval of the mean (normal approx)."""
        if len(self.raw_ms) < 2:
            return 0.0
        return 1.96 * self.std_ms / math.sqrt(len(self.raw_ms))

    def as_row(self) -> dict[str, float | str | int | bool]:
        """Flat dict suitable for CSV writing."""
        return {
            "backend": self.backend,
            "technique": self.technique,
            "accepted": self.accepted,
            "match_len": self.match_len,
            "n": self.n,
            "mean_ms": round(self.mean_ms, 6),
            "std_ms": round(self.std_ms, 6),
            "ci95_ms": round(self.ci95_ms, 6),
        }
