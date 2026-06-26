"""Finer DFA throughput sweep across table sizes — exposes the L2 memory-bound knee.

The DFA dense-table walk is the memory-bound face of the two-faces thesis: one random
``trans[s*256 + byte]`` lookup per input byte. As ``num_states`` grows the table
(``num_states`` KB, since 256x int32 = 1 KB/state) crosses the GPU's L2 and throughput
should drop for the thread-model backends (CUDA, Warp) while the tile/SPMD Triton kernel
stays flat (scalar-gather-bound, never reaching the memory regime).

Sweeps a fine grid of table sizes for cuda/warp/triton, validates each against the CPU
oracle first, then reports median-of-N batch-kernel throughput. Writes the canonical
``paper/data/dfa_regret_rtx4070.csv`` consumed by ``paper/figures.py``.

Usage:  python scripts/sweep_dfa.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import numpy as np

from gpufsm.dfa import random_dfa, simulate_dfa
from gpufsm.dfa_api import run_dfa_batch

# 6 MB L2 on the RTX 4070; 1 KB/state means the table crosses L2 around ~6000 states.
# Grid straddles the knee: below, around, and far above L2.
STATE_GRID = [1024, 2048, 4096, 6144, 8192, 16384, 32768, 50000, 100000]
N_STRINGS = 4096
STR_LEN = 1024  # 4 MB of input per batch -> stable timing, table-walk dominates
WARMUP = 3
RUNS = 9
GPU = "RTX4070"
BACKENDS = ("cuda", "warp", "triton")


def _throughput_gbps(total_bytes: int, kernel_ms: float) -> float:
    if kernel_ms <= 0:
        return float("nan")
    return (total_bytes * 8.0) / (kernel_ms * 1e-3) / 1e9


def _validate(dfa, rng: random.Random) -> bool:
    """Cheap oracle check on a small batch (separate from the timing batch)."""
    batch = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 48))) for _ in range(32)]
    refs = [simulate_dfa(dfa, b) for b in batch]
    for be in BACKENDS:
        got = [(r.accepted, r.match_len) for r in run_dfa_batch(dfa, batch, backend=be)]
        if got != refs:
            print(f"  VALIDATION FAIL backend={be}: GPU != oracle")
            return False
    return True


def _measure(dfa, batch: list[bytes], backend: str) -> float:
    total = sum(len(b) for b in batch)
    for _ in range(WARMUP):
        run_dfa_batch(dfa, batch, backend=backend)
    samples = []
    for _ in range(RUNS):
        res = run_dfa_batch(dfa, batch, backend=backend)
        samples.append(_throughput_gbps(total, res[0].kernel_ms))
    return statistics.median(samples)


SEEDS = (0, 1, 2)  # median over 3 random DFAs/size: the knee must be seed-robust, not noise


def main() -> int:
    rng = random.Random(0)
    # one fixed timing batch (random bytes) reused across sizes for comparability
    timing_batch = [bytes(rng.randint(0, 255) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
    rows: list[tuple[str, int, int, float, str, str]] = []

    for n in STATE_GRID:
        # validate one seed; then measure each backend's throughput as the median over SEEDS
        # (different random DFAs) — so a reported knee reflects the table size, not one DFA.
        if not _validate(random_dfa(n, accept_prob=0.02, seed=n), random.Random(n)):
            print(f"n={n}: validation failed, skipping")
            continue
        table_kb = n  # 256 * int32 = 1 KB/state
        cache = "fits L2" if table_kb <= 6144 else "exceeds L2"
        line = f"n={n:6d} ({table_kb / 1024:.1f} MB, {cache}): "
        for be in BACKENDS:
            per_seed = [
                _measure(random_dfa(n, accept_prob=0.02, seed=s * 1000 + n), timing_batch, be)
                for s in SEEDS
            ]
            tp = statistics.median(per_seed)
            rows.append((be, n, table_kb, round(tp, 1), GPU, cache))
            line += f"{be}={tp:6.1f}  "
        print(line)

    out = Path("paper/data/dfa_regret_rtx4070.csv")
    with out.open("w") as f:
        f.write("backend,num_states,table_kb,throughput_gbps,gpu,note\n")
        for be, n, kb, tp, gpu, note in rows:
            f.write(f"{be},{n},{kb},{tp},{gpu},{note}\n")
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
