"""Minimal single-launch profiling target for Nsight Compute (ncu).

Usage:  python scripts/profile_target.py <backend> <technique> <num_states>
Builds one NFA + a small batch and issues exactly ONE run_batch (one kernel launch)
so ncu can profile that kernel cleanly. CUDA kernels have stable names
(worklist_multistream_kernel, bitpacked_multistream_kernel), so filter ncu with
--kernel-name-base to isolate them.
"""

from __future__ import annotations

import random
import sys

import numpy as np

from gpufsm.api import run_batch
from gpufsm.nfa import NFABuilder


def random_nfa(n: int, seed: int = 1):
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def main() -> None:
    backend, technique, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_strings = int(sys.argv[4]) if len(sys.argv) > 4 else 16384  # default saturates ~46 SMs
    slen = 256
    nfa = random_nfa(n)
    rng = np.random.default_rng(0)
    flat = rng.integers(ord("a"), ord("a") + 5, size=n_strings * slen, dtype=np.uint8).tobytes()
    batch = [flat[i * slen : (i + 1) * slen] for i in range(n_strings)]
    run_batch(nfa, batch, backend=backend, technique=technique)  # the single profiled launch


if __name__ == "__main__":
    main()
