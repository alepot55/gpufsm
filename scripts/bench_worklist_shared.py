"""Shared-memory vs global-memory working-set for the warp-cooperative worklist.

worklist_shared stages the per-string working set (cur/nxt/frontier/newb) in dynamic shared
memory instead of global; worklist_warp keeps it in global. This isolates the effect of
working-set *residency/layout* on the work-efficient kernel. Capped at ~1536 states (the
working set must fit 48 KB). Correctness: worklist_shared == worklist_warp (both validated vs
the CPU oracle in tests). Writes paper/data/worklist_shared_rtx4070.csv.

Usage:  python scripts/bench_worklist_shared.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm import run_batch
from gpufsm.bench import SPARSE_WORKLIST, random_nfa


def _sparse_nfa(n: int, seed: int):
    """The sparse-worklist family plus its alphabet, which the callers need for inputs."""
    return random_nfa(n, seed, SPARSE_WORKLIST), [ord(c) for c in SPARSE_WORKLIST.alphabet]


SIZES = [256, 512, 1024, 1536]  # all fit 48 KB shared (4*words*8 bytes)
N_STRINGS = 256
STR_LEN = 256
WARMUP = 3
RUNS = 7
GPU = "RTX4070"


def _median_ms(nfa, batch, tech):
    for _ in range(WARMUP):
        run_batch(nfa, batch, backend="cuda", technique=tech)
    return statistics.median(
        run_batch(nfa, batch, backend="cuda", technique=tech)[0].kernel_ms for _ in range(RUNS)
    )


def main() -> int:
    rng = random.Random(0)
    rows = []
    for n in SIZES:
        nfa, alpha = _sparse_nfa(n, seed=n)
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
        ww = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
        ss = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_shared")]
        if ww != ss:
            print(f"n={n}: SHARED != WARP — correctness bug, aborting")
            return 1
        tw = _median_ms(nfa, batch, "worklist_warp")
        ts = _median_ms(nfa, batch, "worklist_shared")
        words = (n + 63) // 64
        bits = N_STRINGS * STR_LEN * 8
        w_gbps, s_gbps = bits / (tw * 1e-3) / 1e9, bits / (ts * 1e-3) / 1e9
        rows.append((n, words, round(w_gbps, 3), round(s_gbps, 3), round(s_gbps / w_gbps, 2)))
        print(
            f"n={n:5d} words={words:3d}: warp={w_gbps:6.2f} shared={s_gbps:6.2f} "
            f"shared/warp={s_gbps / w_gbps:.2f}x"
        )

    out = Path("paper/data/worklist_shared_rtx4070.csv")
    with out.open("w") as f:
        f.write("num_states,words,warp_gbps,shared_gbps,shared_over_warp,gpu\n")
        for n, w, wg, sg, r in rows:
            f.write(f"{n},{w},{wg},{sg},{r},{GPU}\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
