"""Block-parallel (warp-per-string) vs 1-thread/string worklist on large automata.

The register/thread worklist under-utilizes the GPU on big automata: one string's many
state-words are processed serially by a single thread. The warp kernel spreads those words
across 32 lanes. This sweep shows the speedup grows with automaton size (more words -> more
lanes busy). Correctness is checked GPU-side (warp == global; both are validated bit-for-bit
against the CPU oracle in the test-suite). Writes paper/data/worklist_warp_rtx4070.csv.

Usage:  python scripts/bench_worklist_warp.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm import NFABuilder, run_batch

SIZES = [512, 2048, 8192, 32768]
N_STRINGS = 256
STR_LEN = 256
WARMUP = 3
RUNS = 7
GPU = "RTX4070"


def _random_nfa(n: int, seed: int):
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.02)
    b.set_start(0)
    alpha = [ord(c) for c in "abcde"]
    for s in range(n):
        for _ in range(2):  # 2 symbol transitions/state -> non-trivial active set
            b.add_transition(s, rng.choice(alpha), rng.randrange(n))
        if rng.random() < 0.3:
            b.add_epsilon(s, rng.randrange(n))
    return b.build(), alpha


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
        nfa, alpha = _random_nfa(n, seed=n)
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
        gg = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_global")]
        gw = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
        if gg != gw:
            print(f"n={n}: WARP != GLOBAL — correctness bug, aborting")
            return 1
        tg = _median_ms(nfa, batch, "worklist_global")
        tw = _median_ms(nfa, batch, "worklist_warp")
        words = (n + 63) // 64
        bits = N_STRINGS * STR_LEN * 8
        g_gbps = bits / (tg * 1e-3) / 1e9
        w_gbps = bits / (tw * 1e-3) / 1e9
        rows.append((n, words, round(g_gbps, 3), round(w_gbps, 3), round(w_gbps / g_gbps, 1)))
        print(
            f"n={n:6d} words={words:4d}: global={g_gbps:7.3f} warp={w_gbps:7.3f} "
            f"speedup={w_gbps / g_gbps:5.1f}x"
        )

    # Real ANMLZoo automata (representative active-set density). GPU-side equivalence
    # (warp == global; both validated vs the CPU oracle in tests). Network-gated.
    real_rows = []
    try:
        from gpufsm.io.anml import load_anml
        from gpufsm.io.datasets import DATASETS, ensure

        for key in ["levenshtein", "fermi", "brill"]:
            nfa = load_anml(ensure(DATASETS[key], "data/anmlzoo"))
            alpha = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
            batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
            gg = [
                (r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_global")
            ]
            gw = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
            if gg != gw:
                print(f"{key}: WARP != GLOBAL — aborting")
                return 1
            tg = _median_ms(nfa, batch, "worklist_global")
            tw = _median_ms(nfa, batch, "worklist_warp")
            words = (nfa.num_states + 63) // 64
            bits = N_STRINGS * STR_LEN * 8
            g_gbps, w_gbps = bits / (tg * 1e-3) / 1e9, bits / (tw * 1e-3) / 1e9
            real_rows.append(
                (
                    key,
                    nfa.num_states,
                    words,
                    round(g_gbps, 3),
                    round(w_gbps, 3),
                    round(w_gbps / g_gbps, 1),
                )
            )
            print(
                f"{key:11s} n={nfa.num_states:6d} words={words:4d}: global={g_gbps:7.3f} "
                f"warp={w_gbps:7.3f} speedup={w_gbps / g_gbps:5.1f}x"
            )
    except Exception as e:  # offline / no dataset -> synthetic-only CSV
        print(f"(real-automata pass skipped: {type(e).__name__}: {e})")

    out = Path("paper/data/worklist_warp_rtx4070.csv")
    with out.open("w") as f:
        f.write("automaton,num_states,words,global_gbps,warp_gbps,speedup,gpu\n")
        for n, w, g, wp, sp in rows:
            f.write(f"synthetic,{n},{w},{g},{wp},{sp},{GPU}\n")
        for key, n, w, g, wp, sp in real_rows:
            f.write(f"{key},{n},{w},{g},{wp},{sp},{GPU}\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
