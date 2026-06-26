"""Block-parallel (warp-per-string) vs 1-thread/string worklist — with batch sensitivity.

The single-thread `worklist_global` processes one string's many state-words serially; the
`worklist_warp` kernel spreads them across 32 lanes. The speedup is **batch-dependent**: at
small batch `worklist_global` cannot fill the GPU (few strings = few threads), so warp wins
hugely; at a GPU-saturating batch `worklist_global` has abundant string-level parallelism and
the fair, conservative speedup is smaller (but warp still wins — each global thread still does
~32x more serial per-word work). We therefore report BOTH: a batch-sensitivity sweep on one
automaton, and per-automaton speedups at a saturating batch (the honest headline).

Correctness is checked GPU-side (warp == global; both validated bit-for-bit vs the CPU oracle
in the test-suite). Writes paper/data/worklist_warp_rtx4070.csv (saturating-batch, per automaton)
and paper/data/worklist_warp_batch_rtx4070.csv (batch sensitivity).

Usage:  python scripts/bench_worklist_warp.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm import NFABuilder, run_batch

SYNTH_SIZES = [512, 2048, 8192]
SAT_STRINGS = 4096  # GPU-saturating batch (>= ~46 SMs * warps): the fair comparison
BATCH_GRID = [64, 256, 1024, 4096, 16384]  # batch-sensitivity sweep
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
        for _ in range(2):
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


def _speedup(nfa, batch):
    """(global_gbps, warp_gbps, speedup) with a GPU-side warp==global correctness gate."""
    gg = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_global")]
    gw = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
    if gg != gw:
        raise RuntimeError("WARP != GLOBAL — correctness bug")
    tg = _median_ms(nfa, batch, "worklist_global")
    tw = _median_ms(nfa, batch, "worklist_warp")
    bits = len(batch) * STR_LEN * 8
    return bits / (tg * 1e-3) / 1e9, bits / (tw * 1e-3) / 1e9, None


def main() -> int:
    rng = random.Random(0)

    # (1) batch sensitivity on one fixed automaton — documents the confound.
    batch_rows = []
    nfa, alpha = _random_nfa(8192, seed=8192)
    print("batch-sensitivity (8192-state synthetic, 128 words):")
    for nstr in BATCH_GRID:
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(nstr)]
        g, w, _ = _speedup(nfa, batch)
        batch_rows.append((8192, nstr, round(g, 4), round(w, 4), round(w / g, 1)))
        print(f"  n_strings={nstr:6d}: global={g:8.4f} warp={w:8.4f} speedup={w / g:6.1f}x")

    # (2) per-automaton at a SATURATING batch — the honest, conservative headline.
    rows = []
    print(f"\nsaturating batch ({SAT_STRINGS} strings) — fair speedup:")
    for n in SYNTH_SIZES:
        nfa, alpha = _random_nfa(n, seed=n)
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(SAT_STRINGS)]
        g, w, _ = _speedup(nfa, batch)
        rows.append(("synthetic", n, (n + 63) // 64, round(g, 4), round(w, 4), round(w / g, 1)))
        print(f"  synthetic n={n:6d}: global={g:8.4f} warp={w:8.4f} speedup={w / g:6.1f}x")

    try:
        from gpufsm.io.anml import load_anml
        from gpufsm.io.datasets import DATASETS, ensure

        for key in ["levenshtein", "fermi", "brill"]:
            nfa = load_anml(ensure(DATASETS[key], "data/anmlzoo"))
            alpha = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
            batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(SAT_STRINGS)]
            g, w, _ = _speedup(nfa, batch)
            rows.append(
                (
                    key,
                    nfa.num_states,
                    (nfa.num_states + 63) // 64,
                    round(g, 4),
                    round(w, 4),
                    round(w / g, 1),
                )
            )
            print(
                f"  {key:11s} n={nfa.num_states:6d}: global={g:8.4f} warp={w:8.4f} "
                f"speedup={w / g:6.1f}x"
            )
    except Exception as e:
        print(f"(real-automata pass skipped: {type(e).__name__}: {e})")

    out = Path("paper/data/worklist_warp_rtx4070.csv")
    with out.open("w") as f:
        f.write("automaton,num_states,words,n_strings,global_gbps,warp_gbps,speedup,gpu\n")
        for a, n, w, g, wp, sp in rows:
            f.write(f"{a},{n},{w},{SAT_STRINGS},{g},{wp},{sp},{GPU}\n")
    outb = Path("paper/data/worklist_warp_batch_rtx4070.csv")
    with outb.open("w") as f:
        f.write("num_states,n_strings,global_gbps,warp_gbps,speedup,gpu\n")
        for n, ns, g, wp, sp in batch_rows:
            f.write(f"{n},{ns},{g},{wp},{sp},{GPU}\n")
    print(f"\nwrote {out} and {outb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
