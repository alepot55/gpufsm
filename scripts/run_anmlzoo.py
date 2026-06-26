"""Run a real ANMLZoo automaton on the GPU and validate GPU == the reference oracle.

Fetches a checksum-pinned ANMLZoo automaton (gpufsm.io.datasets), loads it via the
ANML parser, runs it on the scalable `worklist_global` CUDA kernel, and checks every
verdict against the CPU reference. Also reports throughput. This is the real-suite
credibility check (vs. random NFAs).

Run on a GPU box:  python scripts/run_anmlzoo.py [dataset-key]   (default: levenshtein)
"""

from __future__ import annotations

import random
import sys
import time

from gpufsm.api import run_batch
from gpufsm.io.anml import load_anml
from gpufsm.io.datasets import DATASETS, ensure
from gpufsm.reference import simulate


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "levenshtein"
    ds = DATASETS[key]
    path = ensure(ds, "data/anmlzoo")  # download + SHA-256 verify (cached)
    t0 = time.perf_counter()
    nfa = load_anml(path)
    print(
        f"{key}: states={nfa.num_states} sym_trans={nfa.num_sym_transitions} "
        f"eps={nfa.num_eps_transitions} accept={int(nfa.accept.sum())} "
        f"load={time.perf_counter() - t0:.2f}s"
    )

    alphabet = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
    rng = random.Random(0)
    batch = [
        bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 40))) for _ in range(64)
    ]
    refs = [simulate(nfa, d) for d in batch]
    res = run_batch(nfa, batch, backend="cuda", technique="worklist_global")
    got = [(r.accepted, r.match_len) for r in res]
    mismatches = sum(1 for a, b in zip(got, refs) if a != b)
    total_bytes = sum(len(d) for d in batch)
    kernel_ms = res[0].kernel_ms if res else 0.0
    gbps = (total_bytes * 8.0) / (kernel_ms * 1e-3) / 1e9 if kernel_ms > 0 else 0.0
    print(
        f"GPU(worklist_global) vs reference: mismatches={mismatches}/{len(batch)}; "
        f"accepted={sum(1 for a, _ in refs if a)}; throughput={gbps:.3f} Gbps"
    )
    if mismatches:
        raise SystemExit(f"FAIL: {mismatches} GPU/reference mismatches on real automaton {key!r}")
    print(f"OK: GPU matches the reference oracle on real ANMLZoo automaton {key!r}.")


if __name__ == "__main__":
    main()
