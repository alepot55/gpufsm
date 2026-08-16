"""M0 — reproduce the Triton-vs-CUDA worklist regret anchor on THIS machine.

Paper 2 ("cure") milestone 0. Every later milestone must MOVE this number, so we pin it
honestly here: same algorithm (work-efficient active-set worklist, 1 thread/program per
string), same data, same harness — only the DSL differs. Both kernels are validated
bit-for-bit against the reference.py oracle BEFORE any throughput is reported (correctness
gates speed). Writes paper2/data/m0_anchor_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m0_anchor.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from gpufsm.api import run_batch
from gpufsm.bench import random_batch, random_nfa
from gpufsm.bench.oracle import matches as oracle_match
from gpufsm.bench.timing import repeat, summarize
from gpufsm.core.registry import Backend, available_backends, list_techniques

SLEN = 256
N_STRINGS = 4096  # GPU-saturating batch (the honest regime; small batch inflates the ratio)
ALPHABET = "abcde"
WARMUP = 3
SAMPLES = 9


def _measure(nfa, batch, total_bytes, backend, technique) -> tuple[float, float]:
    """Median throughput (Gbps) and median kernel time (ms) for one (backend, technique)."""
    samples = repeat(
        lambda: run_batch(nfa, batch, backend=backend, technique=technique)[0].kernel_ms,
        warmup=WARMUP,
        samples=SAMPLES,
    )
    stats = summarize(samples, total_bytes)
    return stats["gbps"], stats["median_ms"]


def main() -> int:
    bes = available_backends()
    if Backend.TRITON not in bes or Backend.CUDA not in bes:
        print(f"SKIP: need TRITON+CUDA backends, have {[b.value for b in bes]}")
        return 0

    tri_techs = list_techniques(Backend.TRITON)
    cuda_techs = list_techniques(Backend.CUDA)
    tri = "worklist" if "worklist" in tri_techs else None
    # CUDA work-efficient 1-thread/string counterpart to triton/worklist. triton/worklist is
    # register-resident (scalar int64 working set), so the FAIR apples-to-apples counterpart is
    # cuda/worklist (also register-resident), NOT worklist_global (slower, global working set).
    cuda = next((t for t in ("worklist", "worklist_global") if t in cuda_techs), None)
    if tri is None or cuda is None:
        print(f"SKIP: triton techs={tri_techs} cuda techs={cuda_techs}")
        return 1
    print(f"anchor: triton/{tri}  vs  cuda/{cuda}  (work-efficient, 1 thread/string)\n")

    rows = []
    print(
        f"{'states':>7}{'seed':>5}{'triton_Gbps':>13}{'cuda_Gbps':>11}{'regret(x)':>11}{'oracle':>8}"
    )
    for n in (16, 32, 48, 64):
        for seed in (0, 1, 2):
            nfa = random_nfa(n, seed=1000 + n + seed)
            batch, total = random_batch(N_STRINGS, SLEN, seed, ALPHABET)
            ok_t = oracle_match(nfa, batch, Backend.TRITON, tri)
            ok_c = oracle_match(nfa, batch, Backend.CUDA, cuda)
            if not (ok_t and ok_c):
                print(f"{n:7d}{seed:5d}  ORACLE MISMATCH triton={ok_t} cuda={ok_c} — skipping")
                continue
            gt, mt = _measure(nfa, batch, total, Backend.TRITON, tri)
            gc, mc = _measure(nfa, batch, total, Backend.CUDA, cuda)
            regret = gc / gt if gt > 0 else float("nan")
            print(f"{n:7d}{seed:5d}{gt:13.2f}{gc:11.2f}{regret:11.2f}{'  ok':>8}")
            rows.append((n, seed, round(gt, 3), round(gc, 3), round(regret, 3)))

    if rows:
        regrets = [r[4] for r in rows]
        print(
            f"\nregret over {len(rows)} configs: "
            f"median {statistics.median(regrets):.2f}x  "
            f"min {min(regrets):.2f}x  max {max(regrets):.2f}x"
        )
        outp = Path("paper2/data/m0_anchor_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("states,seed,triton_gbps,cuda_gbps,regret,gpu\n")
            for n, seed, gt, gc, rg in rows:
                f.write(f"{n},{seed},{gt},{gc},{rg},RTX4070\n")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
