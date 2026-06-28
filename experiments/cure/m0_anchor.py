"""M0 — reproduce the Triton-vs-CUDA worklist regret anchor on THIS machine.

Paper 2 ("cure") milestone 0. Every later milestone must MOVE this number, so we pin it
honestly here: same algorithm (work-efficient active-set worklist, 1 thread/program per
string), same data, same harness — only the DSL differs. Both kernels are validated
bit-for-bit against the reference.py oracle BEFORE any throughput is reported (correctness
gates speed). Writes paper2/data/m0_anchor_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m0_anchor.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import numpy as np

from gpufsm.api import run, run_batch
from gpufsm.nfa import NFABuilder
from gpufsm.registry import Backend, available_backends, list_techniques

SLEN = 256
N_STRINGS = 4096  # GPU-saturating batch (the honest regime; small batch inflates the ratio)
ALPHABET = "abcde"
WARMUP = 3
SAMPLES = 9


def random_nfa(n: int, seed: int):
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice(ALPHABET)), rng.randrange(n))
    return b.build()


def make_batch(seed: int) -> tuple[list[bytes], int]:
    rng = np.random.default_rng(seed)
    flat = rng.integers(ord("a"), ord("a") + len(ALPHABET), size=N_STRINGS * SLEN, dtype=np.uint8)
    buf = flat.tobytes()
    return [buf[i * SLEN : (i + 1) * SLEN] for i in range(N_STRINGS)], N_STRINGS * SLEN


def oracle_match(nfa, batch, backend, technique, n_check: int = 64) -> bool:
    """Bit-for-bit check vs the CPU reference oracle on a sample of the batch."""
    got = run_batch(nfa, batch[:n_check], backend=backend, technique=technique)
    for s, r in zip(batch[:n_check], got, strict=True):
        ref = run(nfa, s, backend=Backend.CPU, technique="reference")
        if (r.accepted, r.match_len) != (ref.accepted, ref.match_len):
            return False
    return True


def measure_gbps(nfa, batch, total_bytes, backend, technique) -> tuple[float, float, float]:
    for _ in range(WARMUP):
        run_batch(nfa, batch, backend=backend, technique=technique)
    samples = []
    for _ in range(SAMPLES):
        km = run_batch(nfa, batch, backend=backend, technique=technique)[0].kernel_ms
        if km > 0:
            samples.append(km)
    med = statistics.median(samples)
    gbps = total_bytes * 8.0 / (med * 1e-3) / 1e9
    return gbps, med, statistics.pstdev(samples) if len(samples) > 1 else 0.0


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
            batch, total = make_batch(seed)
            ok_t = oracle_match(nfa, batch, Backend.TRITON, tri)
            ok_c = oracle_match(nfa, batch, Backend.CUDA, cuda)
            if not (ok_t and ok_c):
                print(f"{n:7d}{seed:5d}  ORACLE MISMATCH triton={ok_t} cuda={ok_c} — skipping")
                continue
            gt, mt, _ = measure_gbps(nfa, batch, total, Backend.TRITON, tri)
            gc, mc, _ = measure_gbps(nfa, batch, total, Backend.CUDA, cuda)
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
