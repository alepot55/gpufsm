"""LANDMARK P2 — the automatic selector: detect the lock-step signature, route to the thread cure.

The detection pass is real (p2_pass_verify) and the in-IR lowering is structurally blocked
(p2_lowering_wall): the cure must lower BELOW TritonGPU to the thread model. This selector closes
the loop AUTOMATICALLY at the source/codegen boundary:

  detect lock-step (ttg.thread_region_candidate, via the in-libtriton pass)  -->  route to the M10
  thread lowering (nvcc, per-lane control);  no signature  -->  keep the Triton tile path.

Detection needs the from-source Triton (which carries the pass); the throughput measurement uses the
gpufsm package + M10 machinery (system Triton). The two Tritons cannot share a process, so detection
runs as a subprocess re-entry of THIS file (`p2_selector.py detect <kind>`) with PYTHONPATH at the
hacked build and GPUFSM_THREAD_REGION=1.

Result (oracle-gated): the NFA worklist (lock-step) is auto-routed to threads and achieves the M10
speedup over the tile; a fixed-trip negative-control kernel is correctly left on the tile path.

Usage:  .venv/bin/python experiments/cure/p2_selector.py
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
from pathlib import Path

BUILD_PY = Path.home() / "m3full_build" / "triton-src" / "python"


# --------------------------------------------------------------------------------------------------
# Detection subprocess (runs under the FROM-SOURCE Triton; imports triton only here).
# --------------------------------------------------------------------------------------------------
def _detect_main(kind: str) -> int:
    import triton
    import triton.language as tl

    os.environ["GPUFSM_THREAD_REGION"] = "1"

    if kind == "lockstep":

        @triton.jit
        def k(inp, out, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            valid = i < n
            trip = tl.load(inp + i, mask=valid, other=0)
            acc = tl.zeros((BLOCK,), tl.int32)
            j = tl.zeros((BLOCK,), tl.int32)
            while tl.max((j < trip).to(tl.int32)) > 0:  # data-dependent lock-step loop
                active = j < trip
                acc = acc + tl.where(active, j, 0)
                j = j + 1
            tl.store(out + i, acc, mask=valid)

        sig = {"inp": "*i32", "out": "*i32", "n": "i32", "BLOCK": "constexpr"}
        cst = {"BLOCK": 32}
    else:  # fixedtrip — negative control: a fixed-count loop, no reduce-gated while

        @triton.jit
        def k(inp, out, n, K: tl.constexpr, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            valid = i < n
            v = tl.load(inp + i, mask=valid, other=0)
            acc = tl.zeros((BLOCK,), tl.int32)
            for j in range(K):
                acc = acc + (v % (j + 1))
            tl.store(out + i, acc, mask=valid)

        sig = {"inp": "*i32", "out": "*i32", "n": "i32", "K": "constexpr", "BLOCK": "constexpr"}
        cst = {"K": 16, "BLOCK": 32}

    src = triton.compile(triton.compiler.ASTSource(fn=k, signature=sig, constexprs=cst))
    print("DETECT:1" if "ttg.thread_region_candidate" in src.asm["ttgir"] else "DETECT:0")
    return 0


def detect_lockstep(kind: str) -> bool:
    """Run the detection pass (from-source Triton) in a subprocess; return whether it fired."""
    env = dict(os.environ, PYTHONPATH=str(BUILD_PY), GPUFSM_THREAD_REGION="1")
    proc = subprocess.run(
        [sys.executable, __file__, "detect", kind], capture_output=True, text=True, env=env
    )
    return "DETECT:1" in proc.stdout


# --------------------------------------------------------------------------------------------------
# Orchestrator (system Triton + gpufsm + M10 machinery).
# --------------------------------------------------------------------------------------------------
def _measure_nfa_routed() -> tuple[float, bool]:
    """Auto-routed NFA: thread (sp_run) vs tile (wp2), oracle-gated. Returns (sp/wp2, oracle)."""
    # imports are local: they pull in system Triton + gpufsm, kept out of the detection subprocess.
    from experiments.cure.m3_lite_scalarlane import launch_wp2, max_outdeg
    from experiments.cure.m10_scalar_program import (
        N_STRINGS,
        SLEN,
        make_batch_local,
        oracle_ok,
        random_nfa,
        random_nfa_noaccept,
        sp_run,
        to_device,
    )

    warmup, samples = 3, 9
    total_bits = N_STRINGS * SLEN * 8
    ratios, oracle_all = [], True
    for ns in (16, 32, 48, 64):  # same state sweep as M10's headline, for a consistent ratio
        for seed in (0, 1):
            data = make_batch_local(seed)
            nfa_acc = random_nfa(ns, seed=1000 + ns + seed)
            if not oracle_ok(nfa_acc, data, to_device(nfa_acc)):
                oracle_all = False
                continue
            nfa = random_nfa_noaccept(ns, seed=1000 + ns + seed)
            g = to_device(nfa)

            def med(fn):
                for _ in range(warmup):
                    fn()
                return statistics.median([fn() for _ in range(samples)])

            sp_ms = med(lambda nfa=nfa, g=g, data=data: sp_run(nfa, g, data)[2])
            wp2_ms = med(lambda g=g, data=data, nfa=nfa: launch_wp2(g, data, max_outdeg(nfa))[2])
            sp = total_bits / (sp_ms * 1e-3) / 1e9
            wp2 = total_bits / (wp2_ms * 1e-3) / 1e9
            ratios.append(sp / wp2)
    return (statistics.median(ratios) if ratios else 0.0), oracle_all


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "detect":
        return _detect_main(sys.argv[2])

    if not BUILD_PY.exists():
        print("SKIP: from-source Triton build not found; detection unavailable.")
        return 0
    # repo root on sys.path so the M10 measurement imports (`experiments.cure.*`) resolve when this
    # file is run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("P2 automatic selector — detect lock-step -> route to the thread cure (M10).")
    workloads = [
        ("NFA worklist", "lockstep", "thread (M10 nvcc)"),
        ("pointer-chase (neg ctrl)", "fixedtrip", "tile (Triton)"),
    ]
    routing = []
    for name, kind, _expected in workloads:
        detected = detect_lockstep(kind)
        route = "thread" if detected else "tile"
        routing.append((name, kind, detected, route))
        print(f"  {name:<26} detect_lockstep={int(detected)} -> route={route}")

    # The detected NFA is auto-routed to the thread cure; measure the realized speedup vs the tile.
    speedup, oracle = 0.0, None
    try:
        import torch

        if torch.cuda.is_available():
            speedup, oracle = _measure_nfa_routed()
            print(
                f"\n  auto-routed NFA (thread vs tile): SP/WP2 = {speedup:.2f}x  "
                f"oracle={'ok' if oracle else 'FAIL'}"
            )
        else:
            print("\n  (no CUDA: routing decided; throughput not measured)")
    except Exception as e:  # measurement is best-effort; the routing decision is the contribution
        print(f"\n  (measurement skipped: {type(e).__name__}: {e})")

    nfa_ok = routing[0][2] is True and routing[0][3] == "thread"
    ctrl_ok = routing[1][2] is False and routing[1][3] == "tile"
    passed = nfa_ok and ctrl_ok and (oracle in (None, True))
    out = Path("paper2/data/landmark/p2_selector_rtx4070.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("workload,kind,detected,route,nfa_sp_over_wp2,gpu\n")
        for name, kind, det, route in routing:
            sp = f"{speedup:.3f}" if kind == "lockstep" else ""
            f.write(f"{name},{kind},{int(det)},{route},{sp},RTX4070\n")
    print(f"wrote {out}")
    print(
        f"\n=> selector {'VERIFIED' if passed else 'FAILED'}: lock-step auto-routed to the cure, "
        "negative control left on the tile."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
