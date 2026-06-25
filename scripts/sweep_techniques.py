"""Rigorous multi-technique throughput sweep -> versioned CSV (median + CI95).

Measures the parallel multi-stream-family techniques (the throughput-meaningful ones)
across automaton sizes, on a fixed input batch, with non-Gaussian-appropriate
statistics (median + percentile-bootstrap 95% CI over per-run batch-kernel times).
Captures GPU + library versions for reproducibility. Output feeds the paper figures.

The single-program dense/bitpacked kernels are excluded: they are one latency-bound
GPU thread (~0 throughput) and are reported separately as the naive baseline.

Run on a GPU box:  python scripts/sweep_techniques.py [out.csv]
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

from gpufsm.api import run_batch
from gpufsm.nfa import NFABuilder
from gpufsm.registry import Backend, available_backends, list_techniques

_MULTISTREAM = {
    "multistream",
    "multistream_shared",
    "multistream_async",
    "worklist",
}
_SIZES = [32, 64, 128, 256, 500]
_N_STRINGS = 2048
_SLEN = 256
_SAMPLES = 9
_WARMUP = 3


def env_info() -> dict[str, str]:
    info = {"gpu": "?", "torch": "?", "triton": "?", "warp": "?", "cuda": "?"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "?"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod, key in (("triton", "triton"), ("warp", "warp")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            pass
    return info


def random_nfa(n: int, seed: int) -> NFABuilder:
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def make_batch() -> tuple[list[bytes], int]:
    rng = np.random.default_rng(0)
    flat = rng.integers(ord("a"), ord("a") + 5, size=_N_STRINGS * _SLEN, dtype=np.uint8).tobytes()
    return [flat[i * _SLEN : (i + 1) * _SLEN] for i in range(_N_STRINGS)], _N_STRINGS * _SLEN


def bootstrap_ci95(samples: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return (float(arr[0]) if arr.size else 0.0, float(arr[0]) if arr.size else 0.0)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(arr, size=(iters, arr.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def measure(nfa, backend, technique, batch, total_bytes) -> dict | None:
    for _ in range(_WARMUP):
        run_batch(nfa, batch, backend=backend, technique=technique)
    samples = []
    for _ in range(_SAMPLES):
        km = run_batch(nfa, batch, backend=backend, technique=technique)[0].kernel_ms
        if km > 0:
            samples.append(km)
    if not samples:
        return None
    median = float(np.median(samples))
    lo, hi = bootstrap_ci95(samples)
    gbps = (total_bytes * 8.0) / (median * 1e-3) / 1e9
    return {
        "median_ms": round(median, 5),
        "ci95_lo_ms": round(lo, 5),
        "ci95_hi_ms": round(hi, 5),
        "throughput_gbps": round(gbps, 5),
        "samples": len(samples),
    }


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/data/sweep_techniques.csv")
    env = env_info()
    backends = [
        b for b in available_backends() if b in (Backend.TRITON, Backend.CUDA, Backend.WARP)
    ]
    if not backends:
        print("no GPU backend — run on a GPU box")
        return
    batch, total_bytes = make_batch()
    rows = []
    for be in backends:
        for te in list_techniques(be):
            if te not in _MULTISTREAM:
                continue
            for n in _SIZES:
                if be is Backend.WARP and n > 64:
                    continue
                nfa = random_nfa(n, seed=1000 + n)
                try:
                    m = measure(nfa, be, te, batch, total_bytes)
                except (ValueError, RuntimeError) as e:
                    print(f"{be.value:7}/{te:18} n={n:4d}  SKIP ({type(e).__name__}: {str(e)[:60]})")
                    continue
                if m is None:
                    continue
                row = {
                    "gpu": env["gpu"],
                    "backend": be.value,
                    "technique": te,
                    "num_states": n,
                    "n_strings": _N_STRINGS,
                    "slen": _SLEN,
                    **m,
                    "torch": env["torch"],
                    "triton": env["triton"],
                    "warp": env["warp"],
                    "cuda": env["cuda"],
                }
                rows.append(row)
                ci = f"[{m['ci95_lo_ms']:.4f},{m['ci95_hi_ms']:.4f}]"
                print(
                    f"{be.value:7}/{te:18} n={n:4d}  median={m['median_ms']:9.4f} ms  "
                    f"CI95={ci}  {m['throughput_gbps']:8.4f} Gbps"
                )
    if not rows:
        print("no rows measured")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
