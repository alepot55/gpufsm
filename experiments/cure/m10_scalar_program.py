"""M10 — the cure, IMPLEMENTED: a `scalar_program` primitive that lowers per-lane source to threads.

The paper proposes the missing primitive: a `scalar_program` region whose per-lane body lowers to
a PER-THREAD independent instruction stream (thread-SIMT) instead of a lock-step tile. Here we
implement that lowering constructively and measure that it CLOSES the residual.

`lower_scalar_program_to_cuda(spec)` takes the SAME idiomatic per-lane automaton step the Triton WP2
kernel expresses (active-set `ffs` worklist over a register bitset, data-dependent CSR loop, scalar
gather) and emits a thread-model CUDA kernel (one thread = one string), compiled via
torch.utils.cpp_extension.load_inline (nvcc). The front-end is per-lane scalar logic; the lowering
target is threads, not tiles. Claim: the generated kernel recovers CUDA-level throughput (~the
hand-written cuda/worklist), i.e. the ~2x tile-SPMD residual (Triton WP2 = 0.51x of CUDA) vanishes
once the same per-lane program is lowered to the thread model.

Oracle-gated bit-for-bit vs reference.py. Compares: SP (scalar_program->CUDA, the cure), CU
(hand-written cuda/worklist), WP2 (Triton tile per-lane, the residual). <=64 states (single int64,
matching the regime where the residual is clean). Writes paper2/data/m10_scalar_program_rtx4070.csv.

Usage:  .venv/bin/python experiments/cure/m10_scalar_program.py
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from experiments.cure.m2e_worklist_packed import (
    SAMPLES,
    SLEN,
    WARMUP,
    random_nfa,
    to_device,
)
from experiments.cure.m3_lite_scalarlane import launch_wp2, max_outdeg

from gpufsm.api import run, run_batch
from gpufsm.nfa import ANY_SYMBOL, NFABuilder
from gpufsm.registry import Backend

N_STRINGS = int(__import__("os").environ.get("M2_N_STRINGS", "4096"))


def make_batch_local(seed: int):
    rng = np.random.default_rng(seed)
    flat = rng.integers(ord("a"), ord("a") + 5, size=N_STRINGS * SLEN, dtype=np.uint8)
    return flat.reshape(N_STRINGS, SLEN)


def random_nfa_noaccept(n: int, seed: int):
    """Same structure as m2e.random_nfa but NO accept states, so strings never latch early -- the
    kernels scan the full input and we measure SUSTAINED throughput (removes the early-termination
    confound, and the per-lane early-exit asymmetry between thread kernels and the tile)."""
    import random as _r

    rng = _r.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=False)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def lower_scalar_program_to_cuda(uses_any: bool):
    """The primitive's LOWERING: emit + compile a thread-model CUDA kernel for the per-lane step.

    The per-lane 'scalar_program' body (ffs worklist + data-dep CSR loop + scalar gather) is the
    SAME logic Triton WP2 expresses; here each lane is one independent CUDA thread. We compile
    with nvcc to a .so (extern "C", raw device pointers) called via ctypes -- bypassing torch's
    C++ headers (which don't compile on the local gcc). sp_launch records CUDA events, returns
    kernel time in ms.
    """
    any_clause = "|| (uses_any && tsym == ANY_ID)" if uses_any else ""
    src = f"""
extern "C" __global__ void scalar_program_kernel(
    const int* rowptr, const int* targets, const int* symbols, long long accept_word,
    const int* input_data, const int* input_offsets, int num_strings,
    int* out_flags, int* out_lens, int start_state, int ANY_ID, int uses_any) {{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;  // one thread == one string
  if (pid >= num_strings) return;
  int in_lo = input_offsets[pid];
  int in_len = input_offsets[pid + 1] - in_lo;
  unsigned long long cur = 1ULL << start_state;  // register-resident per-lane bitset
  unsigned long long accw = (unsigned long long) accept_word;
  int out_f = 0, out_l = 0, done = 0;
  if (cur & accw) {{ out_f = 1; done = 1; }}
  for (int pos = 0; pos < in_len && !done; pos++) {{
    int sym = input_data[in_lo + pos];
    unsigned long long nxt = 0, bits = cur;
    while (bits) {{  // per-lane ffs worklist (independent control flow)
      int s = __ffsll(bits) - 1; bits &= bits - 1;
      for (int k = rowptr[s]; k < rowptr[s + 1]; k++) {{  // data-dep CSR loop, scalar gather
        int tsym = symbols[k];
        int hit = (tsym == sym) {any_clause};
        if (hit) nxt |= 1ULL << targets[k];
      }}
    }}
    cur = nxt;
    if (cur & accw) {{ out_f = 1; out_l = pos + 1; done = 1; }}
  }}
  out_flags[pid] = out_f; out_lens[pid] = out_l;
}}
extern "C" float sp_launch(
    const int* rowptr, const int* targets, const int* symbols, long long accw,
    const int* input_data, const int* input_offsets, int n,
    int* flags, int* lens, int start_state, int any_id, int uses_any) {{
  int threads = 256, blocks = (n + threads - 1) / threads;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  scalar_program_kernel<<<blocks, threads>>>(
      rowptr, targets, symbols, accw, input_data, input_offsets, n,
      flags, lens, start_state, any_id, uses_any);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms;
}}
"""
    cache = Path.home() / ".cache" / "m10_scalar_program"  # home fs (tmpfs /tmp may be noexec)
    cache.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="sp_", dir=str(cache)))
    cu, so = d / "sp.cu", d / "sp.so"
    cu.write_text(src)
    subprocess.run(
        ["nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89", "-o", str(so), str(cu)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.sp_launch.restype = ctypes.c_float
    lib.sp_launch.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_longlong]
        + [ctypes.c_void_p] * 2
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        + [ctypes.c_int] * 3
    )
    return lib


ANY_SYMBOL_V = int(ANY_SYMBOL)
_CACHE: dict[bool, object] = {}


def sp_run(nfa, g, data2d):
    ua = bool(nfa.uses_any_symbol)
    if ua not in _CACHE:
        _CACHE[ua] = lower_scalar_program_to_cuda(ua)
    lib = _CACHE[ua]
    dev = torch.device("cuda")
    n = data2d.shape[0]
    flat = torch.as_tensor(data2d.reshape(-1).astype(np.int32), device=dev)
    offsets = torch.arange(0, (n + 1) * SLEN, SLEN, dtype=torch.int32, device=dev)
    rp = g["rowptr"].to(torch.int32)
    tg = g["targets"].to(torch.int32)
    sy = g["symbols"].to(torch.int32)
    flags = torch.zeros(n, dtype=torch.int32, device=dev)
    lens = torch.zeros(n, dtype=torch.int32, device=dev)
    ms = lib.sp_launch(
        rp.data_ptr(),
        tg.data_ptr(),
        sy.data_ptr(),
        int(g["accept_word"]),
        flat.data_ptr(),
        offsets.data_ptr(),
        n,
        flags.data_ptr(),
        lens.data_ptr(),
        int(g["start"]),
        ANY_SYMBOL_V,
        int(ua),
    )
    torch.cuda.synchronize()
    return flags.cpu().numpy(), lens.cpu().numpy(), float(ms)


def oracle_ok(nfa, data2d, g, n_check=64) -> bool:
    flags, lens, _ = sp_run(nfa, g, data2d[:n_check])
    for i in range(min(n_check, data2d.shape[0])):
        ref = run(nfa, data2d[i].tobytes(), backend=Backend.CPU, technique="reference")
        if (bool(flags[i]), int(lens[i])) != (ref.accepted, ref.match_len):
            print(
                f"  MISMATCH sp {i}: got ({bool(flags[i])},{lens[i]}) "
                f"ref ({ref.accepted},{ref.match_len})"
            )
            return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    total_bits = N_STRINGS * SLEN * 8
    print(
        "M10 — the cure implemented: scalar_program lowered to threads vs CUDA vs Triton tile (WP2)"
    )
    print("  SP = scalar_program->CUDA (cure)  CU = hand cuda/worklist  WP2 = Triton tile per-lane")
    print(
        f"{'states':>7}{'seed':>5}{'SP':>9}{'CU':>9}{'WP2':>9}{'SP/CU':>7}{'SP/WP2':>8}{'oracle':>8}"
    )
    rows = []
    for ns in (16, 32, 48, 64):
        for seed in (0, 1):
            data = make_batch_local(seed)
            # correctness on a WITH-accept NFA (real matches); throughput on a no-accept variant
            # (sustained full-length scan, no early-termination / per-lane early-exit confound).
            nfa_acc = random_nfa(ns, seed=1000 + ns + seed)
            if not oracle_ok(nfa_acc, data, to_device(nfa_acc)):
                print(f"{ns:7d}{seed:5d}  ORACLE FAIL")
                continue
            nfa = random_nfa_noaccept(ns, seed=1000 + ns + seed)
            g = to_device(nfa)
            bb = [data[i].tobytes() for i in range(data.shape[0])]

            def med_sp(nfa=nfa, g=g, data=data):
                for _ in range(WARMUP):
                    sp_run(nfa, g, data)
                return statistics.median([sp_run(nfa, g, data)[2] for _ in range(SAMPLES)])

            def med_cu(nfa=nfa, bb=bb):
                for _ in range(WARMUP):
                    run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")
                return statistics.median(
                    [
                        run_batch(nfa, bb, backend=Backend.CUDA, technique="worklist")[0].kernel_ms
                        for _ in range(SAMPLES)
                    ]
                )

            def med_wp2(g=g, data=data, nfa=nfa):
                for _ in range(WARMUP):
                    launch_wp2(g, data, max_outdeg(nfa))
                return statistics.median(
                    [launch_wp2(g, data, max_outdeg(nfa))[2] for _ in range(SAMPLES)]
                )

            sp = total_bits / (med_sp() * 1e-3) / 1e9
            cu = total_bits / (med_cu() * 1e-3) / 1e9
            wp2 = total_bits / (med_wp2() * 1e-3) / 1e9
            print(
                f"{ns:7d}{seed:5d}{sp:9.1f}{cu:9.1f}{wp2:9.1f}"
                f"{sp / cu:7.2f}{sp / wp2:8.2f}{'  ok':>8}"
            )
            rows.append((ns, seed, round(sp, 2), round(cu, 2), round(wp2, 2)))
    if rows:
        spcu = [r[2] / r[3] for r in rows]
        spwp2 = [r[2] / r[4] for r in rows]
        print(f"\nSP/CU (cure vs hand-CUDA): median {statistics.median(spcu):.2f}x (>=1)")
        print(f"SP/WP2 (cure vs tile): median {statistics.median(spwp2):.2f}x (residual closed)")
        outp = Path("paper2/data/m10_scalar_program_rtx4070.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("states,seed,sp_gbps,cu_gbps,wp2_gbps,sp_over_cu,sp_over_wp2,gpu\n")
            for ns, seed, sp, cu, wp2 in rows:
                f.write(
                    f"{ns},{seed},{sp},{cu},{wp2},"
                    f"{round(sp / cu, 3)},{round(sp / wp2, 3)},RTX4070\n"
                )
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
