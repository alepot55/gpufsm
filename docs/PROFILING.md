# Profiling notes (Nsight)

## Status: Nsight Compute counters BLOCKED on this host (permissions)

`ncu` (Nsight Compute) is installed (`/usr/local/cuda/bin/ncu`) but collecting hardware
counters fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

GPU performance counters are admin-gated by the driver. To enable (one-time, needs sudo):

- **Quick (per session):** run ncu under sudo — `sudo ncu ...`.
- **Permanent:** allow non-admin profiling, then reboot:
  ```
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
  sudo update-initramfs -u && sudo reboot
  ```
  See https://developer.nvidia.com/ERR_NVGPUCTRPERM

Once enabled, profile a single clean kernel launch with:

```
ncu --target-processes all -k regex:worklist --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \
    python scripts/profile_target.py cuda worklist 256
```

`scripts/profile_target.py` issues exactly one `run_batch` (one kernel launch) so the
profile is clean. Compare `cuda/worklist` vs `cuda/multistream` (full-scan):
- compute % >> DRAM % ⇒ compute-bound; the reverse ⇒ memory-bound.

`nsys` (Nsight Systems, timeline tracing) does NOT need counter permissions and can show
the `multistream_async` H2D/kernel/D2H overlap; use it for the sync→async evidence.

## The compute-bound claim does NOT depend on ncu

The central regime claim is already established by *controlled experiment*, not counters:

1. **Ablation control:** `multistream_shared` stages the CSR into shared memory (modeled
   global traffic = 0) yet ties `multistream` (global CSR) and `multistream_async` to within
   the bootstrap CI at every size (`paper/data/sweep_techniques.csv`). If the kernel were
   memory-bound, removing CSR global traffic would help — it does not ⇒ compute-bound.
2. **Scaling:** throughput ∝ 1/n² for the full-scan kernels (the O(n²) eps-closure), fit by
   the cost model to <1% at large n (`docs/RESULTS_COSTMODEL.md`).
3. **Cure:** the work-efficient `worklist` kernel (O(active), no O(n²)) is 250×–10000× faster
   and reaches 15–132 Gbps — the regime where memory layout *can* matter; profiling it (once
   ncu is unblocked) is the next confirmatory step.

So Nsight counters are **confirmatory, not load-bearing** for the paper's argument.
