# Submission plan — Paper 2: "From Diagnosis to Cure"

Source: `paper2/gpufsm2.tex` (IEEEtran, 6pp + refs, compiles clean: 0 undefined / 0 overfull).
Companion (paper 1, the diagnosis): the Two-Faces abstraction-regret study. This paper is the **cure**.

## Elevator pitch (1 paragraph)
Tile-based GPU DSLs (Triton, and NVIDIA's cuTile/Tile IR) only handle regular dense tensors; on
irregular, data-dependent control flow they collapse, and no one has a principled, automatic fix. We
take the ${\sim}10\times$ Triton-vs-CUDA gap on a work-efficient NFA worklist apart to the instruction
level: it is a launch-config artifact (${\sim}3.4\times$), a lane-packing-recoverable redundancy, and an
**irreducible ${\sim}2\times$ residual that is abstraction-denied intra-warp memory-level parallelism** —
at matched occupancy and *fewer* warp-instructions than CUDA, a lock-step tile still stalls $15.3\times$
more on the dependent next-state load (issue $9.9\%$ vs.\ $41\%$). We show it is regime-dependent (the
memory-bound DFA closes to $1.05\times$ past L2), name the **missing IR primitive** (a per-lane sub-tile
loop/exit op), and **implement the cure**: lowering the same per-lane source to the thread model runs
$4.2\times$ faster and matches/exceeds hand-CUDA. We take it into the **real compiler** — a TritonGPU
MLIR pass in `libtriton` that *detects* the lock-step region, a *proof that the in-tile-IR lowering is
structurally impossible* (`scf.condition` is a single `i1`), and an *automatic selector* that routes the
detected region to the thread cure ($3.9\times$). Finally the phenomenon **generalizes**: across six
oracle-gated irregular workloads the regret is created by scalar control flow (not memory irregularity),
through two measured channels, with a graph pointer-chase negative control at exactly $1.00\times$.

## Contributions (condensed from the paper)
1. A **falsifiable decomposition** of the tile-SPMD automata regret (artifact / recoverable / irreducible),
   each measured and Nsight-attributed.
2. The irreducible residual identified as **abstraction-denied intra-warp latency hiding** — not
   instruction count, occupancy, or bandwidth — measured (stall composition, issue activity, roofline).
3. The residual is **regime-dependent** (latency-bound NFA vs memory-bound DFA), one mechanism at two
   latency scales.
4. A **launch-config methodological caution** (`num_warps=4` inflates worklist regret ${\sim}3.4\times$).
5. The **missing-primitive design** + payoff bound + thread-model existence proof (CUDA $1.0\times$,
   Warp $0.9\times$), distinguished from Gluon's per-thread *layout*.
6. **The cure, implemented** ($4.2\times$ over the tile, matches/exceeds hand-CUDA — closes the residual
   by construction).
7. **In the real compiler**: a TritonGPU detection pass in `libtriton`; a structural-impossibility proof
   for the in-IR lowering (naming the missing per-lane loop/exit op); an automatic detect-and-lower
   selector ($3.9\times$).
8. A **generality law** across six oracle-gated irregular workloads (two channels — issue starvation +
   masked-lane waste — over a tile-lowering baseline), with a pointer-chase negative control at $1.00\times$.

## Target venue (rationale)
- **ASPLOS / PLDI / OOPSLA (primary).** The bar is a working artifact + a principle + a surprising
  general result + AE. We have: a real in-`libtriton` MLIR pass, a falsifiable structural result that
  *names the missing IR primitive*, an implemented+measured cure, a generality law with a correct
  negative control, and a one-command artifact. The compiler angle (detect-and-lower irregular per-lane
  regions; the IR-expressibility wall) is squarely PL/architecture.
- **CGO (strong fit / fast path).** The contribution is literally a compiler pass + IR analysis; CGO's
  scope and cadence fit the "implemented in the compiler" framing well.
- **IISWC / PACT (honest floor).** If a deadline forces it, the characterization + regret law alone clear
  these; but the pass + wall + selector lift us above the workload-characterization tier.
- **Multi-GPU caveat for the top tier:** reviewers will want ≥2 architectures. P3 is single-GPU-validated
  with a turnkey cross-arch harness ready (below) — running it on one A100/H100 closes this for
  camera-ready and is one command.

## Artifact evaluation
- **Index:** `experiments/cure/README.md` maps every claim → artifact → command → output CSV.
- **One-command cross-arch:** `scripts/run_cross_arch.sh` (or `experiments/cure/p3_cross_arch.py`).
- **The compiler pass:** `experiments/cure/triton_thread_region_pass/` (`ThreadRegion.cpp` +
  `registration.patch` + README; base Triton commit `c05aa65`), reproducible via `docs/P2_PASS_DESIGN.md`.
- **Falsifiable probes:** `p2_pass_verify.py` (pass fires in libtriton), `p2_lowering_wall.py` (the
  structural wall), the Gluon probe. All correctness-gated against a CPU oracle; every number CSV-traced.

## Known gaps (be upfront)
- **Multi-GPU:** all numbers are RTX 4070; the mechanism is architecture-general but absolute factors
  (and the DFA L2 crossover) need an A100/H100 re-run (harness ready).
- **Regime of the clean ${\sim}2\times$:** the sharpest residual is the $\le 64$-state register-resident
  regime; past 64 states both baselines lose the register advantage (confounded head-to-head), and
  ANMLZoo-scale automata make the per-lane multi-word scatter explode (itself a manifestation of the tile
  limitation). Real automata are oracle-valid but sub-Gbps (an *algorithmic* cost, orthogonal).
- **In-IR lowering:** realized below TritonGPU (thread model) because the tile IR cannot express it;
  in-IR realization waits on the named per-lane primitive (the falsifiable "next step").

## NVIDIA-interview framing
Lead with the **Triton-MLIR pass** (real, in `libtriton`) and the **named missing IR primitive**
(per-lane sub-tile loop/exit op) — a concrete, actionable proposal for cuTile/Tile IR, which today
concedes the irregular case to a manual SIMT fallback. Go down to issue-rate / `long_scoreboard` stalls
and the roofline to show it is structurally latency-bound, not a bad kernel. Discuss the
Hopper/Blackwell angle: independent thread scheduling (ITS) is exactly the hardware substrate the
lowering needs, and warp-specialization (Tawa/cuTile) is orthogonal — it restructures *dense* tiles,
not irregular per-lane control. The whole arc — diagnose to the instruction, name the primitive,
implement and measure the cure, prove why it must live below the tile IR — is the engineer's story.
