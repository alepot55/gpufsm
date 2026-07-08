# TACO paper — adversarial review findings + hardening status (2026-07-03)

Adversarial reviewer pass (skeptical TACO reviewer). Verdict: **Major revision**
(borderline reject as-was). The built pass + oracle-gating + Nsight + A100 are
above-bar substance; the issues below were the reject-risk. Status per item.

## Fixed tonight (integrity/honesty — the reject-risk items)
- **M1 [FIXED]** Flagship decomposition didn't multiply to its anchor. Abstract said
  "10x decomposes as 3.4x × B × 2", but 10.1x is at batch 4096 while 3.4x (component A)
  and Table 1's staged decomposition (=26x) are at batch 16384. Verified vs CSVs
  (m0_anchor=10.1x@4096; m2f_numwarps A=2.77x@4096, 3.44x@16384). Fix: abstract now
  states the regret is batch-dependent (10.1x@4096 → 26x@16384 as the occupancy-gated
  component B saturates); A quoted as a 2.8–3.7x range; no false product claim.
- **M2 [FIXED]** Headline "cure" 4.2x is transpile-to-CUDA (nvcc), and the generated
  kernel beats hand-CUDA ~2.15x (soft baseline). Abstract now frames 4.2x as an
  out-of-band existence proof; the in-compiler passes are the load-bearing contribution.
- **M4 [FIXED]** ML generality oversold (witnesses use one-element-per-lane mapping no
  production kernel uses). Abstract rescoped: the law holds on ML-shaped witnesses under
  that mapping; whether it survives production warp-cooperative mappings = future work.
- **M8 [FIXED]** Adjacent 0.51x (≈2x) vs 3.3x read as contradiction. Now labelled:
  0.51x is the median head-to-head across the ≤64-state sweep; 3.3x is the specific
  Nsight-profiled config (32 states, batch 16384).
- **De-anon [FIXED]** Removed the "proposed upstream to Triton + RFC + links redacted"
  fingerprint (anonymity is a hard TACO requirement); kept the verifier's technical
  content (monotonicity condition) phrased neutrally.

## Remaining (for the revise cycle — do before/at submission)
- **M5 [DONE — decomposition]** The flagship staged decomposition now runs on the A100
  (Modal, gpufsm CUDA ext built for sm_80). It reproduces to the SAME 26x total with matching
  stage factors: num_warps 3.7x on BOTH; lane-packing 2.3x(A100)/2.6x(4070); residual-to-thread
  3.1x/2.8x. The batch-4096 anchor rescales 10.1->5.6x (clock-sensitive register-resident regime;
  direction persists). Folded into Threats. Data: paper2/data/cross_arch/{m0_anchor,m2f_numwarps,
  m2_lane_packed,m2e_worklist_packed}_*a100*.csv. NOTE: 2 earlier Modal runs produced FAKE echoes
  (build failed: missing g++, then clang) — caught by a hardened capture guard, deleted, not
  committed. STILL OPEN under M5: the built CURE (4.15x pass) on A100 (needs from-source cure build).
- **M3 [PARTIAL]** In-compiler cure is 4.15x synthetic but 1.14–1.25x on real SpMV/MoE. Lead
  the compiler section with real-workload numbers; present 4.15x as a mechanism-isolating bound.
- **M6 [OPEN, may be intrinsic]** Clean result lives in ≤64-state register-resident toy regime;
  real automata (Brill 42661 states) run sub-Gbps/confounded. Show the built cure on some real
  automaton even if the residual is confounded.
- **M9 [OPEN]** Thin stats: no CI95/IQR on load-bearing numbers (4.15x, 15.3x, 39x are point
  estimates). Add dispersion, esp. on the decomposition factors and the anchor (3 seeds exist).
- **M7 / tone [OPEN]** Tone down over-naming ("straggler law", "the cure", 8 bolded contributions
  → consolidate to 3–4). State the paper1 delta crisply. Frame mechanism as localizing a known
  latency-hiding effect to the tile/thread boundary.
- **Abstract [OPEN]** Still a ~34-line firehose; tighten to ~200 words.

## Bottom line
The integrity holes that risked a reject are closed. Biggest remaining credibility lever = M5
(flagship results on a 2nd GPU). The rest is tightening + honest reframing, all revise-cycle work.


## CGO 2027 pivot (2026-07-08, after TACO desk-reject "too premature")
All remaining items CLOSED with recipe-reproducible data (pinned wheel: triton 3.8.0@81a46fa +
versioned perlane_retire_full.patch; built on Modal CPU, run on Modal A100 + local RTX 4070):
- **M5 [CLOSED]** The built cure now reproduces on the A100: lock-step 1.6-3.8x by distribution,
  same flat cured floor (~75us), SpMV/MoE ~1.0x, verifier declines the out-of-scope latch. Data:
  paper2/data/cross_arch/cure_wheel_a100.csv + cure_nvidia_a100-sxm4-40gb.log.
- **M9 [CLOSED]** Dispersion added: geometric-law cure median 2.46x IQR [2.46,2.50] n=5 seeds
  (cure_ci.py). Straggler law refit on the pinned recipe: t=32.4+1.09*E[warp-max], R2=0.997;
  floor 43.0+-1.6us; out-of-sample 1.5% mean / 2.1% max (was 5%/7.5%); single-straggler 7.2x.
- **M3 [CLOSED]** Honest real-workload framing: with the pinned recipe SpMV/MoE gains are ~1.0x
  (the old 1.14/1.25x did not reproduce); reframed as thesis-confirmation (recoverable regret
  scales with per-step control). The 4.15x headline replaced by the reproducible 2.3-6.7x range.
- **NEW (integrity)** The old local-build numbers (4.15x, 2.5-7.3x, SpMV 1.14x, MoE 1.25x,
  straggler fit 50.3+1.08) were NOT reproducible from the versioned recipe and were REPLACED
  in the paper by wheel-recipe numbers. Slope stability (1.08 vs 1.09) reported as build-stability.
- **NEW (soundness)** cure_rejection witness: the verifier correctly DECLINES the accumulate-OR
  early-exit latch on both GPUs (PTX redux stays 1) -> scope explicit, sound by default.
Paper: paper2/gpufsm_cgo.tex (sigplan, 10pp incl. refs vs 11pp text limit, 0 overfull).
