# PPoPP 2027 submission plan (paper 2 — the built cure)

**Target: PPoPP 2027** (ACM Symposium on Principles and Practice of Parallel Programming),
co-located with HPCA/CGO/CC 2027, Salt Lake City, 20–24 Mar 2027. **Deadline ~3 Aug 2026**
(historical PPoPP pattern; official CFP not yet posted — MONITOR conf.researchr.org/home/PPoPP-2027).
Format: ACM two-column (acmart sigconf), 10 pages excl. references, **double-blind**.

Why PPoPP: the paper's core is a *parallel-execution-model* result — tile-SPMD vs thread-SIMT, intra-warp
latency hiding, per-lane retirement — squarely PPoPP territory (not just a compiler pass).

## Status
- ✅ Paper already anonymous (author "Anonymous", zero repo/name/github refs) → double-blind-ready.
- ✅ Converted IEEEtran → acmart (`paper2/gpufsm_ppopp.tex`), compiles: 8pp, refs resolve, 0 fatal.
  (fonts: using lmodern fallback — install libertine/newtxmath for camera-ready.)
- ⏳ 35 overfull hboxes (ACM columns narrower than IEEE) — fix tables/math/texttt.

## Timeline (≈5 weeks, everything by early August)
- **Wk1 (now–~Jul 8):** finish acmart conversion — fix all overfulls, tune tables to ACM column width,
  verify ≤10pp excl refs, 0 overfull/undefined. Frame abstract+intro for the PPoPP (parallelism) audience.
- **Wk2 (~Jul 8–15):** content polish for PPoPP reviewers — sharpen the parallel-execution framing,
  related work vs PPoPP-relevant prior art (warp specialization, GPU divergence, iNFAnt/ngAP), tighten
  the built-cure results table.
- **Wk3 (~Jul 15–22):** **RunPod A100 datacenter validation** (USER: spin up an A100 pod ~2–3h, ~$5–10).
  Run scripts/run_cross_arch.sh (cure + witnesses) on A100/H100; fold cross-arch numbers into results +
  Threats (closes the single-GPU objection). I prepare everything; user only starts the pod.
- **Wk4 (~Jul 22–29):** final polish — anonymized artifact appendix, self-review, every number CSV-traced,
  LaTeX clean, page-limit fit.
- **Wk5 (~Jul 29–Aug 3):** register abstract + submit on HotCRP (USER: create account + click submit;
  I prepare the anonymized PDF + abstract text).

## Risks / notes
- Deadline is historical-pattern, not yet official → monitor CFP; if it shifts later, we gain buffer.
- Single consumer GPU is the main reviewer risk → RunPod A100 validation (Wk3) is the mitigation.
- If RunPod isn't available in time: submit single-GPU with datacenter as pre-registered falsifiable
  prediction (already the paper's Threats stance), but the A100 run materially strengthens it.
