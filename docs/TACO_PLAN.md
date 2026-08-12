# ACM TACO submission plan (paper 2 — the built cure)

**Target: ACM TACO** (Transactions on Architecture and Code Optimization) — decided 2026-07-01
("pubblicazione certa"). Journal, ROLLING submission (no deadline), revise-and-resubmit
(Accept / Minor / Major / Reject; Major → high eventual-acceptance for solid work). Accepted
papers present at HiPEAC (conference visibility too). Submit via ScholarOne mc.manuscriptcentral.com/taco.
Exactly on-topic (code generation/optimization). Serves the NVIDIA hire (peer-reviewed, respected).

## Why TACO over a conference (PPoPP/CGO)
Certainty: our single-GPU weakness is a *fixable revision* at a journal ("add A100" → Major Revision →
we add it → accept), not a fatal reject as at a competitive conference. No deadline pressure → no rushed
submission. Rolling → submit when genuinely ready + strong.

## Status
- ✅ Paper in TACO format: `paper2/gpufsm_taco.tex` (acmsmall), builds clean — 15pp, 0 overfull/undefined/fatal.
- ✅ Already anonymous (double-blind-ready); built-cure story complete (4.15x/39x/table/SpMV/MoE/Nsight).
- ✅ Turnkey A100 script (scripts/a100_validate.sh) for the datacenter validation.
- Backups: gpufsm_ppopp.tex (sigconf), gpufsm2.tex (IEEE).

## Plan (no deadline → aim for a STRONG, thorough submission; journals reward depth)
1. **A100 datacenter validation (TOMORROW, user provides RunPod ~$5-10):** run a100_validate.sh; fold
   cross-arch numbers into results + Threats. Pre-empts the #1 revision request. This is the biggest lever.
2. **Journal-depth polish:** journals reward thoroughness — expand where a conference forced brevity:
   fuller related work, more mechanism detail, the complete regret-law + built-cure exposition, an artifact
   appendix. Add CCS concepts (\ccsdesc) + keywords (TACO/acmart requirement).
3. **Self-review for rigor:** every claim oracle-gated + CSV-traced; no overclaim; anonymity preserved.
4. **Submit** (USER: create ScholarOne account + upload PDF + cover letter). I prepare everything incl. a
   cover letter drafting the contribution + suggesting reviewers/associate editors (compiler/GPU area).

## Notes
- No page limit stress → the 15pp is fine; can grow with A100 results + depth.
- Confirm TACO blinding policy at submission (keep `review,anonymous` for now; easy to flip).
- Keep the merged/aimed Triton PR (#10766) + the built cure as the artifact backbone (reproducibility is a
  TACO strength — we already gate every number).
