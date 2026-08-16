# ACM TACO submission — step-by-step checklist

**Portal:** ScholarOne Manuscripts — https://mc.manuscriptcentral.com/taco
**Model:** rolling submission (no deadline); revise-and-resubmit. First response ~2 months.

## Files (all ready in `paper2/`)
- `gpufsm_taco.pdf` — the anonymized manuscript (16 pp, acmsmall, builds clean, 0 overfull/undefined).
  Verified anonymous: no author in PDF metadata/text, self-citation to the companion paper is `Anonymous`.
- Cover letter: `docs/TACO_COVER_LETTER.md` (paste into the portal's cover-letter box; fill name/affiliation
  ONLY in the portal, never in the PDF).

## Before you click submit — 3 things only you can do
1. **Author/affiliation** go in the ScholarOne form, NOT the PDF (the PDF stays anonymous). The paper is
   `\author{Anonymous}` on purpose.
2. **Confirm the blinding policy** on the TACO author page. The PDF is set `[review,anonymous]`, safe for
   single- or double-blind. If TACO wants non-anonymous at submission, flip to `[manuscript]` and rebuild
   (I can do that in one edit) — but anonymous is the safe default.
3. **Suggested editors/reviewers** (portal asks): compiler / GPU code-generation area. Reasonable associate
   editors: people in the MLIR / GPU-compiler / DSL space (avoid anyone we've engaged on the Triton PRs to
   keep review independent — see cover letter note).

## Portal steps
1. Create/login ScholarOne account → "Submit a New Manuscript".
2. Type: **Research paper**. Title: *From Diagnosis to Cure: Decomposing the Tile-SPMD Abstraction Regret on
   Irregular Automata*.
3. Abstract: paste the paper's abstract (plain text; the portal has its own box).
4. Keywords / CCS: the paper already declares CCS concepts + keywords; enter the same in the portal.
5. Authors: enter real name(s)/affiliation/contact here.
6. Cover letter: paste `docs/TACO_COVER_LETTER.md`.
7. Upload `gpufsm_taco.pdf` as the main document. (No separate anonymized+non-anonymized needed if the PDF
   is already anonymous.)
8. Confirm, review the PDF proof ScholarOne generates, submit.

## Honest state going in (for your own confidence)
- The reject-risk items from an internal adversarial review are fixed (decomposition now internally
  consistent; ML claim scoped; cure framed as out-of-band existence proof + honest real-workload payoff).
- The single-GPU weakness the plan flagged is pre-empted: the flagship decomposition is validated on an
  A100 (same 26x; num_warps 3.7x on both). This was expected to be the reviewers' #1 ask.
- Remaining, honestly declared as revision-cycle work (not reject-risk): CIs on the single-config cure
  numbers; the built cure on a real large automaton (currently confounded past 64 states); the built cure
  on the A100. If a reviewer asks, these are additive.
- Expected outcome: Major revision (normal for TACO), with a clear path since the asks are additive.
