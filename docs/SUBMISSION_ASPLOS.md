# ASPLOS 2027 (September cycle) — submission packet

**Status (18 Aug 2026, re-verified against the live CFP):** the paper and every
required attachment are ready. **The submission site still does not exist**: the CFP
page contains exactly one HotCRP link, the closed April cycle. See "Blocker" below.

A sentinel now watches for it: `~/.cache/watch_asplos.sh` fetches the CFP every three
hours and shouts on any HotCRP link it has not seen before, plus on the CFP naming a
September submission site. Validated in both directions (silent on today's page, fires
on the target text) so it cannot be a blind guard. Log: `~/.cache/watch_asplos.log`.
It dies on reboot, so **re-run it after a restart**, and do not treat its silence as
proof on a machine that has been off.

Paper conforms: 9 pages of text and figures against a limit of 11.

## Venue facts (read directly from <https://www.asplos-conference.org/asplos2027/cfp/>)

| item | value |
|---|---|
| Paper deadline | **9 September 2026, AoE** = **Thu 10 Sep 2026, 13:59 CEST** |
| Abstract deadline | none (single step, abstract goes in with the paper) |
| Author response | 1-4 December 2026 |
| Notification | 21 December 2026 |
| Conference | 11-15 April 2027, Heraklion, Crete, Greece |
| Page limit | 11 pages of text and figures. References, **acknowledgments** and appendices excluded |
| Format | `\documentclass[sigplan,anonymous,review,nonacm]{acmart}`, 10pt body (not 9pt), captions 9pt, nothing below 8pt |
| Review | Double-blind, with a rapid-review round that reads only the **first two pages** |
| Submissions per author | max 4 per cycle |

## The four questions that needed a browser

1. **Exact deadline time.** All CFP dates are AoE. End of 9 Sep AoE is
   **10 Sep 2026 at 13:59 CEST**. Real usable slack, but do not plan around it.
2. **September-cycle HotCRP URL.** **Not created yet.** As of 14 Aug 2026 the CFP's
   "Submission Website" section links only the April cycle
   (<https://asplos27-apr.hotcrp.com/>, now closed). `asplos27-sep`, `asplos27-sept`
   and `asplos27-fall` all return "no such conference", and HotCRP's own conference
   index lists no September '27 site. Convention across ASPLOS '26 (`asplos26-spring`,
   `asplos26-summer`) and '27 (`asplos27-apr`) makes `asplos27-sep.hotcrp.com` the
   likely name, but **take the URL from the CFP page when it appears; do not guess.**
3. **ORCID.** Not required at submission. Required of authors of **accepted** papers
   to complete ACM publishing. Worth creating one in the meantime.
4. **Generative-AI disclosure.** **Mandatory.** The CFP binds submissions to the ACM
   authorship policy, which requires that use of generative AI be *fully disclosed*
   in the work; the acknowledgments section is the designated place, must sit
   immediately before the references, and does not count against the page limit.
   → **Done:** `\section*{Acknowledgments}` added to both builds, disclosing the
   agentic coding assistant that wrote most of the code, ran the measurements,
   generated the figures and drafted the text, and stating what remains ours.
   (Declared with `\section*` and not the `acks` environment on purpose: acmart's
   `anonymous` option suppresses `acks`, which would have silently dropped a
   disclosure that reviewers must see.)

Also settled while there: **artifact evaluation is post-acceptance**, voluntary, and
does not affect the decision (ASPLOS'27 AEC Summer, artifact deadline 7 Jan 2027).
Nothing to opt into at submission time.

## One requirement we were missing: the resubmission note

The CFP: *"Authors of resubmitted work from ASPLOS or other venues must describe in a
separate note, to be uploaded to the submission site at submission time, the changes
since the previous submission(s)."* This work went to ACM TACO on 5 Jul 2026 and was
returned by the Editor-in-Chief on 7 Jul 2026 (editorial decision, no external
review: novelty/quality bar, results "too premature"). So the note is required.

→ **Done:** `paper2/resubmission_note.pdf` (1 page, anonymous). It states the TACO
decision plainly, then the four substantive changes made since: the cure rebuilt from
a pinned recipe and reproduced on an A100; non-reproducible numbers re-measured and
replaced; the straggler law refit and validated out of sample; and negative results
added (cure-rejection witness, pointer-chase control at 1.00x, the 0.64x sign-flip).

Note that ASPLOS's own resubmission bar does not apply to us: only papers rejected
from an immediately preceding **ASPLOS** cycle are barred, and this work has never
been submitted to ASPLOS.

## Compliance check (against the rebuilt PDF, 21 Aug 2026)

- **9 pages of text and figures**; acknowledgments and references both start on p.10.
  Limit is 11 excluding those. Comfortably inside.
- LaTeX: **0 errors, 0 overfull boxes, 0 undefined references.**
- Fonts: body 10pt, tables and the IR listing `\footnotesize` (8pt floor), captions 9pt.
  No `\resizebox` and no `\vspace` squeezing.
- **Figures carry no hidden downscale.** They are emitted by `paper2/figures.py` as vector
  PDFs at their final printed size (3.33 in for a column, 7.0 in for a `figure*`) and
  included at scale 1.0, so an 8 pt label in the source is 8 pt on the page. The earlier
  set was generated at 6 in wide and included with `width=\columnwidth`, i.e. scaled to
  0.55x, which put every axis label near 5 pt. That was below the CFP floor and this
  compliance check had missed it, because `\resizebox` was absent but the shrink came
  from the `width=` instead.
- The IR listing (Fig. 4) is a trimmed excerpt with all `loc(...)` source locations
  stripped: the raw capture embeds an absolute path containing the author's username.
- Anonymity: no author name, affiliation, repo name or personal URL in the text; PDF
  metadata carries no Author field. The one anonymous self-citation is to our paper 1,
  which is correct under double-blind.
- Every figure carries the ACM-required `\Description`, all six of them, and the build
  emits zero "possible image without description" warnings. Check this by grepping the log
  rather than the source: the restructure dropped all eight of the old ones silently and
  this line claimed otherwise for a while.
- Zero em dashes.
- References: 29 entries, full author names throughout (no "et al."), DOIs or
  resolvable URLs on all but the anonymous self-citation, citations hyperlinked. bibtex
  emits warnings for missing page numbers and publishers on several entries; cosmetic for
  review, worth closing before camera-ready.

## Files to upload

- `paper2/gpufsm_asplos.pdf` — the anonymous paper.
- `paper2/resubmission_note.pdf` — the required change note.
- `paper2/ASPLOS_ABSTRACT.txt` — abstract as plain ASCII, to paste in the form.
- `paper2/gpufsm_asplos_named.{tex,pdf}` — author block restored, for non-blind use.
  **Do not upload this one.**

## Blocker

The September-cycle HotCRP site is not open. Nothing else stands in the way. When it
opens (historically a few weeks before the deadline):

1. Open the site linked from the CFP's "Submission Website" section, create an
   account or sign in.
2. Title: `The Lock-Step Tax: Per-Lane Control Flow as the Missing Primitive in Tile GPU DSLs`
3. Abstract: paste `paper2/ASPLOS_ABSTRACT.txt`.
4. Authors: Alessandro Potenza, Politecnico di Milano (in the form only, never in the PDF).
5. Topics: select accurately. Bad topic selection means bad reviewer matching, and
   under the rapid-review round that is a direct path to an early return. This work
   is compilers / programming languages plus GPU architecture.
6. Conflicts: register all COIs (institutions in the last 4 years, advisors,
   collaborators in the last 4 years).
7. Upload the paper PDF and the resubmission note.
8. **Press Submit.** A saved draft does not count. This is exactly how PPoPP 2027 was
   lost: account created 8 Jul, submission never made.
9. Keep the confirmation email.

Calendar reminders are set for 1 Sep (submit) and 10 Sep (hard deadline).
