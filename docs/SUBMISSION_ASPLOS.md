# ASPLOS 2027 (September cycle) — submission packet

Everything is prepared. The only remaining work is a **manual action on HotCRP**
(the submission site is unreachable from the agent sandbox: the egress proxy
returns a 403 policy denial for `hotcrp.com` and `asplos-conference.org`).

## Venue facts

| item | value |
|---|---|
| Paper deadline | **9 September 2026, AoE** |
| Abstract deadline | none (single deadline, paper submitted in one step) |
| Notification | 21 December 2026 |
| Conference | 11-15 April 2027, Heraklion, Crete, Greece |
| Page limit | 11 pages of text and figures, **references excluded** |
| Format | `\documentclass[sigplan,anonymous,review,nonacm]{acmart}`, 10pt body (not 9pt), no font below 8pt |
| Review | Double-blind, with a rapid-review round that screens only the **first two pages** |

## Our compliance check (verified against the built PDF)

- **9 pages of text and figures**, references begin on p.10, against an 11-page
  limit that excludes references. Comfortably inside.
- **LaTeX: 0 errors, 0 overfull boxes, 0 undefined references, 0 class warnings.**
- **Fonts:** body 10pt; tables `\footnotesize` = 8pt (the floor, not below it);
  captions `\small` = 9pt. No `\resizebox` shrinking anywhere, so nothing is
  scaled under the limit.
- **Anonymity:** no author name, affiliation, repository name, or personal URL in
  the rendered text; PDF metadata carries no Author field. The only anonymous
  self-reference is to our own paper 1, which is correct under double-blind.
- **Accessibility:** every figure carries the ACM-required `\Description`.
- **Typography:** zero em dashes.

## Files

- `paper2/gpufsm_asplos.pdf` - the anonymous PDF to upload.
- `paper2/gpufsm_asplos.tex` - its source.
- `paper2/gpufsm_asplos_named.{tex,pdf}` - the same paper with the author block
  restored, for non-blind use. **Do not upload this one.**
- `paper2/ASPLOS_ABSTRACT.txt` - the abstract as plain ASCII, ready to paste into
  the HotCRP form.

## Steps on HotCRP

1. Open the ASPLOS 2027 September-cycle HotCRP site (linked from
   <https://www.asplos-conference.org/asplos2027/cfp/>) and sign in or create an
   account.
2. Title: `From Diagnosis to Cure: Decomposing the Tile-SPMD Abstraction Regret on Irregular Automata`
3. Abstract: paste `paper2/ASPLOS_ABSTRACT.txt`.
4. Authors: Alessandro Potenza, Politecnico di Milano. HotCRP keeps the author
   list hidden from reviewers; it must still be filled in.
5. Upload `paper2/gpufsm_asplos.pdf`.
6. Answer the topic/conflict questions, then **submit** (a saved-but-not-submitted
   draft does not count - this is exactly what went wrong with PPoPP 2027, where
   the account was created on 8 July and no submission ever followed).
7. Keep the confirmation email.

## Before submitting, confirm at the source

The CFP page is unreachable from the sandbox, so these were established from
search-indexed copies rather than a direct read. Worth one glance in a browser:

- the exact deadline time and the September-cycle HotCRP URL;
- whether an ORCID or a generative-AI-use disclosure is required at submission;
- whether the artifact-evaluation opt-in happens now or after acceptance.
