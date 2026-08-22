# CGO 2027 round 2: compliance state of `gpufsm_cgo27.pdf`

**Target:** CGO 2027, round 2. Deadline **Thu 10 Sep 2026, 23:59:59 AoE** = Fri 11 Sep,
13:59:59 CEST. Site **open**: <https://cgo27.hotcrp.com/>, class "new R2 submissions".

Form-filling material is in `paper2/CGO_SUBMISSION_KIT.md`. This file is the compliance
record for the PDF. The ASPLOS packet is untouched and remains the April 2027 fallback;
see `docs/SUBMISSION_ASPLOS.md`.

## The build

`\documentclass[sigplan,screen,review,anonymous]{acmart}`, exactly the line the CFP quotes
twice. `nonacm` is gone, which brings the ACM copyright block back to page 1 at a cost of
about twelve lines; `screen` selects hyperref's colour-link variant. The explicit `10pt`
ASPLOS demanded is dropped because acmart's `sigplan` format is 10pt by construction, and
that is measured, not assumed: both builds close the body on line 1210.

| | |
|---|---|
| md5, anonymous | `16b36403b7d0b64fcb8d9426d1379ebf` |
| md5, named (never uploaded) | `89a10ca224baffffb5bed6ac2bb73df7` |
| pages | 13. Body p1 to p11, acknowledgments p12, references p12 to p13 |

## Passed

- **Body within 11 pages.** The Conclusion's last line sits on p11 and the acknowledgments
  open p12. Headroom is now about one column, down from about one and a half at ASPLOS: the
  body ends in p11's left column at y=709 against a text block that bottoms out near 715,
  with p11's right column free. **Any addition to the body must be re-measured.**
- **Anonymity.** Text carries only the three protected occurrences of "Potenza", which are
  reference [21] and its two in-text citations, in third person. That is the form CGO
  prescribes verbatim for self-citation under double-blind. XMP `dc:creator` is
  "Anonymous Author(s)". The binary carries no name, institution, path or repository URL;
  the only residue is `dc:source = gpufsm_cgo27.tex`.
- **Geometry.** Identical text block on all 13 pages: xmin 53.6, xmax 559.7, ymin 49.6,
  ymax 730.3. No `\vspace` and no `\resizebox`.
- **Fonts.** 26 faces, all embedded and subsetted, zero Type 3.
- **Raster.** Zero raster images. Every figure is vector.
- **Page numbers and line numbers.** Both present, as CGO requires.
- **Build.** 0 errors, 0 undefined references, 1 overfull box (the same `\output` vbox the
  ASPLOS build had), 18 cosmetic underfull.
- **US Letter**, 612x792.

## Two open questions, both new at CGO

Neither existed at ASPLOS. Both are author decisions, deliberately not taken here.

### 1. Do the acknowledgments count toward the 11 pages?

ASPLOS carved them out explicitly: *"the acknowledgment section (used only to acknowledge use
of Generative AI as per ACM policy above), the bibliographic references section, and the
appendices ... are not included in the page limit."*

**CGO carves out only the bibliography**: *"Your submission is limited to 11 pages of text,
excluding bibliography"* and *"There is no page limit for references"*. Acknowledgments are
not mentioned. Under a strict reading they are text, and this submission's text would then
run to p12, which is 12 pages.

Compounding it: **CGO does not require an AI disclosure at all.** The CFP states that ACM's
authorship policy was updated and that *"ACM no longer requires authors to disclose the use of
AI in preparing a submission"*, placing the emphasis instead on the authors' responsibility
for correctness. So the section that creates the risk is not one CGO asks for.

Measured, both ways:

| | pages | text ends | references |
|---|---|---|---|
| with acknowledgments (current build) | 13 | p12, part of the left column | p12 right column to p13 |
| without acknowledgments | **12** | **p11, exactly** | **p12 only, one page** |

Dropping the section makes the count unambiguous at exactly 11 pages of text. Keeping it
is a defensible reading of an ambiguous rule, and it preserves a disclosure the author chose
to make deliberately at both venues. **Not decided here.**

### 2. Do the figures survive a black-and-white printer?

CGO, verbatim: *"Your submission must be formatted for black-and-white printers and not color
printers. This is especially true for plots and graphs in the paper."*

All five figures carry non-gray colour operators. The exposure is not uniform:

Fig. 5, `fig_regret_law`, encodes **dominant mechanism by colour alone**, with a legend of
colour patches and no second channel: no hatching, no marker shape, no label. Its seven
categories collapse to these greyscale luminances:

| colour | role | luminance |
|---|---|---|
| `#c0392b` red | issue starvation | 33.2 |
| `#8e44ad` purple | baseline plus divergence | 35.8 |
| `#2980b9` blue | masked-lane waste, gather-diluted | 44.6 |
| `#16a085` teal | dense per-step work, tile wins | 50.5 |
| `#7f8c8d` gray | tile-lowering baseline | 53.8 |
| `#27ae60` green | no control divergence | 54.8 |
| `#e67e22` orange | masked-lane waste | 55.5 |

The three at the top are separated by **0.7 and 1.7 points out of 100**. Printed grey, teal,
gray, green and orange are one shade. A reader on paper cannot recover the mechanism, which
is the finding that figure exists to convey.

The other four figures use colour to decorate a distinction that is already carried by
position, axis or annotation, so they degrade rather than fail.

The fix, if wanted, is confined to `paper2/figures.py`: add hatch patterns to the bars in
Fig. 5 and widen the luminance spread of the palette. Figures regenerate from the versioned
CSVs, so no measurement is touched. **Not done here**, because it changes the paper.

## Files, and the ones that must never be uploaded

- `paper2/gpufsm_cgo27.pdf` is the submission.
- `paper2/gpufsm_cgo27_named.pdf` carries the author block.
- `paper2/gpufsm_asplos.pdf` is the other venue's build, with the wrong class options.
- `paper2/gpufsm_cgo.pdf` is a dead build from 15 August, superseded, and its name differs
  from the live one by two characters. It is the most likely wrong file to grab.
