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
| md5, anonymous | `e08c97e7d13dd61c5d629493188b9f56` |
| md5, named (never uploaded) | `19fa5ee4e042816e6614accb6588ab2b` |
| pages | **12. Body p1 to p11 exactly, references p12** |

Note that the figure PDFs are not byte-reproducible: matplotlib stamps a creation date, so
every regeneration changes the md5 of the paper even when nothing visible moves. Compare
figures by their page size and by eye, not by checksum.

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

## Two questions that were open, now closed

Both were new at CGO and neither existed at ASPLOS.

### 1. The acknowledgments are gone from this submission

ASPLOS carved acknowledgments out of its page limit in so many words. **CGO carves out only
the bibliography**: *"Your submission is limited to 11 pages of text, excluding bibliography"*
and *"There is no page limit for references"*. Under that wording the disclosure section was
text, and the text ran onto p12, which is 12 pages against a limit of 11.

**CGO also does not ask for the disclosure.** Its CFP records that ACM's policy changed:
*"ACM no longer requires authors to disclose the use of AI in preparing a submission"*, with
the emphasis moved onto the authors' responsibility for correctness.

So the section was removed for this venue, and the count is now unambiguous:

| | pages | text ends | references |
|---|---|---|---|
| with acknowledgments | 13 | p12, part of a column | p12 to p13 |
| **without, as shipped** | **12** | **p11 exactly** | **p12, one page** |

**Restore it at camera-ready if the paper is accepted**, so the disclosure stays aligned with
the HPEC paper covering the same work. A comment in `gpufsm_cgo27.tex` says so at the point
where the section used to be, and the text to restore is intact in `gpufsm_asplos.tex`.

Side effect worth recording: removing the section also cleared the single overfull vbox the
build had been carrying. The build is now **0 errors, 0 overfull, 0 undefined references**.

### 2. The figures are now readable on a black-and-white printer

CGO, verbatim: *"Your submission must be formatted for black-and-white printers and not color
printers. This is especially true for plots and graphs in the paper."*

The old palette failed that. Fig. 6 keyed its eight bars to a dominant mechanism by hue
alone, with a legend of plain colour patches and no second channel, and five of its seven
categories landed within a few points of the same grey: teal 50.5, grey 53.8, green 54.8,
orange 55.5, with the closest pair **0.7 points apart out of 100**. Printed, five of the
eight bars were one shade, and the figure that carries the generality claim conveyed nothing.

What changed, in `paper2/figures.py` only:

- A palette built on a **luminance ladder** rather than a hue wheel: 25.9, 38.6, 48.0, 58.1,
  66.4, 74.5, 87.3. Closest pair now **5.1 points apart**, up from 0.7.
- **Hatches on every bar**, in all three bar figures, with the legend patches carrying the
  same hatch so the mapping is readable: dots, forward and back diagonals, cross, grid,
  circles, and plain fill.
- Fig. 6's y-axis opened from `max*1.18` to `max*1.34`, and one legend label shortened from
  "dense per-step work: tile wins" to "dense per-step work", because the old label ran onto
  the rejection-sampling bar and the two ink patterns collided. The dropped words were
  redundant: the arrow annotation already says "the tile wins".
- Fig. 6's caption and its `\Description` now say the categories are keyed by fill **and**
  hatch, which tells a reader printing in grey where to look.

**No measurement was touched.** Figures still regenerate from the versioned CSVs through the
one script, and every figure comes out at **exactly its previous page size to three decimal
places**, which matters because the body has only about one column of slack on p11:

```
fig_anatomy_and_cure  224.171 x 265.87     fig_regret_law     488.484 x 134.364
fig_dfa_crossover     226.478 x 125.116    fig_straggler_law  224.171 x 121.87
fig_mechanism         488.741 x 127.482
```

Verified by converting the built PDF to true greyscale with Ghostscript
(`-sColorConversionStrategy=Gray`) and reading the rendered pages, not by trusting the
palette arithmetic. All eight bars in Fig. 6 separate; the teaser on p2 separates; the three
panels of Fig. 2 separate.

⚠️ **The frozen ASPLOS PDF now predates these figures.** `gpufsm_asplos.pdf` is deliberately
not rebuilt and stays byte-identical, so it no longer matches the current
`paper2/figures/*.pdf`. If the ASPLOS April 2027 fallback is ever taken up, rebuild it: it
will pick up the greyscale-safe figures, which is an improvement, and the caption wording
should be updated to match.

## Files, and the ones that must never be uploaded

- `paper2/gpufsm_cgo27.pdf` is the submission.
- `paper2/gpufsm_cgo27_named.pdf` carries the author block.
- `paper2/gpufsm_asplos.pdf` is the other venue's build, with the wrong class options.
- `paper2/gpufsm_cgo.pdf` is a dead build from 15 August, superseded, and its name differs
  from the live one by two characters. It is the most likely wrong file to grab.
