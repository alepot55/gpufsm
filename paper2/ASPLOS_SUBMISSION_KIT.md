# ASPLOS 2027, September cycle: submission kit

Everything the HotCRP form will ask for, written down before the form exists. When the site
opens, this file is the only thing that needs to be read.

**Verified 21 August 2026** against the live CFP
(<https://www.asplos-conference.org/asplos2027/cfp/>), the ASPLOS 2027 committee page, and
the April-cycle HotCRP participant list.

| item | value |
|---|---|
| Deadline | **9 September 2026 AoE** = Thu 10 Sep 2026, 13:59 CEST |
| Abstract registration | **none.** The CFP states "No separate abstract submission deadline" |
| Portal | `https://asplos27-sep.hotcrp.com/`. **Now linked from the CFP, but still returns 404.** The link exists; the site is not provisioned |
| Author response | 1–4 December 2026 |
| Notification | 21 December 2026 |
| Paper | `paper2/gpufsm_asplos.pdf`, md5 `3824e23b4d01c7873489a14d1fb006af`, 13 pages (body p1–11, acks + refs p12–13) |
| Change note | `paper2/resubmission_note.pdf`, md5 `e47de7fd1b813e5ede74e5d8956486ac`, 1 page |

---

## a) Abstract, plain text

Paste from `paper2/ASPLOS_ABSTRACT.txt`. **295 words, 1811 characters, pure ASCII**, verified
byte-for-byte against the abstract of the final PDF on 21 Aug 2026.

> Tile abstractions are how GPU kernels are written now, in Triton and increasingly in the vendor's own stack, and on dense tensor work they match hand-written CUDA. On irregular, control-driven kernels, from automata to mixture-of-experts routing, they lose an order of magnitude. Whether that gap is a tuning deficit or a property of the abstraction is a question about language design, and it has not been answered mechanistically.
>
> We answer it. The gap is structural, and one shared loop latch explains it. A tile program's lanes advance together, so a data-dependent loop runs as long as its warp's slowest lane and a dependent load cannot be overlapped across lanes. On an NFA worklist the gap decomposes into a launch-configuration artifact, a redundancy removable by lane-packing, and an irreducible ~2x residual. At matched occupancy, issuing fewer warp-instructions than CUDA and below both roofline ceilings, the tile kernel still spends 15.3x more cycles stalled on a dependent load. The same mechanism predicts, and we confirm, that the residual disappears for a memory-bound DFA once its table spills to DRAM.
>
> The missing primitive is a per-lane loop exit, and we prove TritonGPU cannot express it: scf.condition carries a single i1. We therefore build it below the tile IR, as an MLIR pass in libtriton guarded by a soundness verifier. It gives 2.3-6.7x on control-bound kernels and 1.6-3.8x on an A100 from the same wheel, and ~1.0x on SpMV and MoE, which carry larger stragglers still but a per-iteration gather that dwarfs the tax. Run on that same distribution with a cheap body the pass gives 6.3x, so the scope condition is a divergent trip count over a cheap body, measured rather than argued. A single-predictor straggler model predicts the cost with R^2 = 0.997, and out-of-sample to 1.5%.

### Symbol audit

Every symbol that the LaTeX to text conversion can silently destroy, checked one by one
against the typeset PDF. This is the step that turned `330x-10^4x` into `330x-104x` in the
HPEC abstract.

| in the PDF | in the plain text | status |
|---|---|---|
| `∼2×` | `~2x` | ok |
| `15.3×` | `15.3x` | ok, **digits intact**. Raw `pdftotext` drops the `15` here because the number straddles two text runs; the file was built from the LaTeX source, not from an extraction |
| `2.3–6.7×` | `2.3-6.7x` | ok, en dash to hyphen |
| `1.6–3.8×` | `1.6-3.8x` | ok |
| `≈1.0×` | `~1.0x` | ok |
| `6.3×` | `6.3x` | ok |
| `R² = 0.997` | `R^2 = 0.997` | ok, superscript preserved as caret |
| `1.5%` | `1.5%` | ok |
| apostrophes | straight `'` | ok, no smart quotes |

No exponent, no order of magnitude and no digit is lost. There is no `10⁴`-class symbol in
this abstract, so the specific HPEC failure cannot recur here.

**If HotCRP accepts UTF-8** (it does, but the field may cap length), the typographically
faithful variant is the same text with `~`→`∼`/`≈`, `x`→`×`, `R^2`→`R²`, `-`→`–` in the two
ranges. Either is acceptable. **Read the character limit off the form; do not assume one.**
At 1811 characters this abstract clears the usual 1500-word style caps but would breach a
1500-*character* cap, which some HotCRP instances set. If capped, cut the third paragraph's
SpMV/MoE clause first, and then re-sync `ASPLOS_ABSTRACT.txt` and the PDF so they still match.

---

## b) Title, keywords, CCS

**Title**, exact, copy verbatim:

```
The Lock-Step Tax: Per-Lane Control Flow as the Missing Primitive in Tile GPU DSLs
```

**Keywords**, as in the paper:

```
GPU compilers, tile DSLs, Triton, SIMT, warp-level parallelism, irregular parallelism, per-lane control flow
```

**CCS concepts**, as in the paper (`\ccsdesc` weights in brackets):

- [500] Software and its engineering → Compilers
- [300] Software and its engineering → Parallel programming languages
- [300] Computer systems organization → Single instruction, multiple data

---

## c) Conflicts of interest

The CFP is strict: *"If a submission is found to have an undeclared conflict that causes a
problem or if a paper is found to declare false conflicts in order to abuse or 'game' the
review system, the submission will be summarily rejected."* Conflicts must be declared
against **all** conflicts, not only PC and ERC members. Single-author submission, so there
is no co-author conflict set to merge.

### Declare, certain

| kind | entry |
|---|---|
| Institution, current | **Politecnico di Milano** |
| Institution, past 4 years | any other affiliation held since September 2022. **Author to confirm**; if there is none, Politecnico di Milano is the whole institutional list |

### Checked, nothing to declare

Reference **[24]** (Somaini, Carloni, Agosta, Santambrogio, Conficconi, CGO 2025) is a
Politecnico di Milano group, so it was checked by name against both public rosters:

- ASPLOS 2027 committee page (organizing, steering, vice program chairs, program committee,
  external review committee): **no Politecnico di Milano, no Italian institution at all, and
  none of the five names.**
- April-cycle HotCRP participant list, ~500 rows: **same result.**

So as of 21 Aug 2026 there is nothing to declare on their account. **Institutional conflict
with Politecnico di Milano still gets declared**, because HotCRP matches institutions, not
just the people currently visible.

⚠️ **Re-run this check inside the September HotCRP.** The September cycle may extend the PC,
and the April roster is not authoritative for it. HotCRP's conflict page lists every PC and
ERC member with affiliation; search it for `Polite`, `Milan`, `Italy`, and for the five names
above.

### To be filled in by the author before submitting

These cannot be derived from the repository. Every one is a *declare* under the CFP rules:

- [ ] **PhD or thesis advisor**, and anyone who supervised the thesis, ever. No sunset.
- [ ] **Co-authors of the last 4 years**, on anything, published or not.
- [ ] **Ongoing collaborators**, even without a joint paper.
- [ ] **Internships** held since September 2022, and the employing institution.
- [ ] Any institution the author has **applied to** and is in active discussion with.
- [ ] An institution about to become an employer.

Guidance from the CFP, worth re-reading before ticking boxes: a reviewer working on a
*similar topic* is **not** a conflict and must not be declared as one. Over-declaring on
topic similarity is itself listed as gaming. Over-declaring on **real** relationships is the
safe side; over-declaring on topic is not.

---

## d) Topic and area selection

The CFP: *"Authors should indicate on the submission form the focused topics matching their
work. The selection should be as accurate as possible. These will be used for reviewer
assignments. Incorrect topic selection will lead to difficulty in assigning suitable
reviewers. As mentioned above, lack of suitable reviewers may lead to paper rejection."*

Under a rapid-review round that returns papers for which the committee lacks expertise, a bad
topic pick is not a mismatch, it is a rejection path. The HotCRP topic checkboxes are not
readable while the site is down, so this is a mapping, in priority order, to be applied to
whatever list appears.

**Proposed selection, in order:**

1. **Compilers / programming languages / language implementation**, whichever wording the
   form uses. This is the primary. The contributions are a proof that a production tile IR
   cannot express a primitive, and an MLIR pass in `libtriton` that supplies it. The paper
   lives or dies on compiler reviewers.
2. **Heterogeneous architectures and accelerators.** Present verbatim on the CFP list. It is
   what puts warp-level and SIMT expertise on the panel, which the mechanism section needs.
3. **Experimental methodologies.** Present verbatim on the CFP list. Fits the oracle-gated
   protocol, the held-out validation and the negative controls, and signals to leadership
   that the paper is a measurement paper as well as a compiler paper.
4. **Profiling, debugging, and testing**, only if more than three picks are allowed. The
   Nsight-based mechanism isolation belongs here, but it is method, not contribution.

**Do not select**: machine learning / ML systems, datacenters, security, storage, or
virtualization. MoE routing and attention appear in this paper as *workloads that exhibit the
mechanism*, not as ML contributions, and an ML reviewer would correctly find no ML result.

**Rapid-review framing.** That round scores the paper against four pillars: architecture, OS,
programming languages, and new domains. This paper's pillar is **programming languages**,
with architecture second. The first two pages are already built that way: the claim, Fig. 1,
and the contribution list all land before the page-2 break.

---

## e) Upload checklist, in order

| # | step | artifact | state |
|---|---|---|---|
| 1 | Open the site linked from the CFP "Submission Website" section | `https://asplos27-sep.hotcrp.com/` | ⏳ **404, not yet open** |
| 2 | Sign in or create the account | n/a | author action |
| 3 | Title | section (b) above | ✅ ready |
| 4 | Abstract | `paper2/ASPLOS_ABSTRACT.txt` | ✅ ready, verified |
| 5 | Authors | Alessandro Potenza, Politecnico di Milano. **Form only, never in the PDF** | ✅ ready |
| 6 | Conflicts | section (c) above, plus the six author-only boxes | ⚠️ **incomplete, needs the author** |
| 7 | Topics | section (d) above | ✅ ready |
| 8 | Prior-work declaration, if the form has one | section (f) below | ✅ text ready |
| 9 | Upload the paper | `paper2/gpufsm_asplos.pdf` | ✅ ready |
| 10 | Upload the change note, **required at submission time** | `paper2/resubmission_note.pdf` | ✅ ready |
| 11 | **Press SUBMIT** | n/a | author action |
| 12 | Keep the confirmation email | n/a | author action |

**Never upload `paper2/gpufsm_asplos_named.pdf`.** It carries the author block. It exists for
non-blind use only and regenerates from the anonymous source.

Step 11 is the one that has already been lost once, at PPoPP 2027: the account was created
and the submission never made. A saved draft is not a submission.

---

## f) Prior work: the HPEC overlap

There is an accepted paper by the same author on the same workload, *The Two Faces of
Abstraction Regret* (IEEE HPEC 2026), which reaches IEEE Xplore around late September 2026,
before the 21 December ASPLOS notification. It is cited in this submission as **[21]**, its
result is stated, and the delta is spelled out in the related-work section.

It is not a prior publication in the CFP's sense, and that is measured rather than asserted.
A 10-word shingle comparison over both normalized texts, re-run 21 Aug 2026 on the final
builds:

- **ASPLOS body (p1–11) against the whole HPEC paper: zero shared 10-word runs. 0.00%.**
- Whole paper against whole paper: 4 runs, 56 words, **0.46%** of this submission. All four
  are **bibliography entry titles**: HybridSA, Triton, Hexcute, and the title of [21] itself.

So the shared text is exactly the set of papers both cite, plus the citation to [21]. There is
no shared prose.

### Text for a HotCRP prior-work field

Written to be safe if the field turns out to be reviewer-visible: it names no one.

```
An earlier and shorter paper by an author of this submission, cited here as [21],
was accepted at IEEE HPEC 2026 and will appear in IEEE Xplore before the ASPLOS
notification date. It measures the DSL-versus-CUDA gap on the same automata
workload and stops there. This submission shares no prose with it: a 10-word
shingle comparison finds zero shared runs between this paper's body and that
paper, and the only matches anywhere are four bibliography entry titles. The
delta is stated explicitly in the related-work section. The impossibility result
for TritonGPU, the compiler pass, the second architecture, the out-of-sample
straggler law and the negative controls are new here.
```

If the field is stated to be **chair-only**, the author may name the venue and themselves
directly, which reads better to a chair. Otherwise use the text above unchanged.

### If no such field exists

Do **not** email anyone without approval. Draft, held for the author's decision, addressed to
the ASPLOS 2027 program chairs (Abhishek Bhattacharjee, Ada Gavrilovska, Steve Blackburn):

```
Subject: ASPLOS'27 September, submission #NNN: disclosing an overlapping accepted paper

Dear program chairs,

I am disclosing, ahead of any question, an accepted paper of mine that overlaps
this submission in workload. "The Two Faces of Abstraction Regret" was accepted at
IEEE HPEC 2026 and will appear in IEEE Xplore in late September 2026, before the
ASPLOS notification. My submission cites it as reference [21] and states the delta
in its related-work section.

I do not believe it is a prior publication in the sense of the CFP, and I would
rather you judge that than assume it. The two papers share no prose: a 10-word
shingle comparison over the normalized texts finds zero shared runs between the
body of the submission and the HPEC paper, and the only matches anywhere are four
bibliography entry titles that both papers cite. The HPEC paper measures the gap
on one architecture and stops. The submission adds a proof that the missing
primitive is inexpressible in TritonGPU, a soundness-verified MLIR pass that
supplies it, a second architecture, a predictive law tested out of sample, and
negative controls.

I am happy to provide the HPEC paper if that would help.

Sincerely,
Alessandro Potenza
Politecnico di Milano
```

---

## Open items

- ⚠️ **Conflicts, section (c)**: six boxes only the author can fill.
- ⚠️ **Re-run the Polimi conflict sweep inside the September HotCRP** once it opens.
- ⏳ **The portal.** `asplos27-sep.hotcrp.com` is linked from the CFP and returns 404. The
  signal to watch is that URL returning anything other than 404, not the CFP text, which has
  already changed.
