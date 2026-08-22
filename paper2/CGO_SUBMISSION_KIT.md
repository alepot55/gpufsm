# CGO 2027 round 2: submission kit

Everything the HotCRP form asks for. Unlike the ASPLOS packet, **the site is open now**, so
this can be executed today rather than waited on.

**Researched 22 August 2026** against the CGO 2027 CFP, the artifact-evaluation call, all five
published committee rosters, and the live HotCRP instance. The ASPLOS packet
(`ASPLOS_SUBMISSION_KIT.md`, `gpufsm_asplos.pdf`, `resubmission_note.pdf`) is untouched and
stays the fallback for the April 2027 cycle.

| item | value |
|---|---|
| Deadline | **Thu 10 September 2026, 23:59:59 AoE** = **Fri 11 Sep, 13:59:59 CEST** |
| Abstract registration | **none.** One shot, paper and abstract together |
| Portal | **<https://cgo27.hotcrp.com/> is OPEN** (`sub:{"open":true}`) |
| Submission class | **"new R2 submissions"**. Not "new R1-revision submissions" |
| Re-upload | allowed: "You can submit multiple times before the deadline. Only the last submission will be reviewed" |
| Rebuttal | ⚠️ **contested**, see below |
| Notification | ⚠️ **contested**, see below |
| Conference | 20 to 24 March 2027, Salt Lake City, Utah, USA |
| Paper | `paper2/gpufsm_cgo27.pdf`, md5 `e08c97e7d13dd61c5d629493188b9f56`, **12 pages**, body p1 to p11 |

### Two dates CGO contradicts itself on

The CFP prose and the Important Dates widget disagree, on the same page, by one week.

| | CFP prose | Important Dates widget, /dates, home page |
|---|---|---|
| R2 rebuttal | 20 to 22 Oct 2026 | **27 to 29 Oct 2026** |
| R2 notification | 2 Nov 2026 | **9 Nov 2026** |

Three of four surfaces say 9 November. No time of day is published for either. Only the
program co-chair can settle it: **Zheng Wang, University of Leeds, z.wang5@leeds.ac.uk**.
Do not plan around either until asked. This matters only after submitting.

The submission deadline itself is not contested: HotCRP carries the machine timestamp
`data-ts="1789127999"`, which decodes to 2026-09-11 11:59:59 UTC, exactly 23:59:59 AoE on
10 September.

---

## a) Abstract, plain text

Unchanged from the ASPLOS packet: the body text of the paper did not change, only the
document class. Paste from `paper2/ASPLOS_ABSTRACT.txt`. **295 words, 1811 characters, pure
ASCII**, re-verified byte for byte against the abstract of `gpufsm_cgo27.pdf`.

> Tile abstractions are how GPU kernels are written now, in Triton and increasingly in the vendor's own stack, and on dense tensor work they match hand-written CUDA. On irregular, control-driven kernels, from automata to mixture-of-experts routing, they lose an order of magnitude. Whether that gap is a tuning deficit or a property of the abstraction is a question about language design, and it has not been answered mechanistically.
>
> We answer it. The gap is structural, and one shared loop latch explains it. A tile program's lanes advance together, so a data-dependent loop runs as long as its warp's slowest lane and a dependent load cannot be overlapped across lanes. On an NFA worklist the gap decomposes into a launch-configuration artifact, a redundancy removable by lane-packing, and an irreducible ~2x residual. At matched occupancy, issuing fewer warp-instructions than CUDA and below both roofline ceilings, the tile kernel still spends 15.3x more cycles stalled on a dependent load. The same mechanism predicts, and we confirm, that the residual disappears for a memory-bound DFA once its table spills to DRAM.
>
> The missing primitive is a per-lane loop exit, and we prove TritonGPU cannot express it: scf.condition carries a single i1. We therefore build it below the tile IR, as an MLIR pass in libtriton guarded by a soundness verifier. It gives 2.3-6.7x on control-bound kernels and 1.6-3.8x on an A100 from the same wheel, and ~1.0x on SpMV and MoE, which carry larger stragglers still but a per-iteration gather that dwarfs the tax. Run on that same distribution with a cheap body the pass gives 6.3x, so the scope condition is a divergent trip count over a cheap body, measured rather than argued. A single-predictor straggler model predicts the cost with R^2 = 0.997, and out-of-sample to 1.5%.

### Symbol audit

The step that turned `330x-10^4x` into `330x-104x` in the HPEC abstract. Checked one by one
against the typeset PDF.

| in the PDF | in the plain text | status |
|---|---|---|
| `∼2×` | `~2x` | ok |
| `15.3×` | `15.3x` | ok, **digits intact**. Raw `pdftotext` drops the `15` here because the number straddles two text runs. The file is built from the LaTeX source, never from an extraction |
| `2.3–6.7×` | `2.3-6.7x` | ok, en dash to hyphen |
| `1.6–3.8×` | `1.6-3.8x` | ok |
| `≈1.0×` | `~1.0x` | ok |
| `6.3×` | `6.3x` | ok |
| `R² = 0.997` | `R^2 = 0.997` | ok, superscript as caret |
| `1.5%` | `1.5%` | ok |
| apostrophes | straight `'` | ok, no smart quotes |

No exponent, order of magnitude or digit is lost. **Read any character limit off the form**;
do not assume one. At 1811 characters this would breach a 1500-character cap if HotCRP sets
one. If capped, cut the SpMV/MoE clause in the third paragraph first, then re-sync the txt
file and the PDF so they still match.

---

## b) Title, keywords, CCS

**Title**, verbatim. Note that CGO asks tools papers to prepend `Tool:` and practical papers
`Practical:`. **This is a standard research paper: no prefix.**

```
The Lock-Step Tax: Per-Lane Control Flow as the Missing Primitive in Tile GPU DSLs
```

**Keywords:**

```
GPU compilers, tile DSLs, Triton, SIMT, warp-level parallelism, irregular parallelism, per-lane control flow
```

**CCS concepts**, as typeset in the paper:

- [500] Software and its engineering → Compilers
- [300] Software and its engineering → Parallel programming languages
- [300] Computer systems organization → Single instruction, multiple data

---

## c) Conflicts of interest

⚠️ **Unlike ASPLOS, one conflict fires, and it fires twice over.**

### DECLARE: Davide Conficconi

**Davide Conficconi, Politecnico di Milano, Italy, sits on the CGO 2027 Program Committee.**

He is a conflict on **three independent grounds**, any one of which would be sufficient
on its own:

1. **Same institution.** Politecnico di Milano, the author's own.
2. **Supervision within the last four years.** He was the tutor of the project this line of
   work grew out of. The relationship is closed, and he knows nothing of these publications,
   but advisor and supervisor conflicts do not sunset on the CFP's terms and this one falls
   inside the window regardless.
3. **Cited co-author.** Last author of reference **[24]**, Somaini, Carloni, Agosta,
   Santambrogio, Conficconi, *Combining MLIR Dialects with Domain-Specific Architecture for
   Efficient Regular Expression Matching*, CGO 2025.

### Declare him BY NAME, not only as an institution

Tick him individually on the conflicts page, in addition to declaring Politecnico di Milano
as an institution. HotCRP matches institutional conflicts by string against affiliation
fields, and that match misses routinely: an abbreviation, a second affiliation, an English
rendering, a stale profile, any of these and the institutional rule silently fails to fire.
The named tick does not depend on any of that.

**What declaring achieves**, and it is the whole point: it removes him from bidding, from
assignment, from access to the PDF, from the reviews, and from the discussion of this paper.
That is the mechanism working as designed, not a cost.

### No contact with him, before or after

Do not write to him about this submission at any point in the cycle. No rule requires it, he
is not expecting it, and CGO's double-blind rules cut directly against approaching a PC
member about a paper under review. Declaring the conflict in HotCRP is the complete and
correct action; a message on top of it would be the thing that causes a problem, not the
thing that prevents one.

Note what the declaration costs, so it is not a surprise: he works on MLIR dialects for regex
and automata acceleration, which plausibly makes him the single best-matched reviewer on the
committee. Declaring removes that. It is still not optional, on any of the three grounds.

### The full sweep, and how far it reaches

Every published CGO 2027 roster was fetched and read, **99 people in total**:

| roster | members read | Politecnico or Italy | any of the five names in [24] |
|---|---|---|---|
| Program Committee | **80** | **Davide Conficconi** | **Conficconi** |
| External Review Committee | 6 | none | none |
| Artifact Evaluation Committee | 2 | none | none |
| Organizing Committee | 5 | none | none |
| Steering Committee | 6 | none | none |

Names swept for: Andrea Somaini, Filippo Carloni, Giovanni Agosta, Marco D. Santambrogio,
Davide Conficconi. Institution keys swept for: Politecnico, Milan/Milano, Italy, plus Torino,
Bologna, Pisa, Bari, Padova, Napoli, Roma, Sapienza, Trento, Genova, Catania, Salerno.

⚠️ **The ERC and the AEC are obviously incomplete.** Six people and two people are not a
finished external review committee and artifact committee for a conference this size; they
are the chairs plus early recruits. **Re-run the sweep inside HotCRP's conflict page before
submitting**, where every PC and ERC member is listed with affiliation. Search for `Politec`,
`Milan`, `Ital`, and the five names. The absence of a name from the table above proves
nothing about a roster that is still being filled.

### Author-only boxes, unchanged from the ASPLOS kit

Single-author submission, so there is no co-author conflict set to merge. These cannot be
derived from the repository:

- [ ] **PhD or thesis advisor**, and anyone who ever supervised the thesis. No sunset.
- [ ] **Co-authors of the last 4 years**, published or not.
- [ ] **Ongoing collaborators**, even with no joint paper.
- [ ] **Internships** since September 2022, and the employer.
- [ ] Any institution applied to and in active discussion with.
- [ ] An institution about to become an employer.

Same rule as ASPLOS, and it is worth restating: **a reviewer working on a similar topic is
not a conflict.** Declaring topic similarity is gaming. Declaring real relationships is not.

---

## d) Topics

CGO publishes 17 topics. The form asks for "relevant topics" and uses them for reviewer
assignment. Proposed selection, in priority order:

1. **Compiler abstraction and intermediate representations.** The primary. The paper's
   central result is that a production tile IR cannot express a primitive, proved on
   `scf.condition`, plus an MLIR pass in `libtriton` that supplies it below that IR.
2. **Code generation and optimizations for heterogeneous or specialized targets, TPUs, GPUs,
   SoCs, CGRA, and quantum computers.** Everything measured here is a GPU code-generation
   result on four architectures.
3. **Optimization and code generation for emerging programming models, platforms, and
   domain-specific languages.** Tile DSLs are precisely an emerging programming model, and
   the paper is about what one of them forecloses.
4. **Compiler support for vectorization, thread extraction, task scheduling, speculation,
   transactions, memory management, data distribution, and synchronization.** Per-lane
   retirement is thread extraction under a different name.
5. **Program characterization methods**, if more than four are allowed. The straggler law
   and its out-of-sample validation are exactly a characterization method.

**Do not select** machine-learning-based code generation. MoE routing and ragged attention
appear as workloads that exhibit the mechanism, not as ML contributions, and an ML reviewer
would correctly find no ML result.

CGO's rapid-rejection risk is not a screening round as at ASPLOS, but bad topics still mean
bad assignment. The paper's home is compilers and IR design.

---

## e) Upload checklist, in order

| # | step | artifact | state |
|---|---|---|---|
| 1 | Open <https://cgo27.hotcrp.com/>, sign in or create an account | | ✅ site open |
| 2 | Choose the class **"new R2 submissions"** | | ⚠️ the easy mistake: not "R1-revision" |
| 3 | Title, no `Tool:` or `Practical:` prefix | section (b) | ✅ ready |
| 4 | Abstract | `paper2/ASPLOS_ABSTRACT.txt` | ✅ verified |
| 5 | Authors | Alessandro Potenza, Politecnico di Milano. **Form only, never the PDF** | ✅ ready |
| 6 | Conflicts | **Davide Conficconi, ticked by name**, plus Politecnico di Milano as an institution, plus the six author-only boxes, plus a fresh in-HotCRP sweep | ⚠️ needs the author |
| 7 | Topics | section (d) | ✅ ready |
| 8 | **Artifact evaluation interest: TICK YES** | section (f) | ✅ decided, see below |
| 9 | Upload the paper | `paper2/gpufsm_cgo27.pdf`, md5 `e08c97e7d13dd61c5d629493188b9f56` | ✅ ready |
| 10 | Supplementary material | none. See the note below | n/a |
| 11 | **Press SUBMIT** | | author action |
| 12 | Keep the confirmation email | | author action |

**Never upload `paper2/gpufsm_cgo27_named.pdf`.** It carries the author block.
**Never upload `paper2/gpufsm_asplos.pdf`.** Wrong venue, wrong class options.
**Never upload `paper2/gpufsm_cgo.pdf`.** That is a dead build from 15 August, superseded, and
its name is one character away from the live one.

On supplementary material: CGO says appendices go in as supplementary material rather than in
the main PDF, that it must be anonymized, and that reviewers are not required to read it. This
paper is self-contained and has no appendix, so nothing to attach. The reproducibility section
already points at the versioned CSVs and the pinned recipe.

Step 11 is the one already lost once, at PPoPP 2027: account created, submission never made.
A saved draft is not a submission. CGO allows re-uploading before the deadline, so **submit
early and refine**, rather than holding a perfect PDF until the last hour.

---

## f) Artifact evaluation: tick yes

**Interest is declared on the submission form, not later**, verbatim: *"To ease the
organization of the AE committee, we kindly ask authors to indicate at the time they submit
the paper, whether they are interested in submitting an artifact."* Voluntary for research
papers, mandatory only for tools papers, and it *"does not influence the final decision
regarding paper acceptance"*. There is no downside to ticking it and the artifact is pinned
and freshly audited, so tick it.

### What CGO awards

Three badge families, up to one of each per paper:

| badge | what it takes |
|---|---|
| **Artifacts Available** | deposit in a *qualified archival repository*. **Zenodo, figshare, Dryad.** CGO states explicitly that "Personal webpages, GitHub repositories or alike are **not** sufficient as it can be changed after the submission deadline". No audit needed, awarded by the publisher on a link. ⚠️ **Post-acceptance only, see below** |
| **Artifacts Evaluated, Functional or Reusable** | independent audit. Functional means documented, consistent, complete, exercisable, with evidence of verification. Reusable means significantly beyond that, carefully documented and structured for reuse |
| **Results Validated and Reproduced** | the main results obtained by someone other than the author, within a tolerance that does not change the paper's claims |

### What is missing today

| requirement | state |
|---|---|
| Artifact appendix, **up to 2 pages, placed before the References section**, ctuning.org template | ❌ **not written.** Only needed after acceptance, but it is a real writing task |
| Single-blind: author details may be included in the appendix | note: opposite of the paper |
| Container or VM (Docker, Singularity, VirtualBox, Vagrant) "strongly encouraged" | ❌ **not built.** Today the artifact is a pinned Triton wheel recipe plus a Python package |
| Archival deposit for the Available badge | ⏸️ **deliberately deferred to after the notification.** See below |
| Oracle-gated correctness, versioned CSVs, pinned rebuild recipe | ✅ already true, and it is the strongest part |
| ⚠️ **Specific hardware** | ❗ **action required, see below** |

### Do NOT deposit on Zenodo yet

A Zenodo record is public and indexed the moment it is minted. Depositing this artifact
before the notification would put the repository, its commit history and its author in front
of anyone who searched for the kernels or the pass, while the paper is under double-blind
review. That breaks the blind from the author's side, which is the one side that is
controllable.

**Sequence it after the notification**, whichever date that turns out to be. The Available
badge does not require an audit and is awarded by the publisher on a link, so it can be
obtained late without cost. Nothing about the submission on 10 September depends on it.

### The one thing to do before submitting, not after

CGO: *"If you have an unusual experimental setup that requires specific hardware ... or
proprietary software please **contact the Artifact Evaluation Chairs** ... **before the
submission**."*

This artifact needs NVIDIA GPUs across four architectures (RTX 4070, A100, H100, H200), CUDA,
and a from-source Triton build. That is squarely "requires specific hardware". The chairs are
**Olivia Hsu (owh@cmu.edu)** and **Jackson Woodruff (Jackson.Woodruff@ed.ac.uk)**.

A draft mail is in section (h). **It is not sent without approval.**

### AE timeline

Published dates are Round 1 only: R1 artifact submission 31 Aug 2026, clarification 5 to 9 Oct,
notification 15 Oct. **No R2 artifact dates are published yet.** The CFP also contradicts
itself on the window, saying "within two weeks of paper acceptance" in one section and "within
one week" in another. **Plan for one week after acceptance.** AE has its own site,
<https://cgo27ae.hotcrp.com/>, which the AE page says will open later.

---

## g) Prior work: the HPEC overlap

CGO's originality rule, verbatim: *"The paper must be original material that has not been
previously published in another conference or journal, nor is currently under review by
another conference or journal. Note that you may submit material presented previously at a
workshop without copyrighted proceedings."*

Two things follow, and they point the same way.

**Nothing is under review anywhere.** The ACM TACO submission of 5 July 2026 was returned by
the Editor-in-Chief on 7 July 2026. It has been closed for six weeks. ASPLOS was never
submitted to.

**The HPEC paper is a different paper, and that is measured.** *The Two Faces of Abstraction
Regret* (IEEE HPEC 2026, accepted, reaching IEEE Xplore around late September) is cited here
as **[21]**. HPEC has copyrighted proceedings, so it does not fall under the workshop
exemption, which makes the question "is this the same material" rather than "is it exempt".
A 10-word shingle comparison over both normalized texts, re-run on the final builds:

- **Body of this submission (p1 to p11) against the whole HPEC paper: zero shared 10-word
  runs. 0.00%.**
- Whole paper against whole paper: 4 runs, 56 words, **0.46%**. All four are bibliography
  entry titles: HybridSA, Triton, Hexcute, and the title of [21] itself.

So the only shared text is the set of papers both cite. There is no shared prose. The delta
is stated explicitly in the related-work section: the impossibility result for TritonGPU, the
compiler pass, the second architecture, the out-of-sample straggler law and the negative
controls do not exist in [21].

**CGO's own anonymity guidance endorses exactly how [21] is handled here**: *"if you are
extending your own work, you need to reference and discuss the past work in third person, as
if you were extending someone else's research."* The paper cites [21] by full name in third
person and never writes "our prior work". This is the prescribed form, not a leak.

⚠️ **CGO publishes no mechanism for declaring overlap.** There is no prior-work field
documented, and the form is behind a login. Check for one on first sign-in. If a field
exists, paste this, which names no one and is safe even if reviewer-visible:

```
An earlier and shorter paper by an author of this submission, cited here as [21],
was accepted at IEEE HPEC 2026 and will appear in IEEE Xplore before the CGO
notification date. It measures the DSL-versus-CUDA gap on the same automata
workload and stops there. This submission shares no prose with it: a 10-word
shingle comparison finds zero shared runs between this paper's body and that
paper, and the only matches anywhere are four bibliography entry titles. The
delta is stated explicitly in the related-work section. The impossibility result
for TritonGPU, the compiler pass, the second architecture, the out-of-sample
straggler law and the negative controls are new here.
```

---

## h) Draft mails, held for approval. Not sent.

### h1. To the artifact evaluation chairs, about the hardware

Required by the Call for Artifacts before submission, since AE interest is being ticked and
the artifact needs specific hardware. **Ready to send: the paper and the figures are final,
so this describes the artifact as it actually stands.**

```
To: owh@cmu.edu, Jackson.Woodruff@ed.ac.uk
Subject: CGO 2027 AE: artifact requiring specific NVIDIA GPUs, checking in before submission

Dear Olivia Hsu and Jackson Woodruff,

The Call for Artifacts asks authors with an unusual hardware setup to contact you
before submitting, so I am doing that ahead of the 10 September R2 deadline. I
intend to tick artifact interest on a research paper submission.

What the artifact is. The paper's central claim is that a primitive is
inexpressible in a production tile IR, and its cure is a compiler pass that
supplies that primitive below the IR. So the artifact is a from-source Triton
build: a pinned upstream commit plus a versioned patch, built to a wheel by a
one-command recipe, plus a Python package of oracle-gated kernels and the
versioned CSVs every number in the paper is read from.

What it needs. NVIDIA GPUs and CUDA. The headline speedups were measured on an
RTX 4070 and reproduced on an A100 from the same wheel, with two further results
on an H100 and an H200.

What splits cleanly. Two parts of the artifact need no GPU at all: the CPU
correctness oracle every kernel is gated against, and the regeneration of every
figure, which is driven entirely from the versioned CSVs. Two more are runnable
probes rather than prose, and also need no GPU: one exits zero on the expected
compile failure and one asserts that the MLIR verifier rejects the construct the
paper says is inexpressible. What needs a GPU is the speedups themselves.

Two questions:

1. Does the committee have access to NVIDIA GPUs of any recent generation? A
   single consumer-class card reproduces the main speedup; the cross-architecture
   result needs a datacenter card.
2. If not, would you accept an artifact evaluated on cloud GPUs, with a prepared
   environment provided by me for the evaluation period?

I would rather learn the answer now than package for an environment that does not
exist. I am happy to follow whichever route you prefer.

Sincerely,
Alessandro Potenza
Politecnico di Milano
```

### h2. To the program co-chair, about the date contradiction

Low priority, cosmetic, and only worth sending if the rebuttal window has to be planned around.

```
To: z.wang5@leeds.ac.uk
Subject: CGO 2027: R2 rebuttal and notification dates differ between the CFP and the calendar

Dear Zheng Wang,

The CGO 2027 pages carry two different sets of R2 dates. The CFP prose says the
author rebuttal period is 20 to 22 October and notification is 2 November. The
Important Dates panel, the /dates page and the conference home page say 27 to 29
October and 9 November. Both are live today, on the same site.

Could you confirm which is current? I ask only to keep the rebuttal window free.

Sincerely,
Alessandro Potenza
Politecnico di Milano
```

---

## Open items

- ⚠️ **Conflicts**: declare Davide Conficconi, fill the six author-only boxes, and re-run the
  Politecnico sweep inside HotCRP once signed in. The ERC and AEC are still being populated.
- ✅ **Both PDF compliance questions are closed.** The acknowledgments section is out, so the
  text is 11 pages exactly, and the figures now key their categories by hatch as well as
  fill and were checked by eye on a true greyscale render. See `docs/SUBMISSION_CGO.md`.
- ❗ **Mail h1 to the AE chairs**, ready to send. It describes the final artifact and must go
  before the submission.
- ⏸️ **Zenodo deposit is a post-notification step**, not a pre-submission one. Minting a
  public record now would break the double blind from the author's side.
- ⏳ **R2 artifact dates** are unpublished. Re-check the AE page nearer acceptance.
- ⏳ **Re-fetch the CFP in the last week before submitting.** It carries at least five internal
  contradictions today, which suggests it is still being edited.
