# Upstream strategy — the "hire-first" path (chosen 2026-06-30)

Direction chosen by the user: **maximize the NVIDIA-hire signal directly** via upstream engagement,
rather than leading with a flagship paper. Hardware: RTX 4070 only for now (datacenter results stay
pre-registered falsifiable predictions). Source: parallel research (frontier / hire-signal / cure-
feasibility / theory) + a Triton-codebase recon, 2026-06-30.

## The three arms, ranked by signal (from the research)

1. **A merged PR in `triton-lang/triton`** — highest signal: proves you ship in TritonGPU/MLIR and earns a
   named-maintainer contact (referrals dominate compiler hiring). BUT it must fix a *known functional or
   performance issue* — `CONTRIBUTING.md` auto-classifies anything else as "controversial." A niche pass is
   not it.
2. **A credible design issue** characterizing the irregular-control-flow gap in the tile model — earns a
   maintainer thread (Tillet / Lezcano / Raoux) *without* asking them to merge niche code. Immediately
   shippable; backed by our measured law + partial pass. **This is the first concrete move.**
3. **A talk** (C4ML / LLVM Dev Mtg / GTC poster) backed by merged code — strong only after arm 1.

The cure-RFC fight (add a per-lane primitive to TritonGPU) is explicitly **de-prioritized**: core-maintainer
territory, slow, and the ecosystem energy is moving *toward* the tile abstraction (cuTile/Tile-IR), not
toward SIMT escape hatches. We pitch the gap as *complementary to NVIDIA's Tile-IR backend*, which their
own engineers say is weak on non-tensor / tensor-of-pointer / irregular-control paths.

## Honest finding that kills a tempting-but-fake PR (verified 2026-06-30)

The recon suggested "generalize the reduce-hoist into a uniform-reduce LICM PR." **Verified false, empirically:**
`tt.ReduceOp` carries the `Pure` trait (`include/triton/Dialect/Triton/IR/TritonOps.td:758`), so Triton's
existing `-triton-licm` already hoists a genuinely loop-invariant reduce. Reproduced with `triton-opt`:

```
# input: scf.for { %r = tt.reduce(%loop_invariant_tensor); acc += %r }
# output of `triton-opt -triton-licm`: the tt.reduce is moved ABOVE the scf.for.  (confirmed)
```

So there is **no LICM gap**. Our `tritongpu-thread-region` reduce-hoist is novel only for the *loop-variant*
case (the reduce depends on the loop counter `%j`; we recognize that the loop **bound** is `max(trip)` —
loop-invariant — and rewrite to a hoisted `reduce_max(%trip)` + scalar counter). That rewrite is real but
**automata-shaped and does not fire on any canonical Triton kernel** → it stays a **paper artifact**, not a PR.

## First concrete move (this path)

- `docs/upstream/triton-issue-irregular-control.md` — reviewer-ready design issue (user posts it). High
  signal, no niche-merge ask.
- Open hunt (arm 1): a genuine, mergeable canonicalization/fold or a functional/perf fix discovered while
  building — NOT fabricated. Template = merged PRs #10734 (ReduceOp combiner robustness, rev. @lezcano),
  #9971 (ptr canonicalization). Status: searching; will not ship until a real gap is found and verified.

## Reframe for everything

Pitch = *"I characterized and partially closed a regret class the Tile-IR backend currently exhibits,"* not
*"add a primitive to Triton."* Maps onto a real open NVIDIA problem + a real team (cuTile: Jie Xin, Jonathan
Bentz), instead of an OpenAI-owned IR governance fight.
