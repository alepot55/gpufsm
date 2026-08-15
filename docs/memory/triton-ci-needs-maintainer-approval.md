---
name: triton-ci-needs-maintainer-approval
description: "Every push to a Triton PR from an outside contributor parks CI in action_required, so a push destroys the green status"
metadata: 
  node_type: memory
  type: project
  originSessionId: f267ab28-27c7-4488-84d8-a06a0b48733c
  modified: 2026-08-15T18:45:28.966Z
---

`triton-lang/triton` requires maintainer approval for all outside-collaborator workflow runs. Any
push — including merging `main` to refresh a branch — puts the run in `action_required` and the PR
shows no checks at all until someone clicks "Approve and run workflows".

**Why:** it explains why PRs sit with `mergeable_state: "blocked"` and zero check-runs while looking
healthy, and why refreshing a stale branch is a real cost, not a free hygiene move.

**How to apply:** batch every push, then freeze the branch and ping citing the pending run URL from
`gh api "repos/triton-lang/triton/actions/runs?status=action_required"`. Do not re-push to "retrigger"
anything. See [[github-api-works-from-local]].
