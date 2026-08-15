---
name: github-api-works-from-local
description: "From the laptop the full GitHub API on triton-lang/triton works; the 403 was the cloud session proxy, not GitHub"
metadata: 
  node_type: memory
  type: project
  originSessionId: f267ab28-27c7-4488-84d8-a06a0b48733c
  modified: 2026-08-15T18:45:23.602Z
---

Verified 2026-08-15 from the laptop: `gh api repos/triton-lang/triton/...` returns 200 for PR
details, comments, timeline, and `actions/runs`. Our permissions on that repo are `pull` only (no
push) — normal for a fork-based contributor: pushes go to `alepot55/triton`, comments go through the
issues API.

**Why:** the recorded "GitHub API gives 403 even with a valid token" finding was true only inside a
**cloud** session, where the session proxy intercepts `api.github.com`. It is not a property of the
token or of GitHub, so it must not be carried over to local work.

**How to apply:** from the laptop, verify upstream PR state directly with `gh` (never from memory,
never via `git ls-remote` workarounds — those are the cloud fallback). Posting comments upstream is
possible from here too, so any text parked in `docs/upstream/PING_DRAFTS.md` can actually be sent.
See [[laptop-tokens-in-env]] and [[triton-ci-needs-maintainer-approval]].
