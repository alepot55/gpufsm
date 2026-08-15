---
name: laptop-tokens-in-env
description: "On the laptop, GITHUB_TOKEN and MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are already exported in the environment"
metadata: 
  node_type: memory
  type: project
  originSessionId: f267ab28-27c7-4488-84d8-a06a0b48733c
  modified: 2026-08-15T18:45:17.258Z
---

Since 2026-08-15 the laptop shell exports `GITHUB_TOKEN` (classic PAT, 40 chars, scopes include
`repo`, `workflow`, `admin:org`, `gist`) plus `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Never print
or commit the values.

**Why:** the user set them up so a local Claude Code session can do upstream GitHub work and drive
Modal GPU runs without any interactive login.

**How to apply:** just use `gh` — it already picks `GITHUB_TOKEN` up (`gh auth status` shows
"Logged in ... (GITHUB_TOKEN)", which shadows the older keyring `gho_` token with narrower scopes).
Modal è installato il 15 ago sera nel venv `/home/alepot55/Desktop/projects/gpufsm/.venv`
(`python3 -m venv .venv --system-site-packages`, l'host è PEP 668; `modal 1.5.4`) e
`.venv/bin/python scripts/modal_gpu.py --preflight` dà 3 PASS. Da qui la GPU si affitta.
See [[github-api-works-from-local]].
