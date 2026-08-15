---
name: git-push-credential-helper
description: git push dava 403 perché il credential helper "store" serviva un token vecchio prima di GITHUB_TOKEN
metadata:
  type: project
---

Il 15 ago `git push` su `alepot55/gpufsm` rispondeva `remote: Permission to alepot55/gpufsm.git
denied to alepot55` (403) mentre `gh api repos/alepot55/gpufsm` riportava `permissions.push: true`.
Causa: `credential.helper=store` globale serviva una credenziale vecchia da `~/.git-credentials`
**prima** che entrasse in gioco `GITHUB_TOKEN`. Non era il sandbox (stesso errore con sandbox
disabilitato) e non era GitHub.

**Why:** il sintomo (403 su push, API di lettura e permessi OK) porta fuori strada verso scope del
token o blocchi del proxy; la vera variabile è quale credenziale il *client git* presenta.

**How to apply:** la lista di helper si azzera con una voce vuota, poi si aggiunge quello di `gh`.
Già configurato in locale su questa repo (`git config --local`):

```
git config --local --replace-all credential.helper ""
git config --local --add credential.helper '!gh auth git-credential'
```

Una tantum per un push singolo: `git -c credential.helper= -c credential.helper='!gh auth
git-credential' push ...` — attenzione, senza la voce vuota il `store` globale vince lo stesso.
Vedi [[laptop-tokens-in-env]].
