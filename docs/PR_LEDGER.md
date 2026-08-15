# Ledger delle PR — stato verificato il 2026-08-15 (ri-verificato la sera, da locale)

Registro unico di **tutte** le pull request del progetto, interne e upstream, con lo stato verificato
alla fonte (non dalla memoria). Serve a non riscoprire ogni sessione cosa è stato mandato, cosa è vivo
e cosa vale.

**Metodo — da sessione LOCALE (preferito).** Con `GITHUB_TOKEN` in env l'API GitHub su
`triton-lang/triton` risponde **200**: il 403 di cui parlano le note più vecchie era del **proxy della
sessione cloud**, non di GitHub e non del token. Verificato il 15 ago dal portatile.

```
gh api "search/issues?q=repo:triton-lang/triton+author:alepot55+type:pr" \
  --jq '.items[] | "\(.number) \(.state) \(.pull_request.merged_at // "not-merged") \(.title)"'
gh api repos/triton-lang/triton/pulls/<N> --jq '{state,mergeable,mergeable_state,updated_at}'
gh api "repos/triton-lang/triton/actions/runs?status=action_required" \
  --jq '.workflow_runs[] | select(.head_repository.owner.login=="alepot55") | "\(.name) \(.head_branch) \(.html_url)"'
```
Permessi sulla repo upstream: `pull` only (nessun push) — normale per contributor esterno: i push vanno
sul fork `alepot55/triton`, i commenti passano dalle issues API (**da locale si può commentare davvero**).

**Metodo — da sessione CLOUD (ripiego, l'API dà 403):**

```
git ls-remote https://github.com/triton-lang/triton 'refs/pull/<N>/*'
# head+merge  -> PR aperta e mergeable
# solo head   -> PR chiusa oppure mergiata
```
poi la pagina pubblica della PR per lo stato esatto.

---

## A. Upstream `triton-lang/triton` — l'unico contributo esterno che conta

Score: **1 mergiata, 3 aperte (#10766, #11323, #11324), 4 chiuse + 1 RFC chiusa.**

| PR | Titolo | Aperta | Stato oggi | Chi ha deciso |
|----|--------|--------|-----------|---------------|
| [#11324](https://github.com/triton-lang/triton/pull/11324) | `[Membar] Treat warp_yield as a CTA sync point` | 16 ago | **APERTA** — −30 barriere nel PTX generato, misurate su H100 | in attesa |
| [#11323](https://github.com/triton-lang/triton/pull/11323) | `[Membar] Treat warp_specialize entry as a CTA sync point` | 16 ago | **APERTA**, CI da approvare, review chiesta a Jokeren+ptillet | in attesa |
| [#11311](https://github.com/triton-lang/triton/pull/11311) | `[DOCS] examples/plugins: make the Example 4 python block parse` | 14 ago | **MERGIATA** 15 ago | Jokeren |
| [#10766](https://github.com/triton-lang/triton/pull/10766) | `[TRITON] Fold split(join(a,b)) -> (a,b) and join(split(x)) -> x` | 30 giu | **APERTA**, `mergeable:true` / `blocked`, CI da approvare | in attesa (ping 14 ago) |
| [#10780](https://github.com/triton-lang/triton/pull/10780) | `[DOCS] Precompute the stages-inspection key/hash in the plugin examples` | 2 lug | **APERTA**, contestata, CI da approvare | CRobeck ha obiettato |
| [#10788](https://github.com/triton-lang/triton/pull/10788) | `[FRONTEND] Fix fp8 block-pointer load with padding_option="zero"` | 4 lug | CHIUSA 4 lug | ThomasRaoux |
| [#10785](https://github.com/triton-lang/triton/pull/10785) | `[TritonGPU] Fail cleanly when lowering global_scratch_alloc without an offset` | 3 lug | CHIUSA 3 lug | ThomasRaoux |
| [#10774](https://github.com/triton-lang/triton/pull/10774) | `[NVIDIA] Add verified opt-in per-lane loop retirement pass (RFC #10773)` | 1 lug | CHIUSA (era draft) | ritirata da noi |
| [#10773](https://github.com/triton-lang/triton/issues/10773) *(issue)* | `[RFC] Per-lane retirement for divergent while-loops` | 1 lug | CHIUSA "completed" 3 lug | — |

### #11311 — la mergiata (quella che "vale poco", e perché va comunque tenuta)
`c346e50c7bb102f35f04d7200a3bc6194bec4c33`, un solo file (`examples/plugins/README.md`), **+9/-5**,
solo documentazione: l'`IndentationError` dell'Example 4 (guardie di early-return a 2 spazi in corpi a
4, difetto entrato con #8401) più `pathlib`/`hashlib` usati senza import. Aperta il 14, mergiata il 15,
CI sbloccata nello stesso giro.

Valore tecnico ≈ zero. Valore reale: (a) chiude la debolezza "nessun contributo upstream" con una riga
verificabile nella history di Triton; (b) **prova la tattica dello split** — la stessa identica
correzione, impacchettata dentro #10780 insieme a una modifica già respinta da un maintainer, era ferma
da 43 giorni senza CI; estratta da sola su un branch nuovo da `main`, un file solo, è passata in meno di
24 ore. Regola derivata: **mai legare un fix non controverso a una modifica contestata.**

### #10766 — la PR che vale davvero (viva, in attesa)
Codice nel dialect (`lib/Dialect/Triton/IR/`), non docs. Ping del 14 ago con lo use case misurato che
Raoux chiedeva dal 1 luglio: helper `@triton.jit` che ritorna `tl.join(cheap, tl.dot(x,w))` + chiamante
che splitta e usa solo la metà economica → senza il fold il round-trip tiene vivo il `tt.dot` morto **e
il suo staging in shared memory**: 16 KB smem, 78 vs 40 registri, 224 vs 72 istruzioni SASS, 31,5 → 23,3 µs
(1,35× su 4070, 3 run entro l'1%). Caveat tenuto nel testo: se la metà morta è solo una load, ptxas la
toglie da sola e il SASS è identico → rivendicare **solo** il caso shared-memory.
Sound vs #10749 (che rilassa `JoinOp::verify` a `ignoreRegBroadcast=true`): i nostri fold confrontano i
tipi per uguaglianza **esatta**, quindi sono strettamente più conservativi; declinano sui round-trip che
differiscono solo per register broadcast.
Nessuna risposta di un maintainer dopo il 14 ago. **Branch `fold-split-join` da congelare.**

Stato esatto al 15 ago sera (`gh api`, head `00ae4d7e`): `state: open`, `mergeable: true`,
`mergeable_state: "blocked"`, 8 commenti + 1 review comment (peterbell10, 2 lug), ultimo evento = il
nostro ping del 14 ago 10:08 con menzione di ThomasRaoux/peterbell10/neildhar. ⚠️ **Zero check-runs sul
head attuale**: il refresh del 14 ago ha rimesso la run in `action_required`
([run 31785463900](https://github.com/triton-lang/triton/actions/runs/31785463900)) — cioè oggi la PR
**non ha il verde**, non perché sia rotta ma perché nessuno ha cliccato "Approve and run workflows".
Nota di contesto: **#10749 ("Generalize split and join layout handling") è stata mergiata il 7 lug**;
i nostri fold restano sound perché confrontano i tipi per uguaglianza esatta.

### #10780 — resta solo la parte contestata
Dopo lo split contiene **solo** il precompute di key/hash (+26/-17). Obiezione di CRobeck (le chiavi non
vanno ricalcolate per intercettare cambi di pipeline?) già risposta: una pipeline statica costruita da
moduli importati non cambia senza re-import, e l'Example 4 dinamico deriva comunque la chiave dal
contenuto corrente. Rimosso dalla descrizione il numero "38,9 → 12,4 µs": veniva da un runtime patchato
in locale, non da Triton stock. Chiusura già offerta a Jokeren/CRobeck.

Stato esatto al 15 ago sera (head `0fd6ff8b`, 5 commit, 1 file): `mergeable: true`,
`mergeable_state: "blocked"`, ultima attività = il **nostro** commento delle 13:23 dopo aver mergiato
`main` (post-merge di #11311, così il branch porta solo il precompute). Quel merge ha creato una nuova
run in `action_required` ([31887095757](https://github.com/triton-lang/triton/actions/runs/31887095757)):
anche qui nessun check verde finché un maintainer non approva.

### Le chiuse — perché, senza girarci intorno
- **#10788**: ThomasRaoux, *"block pointer is deprecated and will be removed soon, so I don't think we
  want patches for it"*. Niente da salvare.
- **#10785**: ThomasRaoux, *"either this path should never happen or we should make it work"*. Accettato:
  il path è raggiungibile solo da MLIR scritto a mano su pipeline AMD (l'op la genera solo la TMA
  device-side NVIDIA) → un band-aid su un path che non dovrebbe esistere non è la fix giusta.
- **#10774 + RFC #10773**: il pass per-lane loop retirement. L'RFC contiene i numeri buoni (2,5–4,2×
  end-to-end, 39× istruzioni emesse in meno — 36,1M → 0,92M, 1,14–1,25× su workload reali) ma la strada
  "aggiungere una primitiva SIMT a TritonGPU" è territorio dei core maintainer e va contro la direzione
  dell'ecosistema (cuTile/Tile-IR). **De-prioritizzata di proposito** (`docs/upstream/STRATEGY.md`), non
  fallita: il materiale resta artefatto del paper (la "cura").

### Meccaniche upstream da non riscoprire
- **La CI degli outside contributor non parte da sola**: il repo gira con "require approval for all
  outside collaborators", ogni push mette la run in `action_required`. Non è la variante primo-contributo
  (alepot55 era già stato approvato 3 volte). ⇒ **ogni push azzera il verde**, anche il merge di `main`
  per rinfrescare il branch. Fare tutti i push in una volta, poi **congelare** e pingare citando l'URL
  della run pendente.
- **CODEOWNERS è un vicolo cieco**: nessuna regola per `lib/Dialect/Triton/IR/`, `test/`, `examples/` ⇒
  tutto cade su `* @ptillet`, auto-assegnato ma senza review né merge dal 17 giugno. **Mai pingare lui.**
  Chi mergia davvero: Jokeren, ThomasRaoux, peterbell10, lezcano. Dialect/canonicalize → ThomasRaoux;
  `examples/plugins/` → Jokeren; semantica split/join → neildhar.
- **Niente `git push --force`** (bloccato dal classifier e comunque sgradito): per aggiornare un branch di
  PR usare il merge di `main`, come il pulsante "Update branch".
- **Le sessioni cloud non possono commentare** su `triton-lang/triton`, e lì **un token GitHub in env non
  cambia niente** (verificato il 15 ago, non assunto). ⚠️ Vale **solo per il cloud**: dal portatile, con
  lo stesso tipo di token in env, la stessa API risponde 200 (vedi il metodo in cima). Il blocco è del
  **proxy della sessione**, non di GitHub: `api.github.com` è intercettato e risponde 403 con un messaggio suo. Con `GITHUB_TOKEN` in env:
  `GET /user` → 200 `alepot55` (il token è valido), `GET /repos/triton-lang/triton/pulls/10766` → 403
  "not enabled for this session", e persino `GET /repos/alepot55/gpufsm` → 403 "An org admin must connect
  the Claude GitHub App". `/root/.ccr/README.md` dice esplicitamente di non aggirare il blocco. Vie
  tentate e chiuse: `add_repo` **read** = solo git anonimo (niente API), `add_repo` **push** = negato dal
  classifier (attach cross-owner). ⇒ **le uniche vie sono i tool MCP `mcp__github__*` (scoped a
  `alepot55/gpufsm`) per la repo nostra, e le pagine pubbliche + `git ls-remote` in lettura per Triton.**
  I testi dei commenti si preparano in `docs/upstream/PING_DRAFTS.md` e si postano da sessione locale
  con `gh`.
- **Come allargare l'accesso oltre la singola repo** (dai doc ufficiali, 15 ago): l'installazione della
  GitHub App **non è** il controllo d'accesso — *"a cloud session can access any repository the connecting
  GitHub account can see, not just the repositories the Claude GitHub App is installed on. App
  installation enables PR webhooks for Auto-fix; it is not a session-level access control."* Quindi:
  1. **una volta sola**, dal terminale locale dentro `claude`: **`/web-setup`** → lega il token del `gh`
     locale all'account Claude (`Connected as alepot55`); le sessioni cloud vedono quello che vede `gh`;
  2. **per sessione**: il selettore repo di claude.ai/code accetta **più repo**, oppure si prefilla
     `https://claude.ai/code?repositories=alepot55/gpufsm,triton-lang/triton`;
  3. **a sessione avviata**: `add_repo`, che però chiede un'approvazione di permesso (in auto mode il
     classifier nega da solo).
  ⚠️ Non verificato: se la piattaforma accetti di attaccare una repo di un'**org terza** dove non siamo
  collaboratori (`add_repo` avvisa che "cross-owner attachments may still be refused"). Ripiego sicuro =
  il fork `alepot55/triton` (nostro). Per **commentare** upstream resta comunque `gh` locale.
  Fonti: `code.claude.com/docs/en/claude-code-on-the-web`, `.../web-quickstart`.

---

## B. Interne `alepot55/gpufsm` — 25 PR, **tutte mergiate**, nessuna aperta

Sono PR di lavoro auto-mergiate: il contributo è il contenuto, non la PR. Elencate perché sono l'indice
cronologico più leggibile di cosa è stato fatto e quando.

| # | Cosa ha portato | Data |
|---|-----------------|------|
| 1 | Validazione GPU + ablation memory-centric + multi-DSL (Triton/CUDA/Warp) | 25 giu |
| 2 | Paper "two faces" + suite 6 famiglie reali + probe Gluon falsificabile | 26 giu |
| 3 | Kernel block-parallel (3–9× auditato), knee L2 del DFA, ablation shared-mem, tabella SOTA, packaging AE | 26 giu |
| 4 | Nsight: `worklist_warp` è **latency-bound**, non memory-bound | 26 giu |
| 5 | Worklist compatto (**ipotesi confutata**), holdout del cost model (scopato), freschezza letteratura | 26 giu |
| 6 | Knee DFA multi-seed + riproduzione dei numeri headline + hardening dello sweep | 26 giu |
| 7 | Verifica: Warp-batte-CUDA robusto + Triton-DFA-flat è un ceiling scalare | 26 giu |
| 8 | **2×2 paradigma × altezza** + ablation causale della primitiva + SOTA a 35 ref | 26 giu |
| 9 | Throughput su automi reali, unificazione della capability→cost map, piano 2ª GPU | 26 giu |
| 10 | Hardening statistico + completezza AE | 26 giu |
| 11–12 | Prosa del paper a profondità full-conference (intro/metodologia, background/conclusione) | 26 giu |
| 13 | Audit di consistenza dei numeri vs CSV (1 drift trovato e corretto) | 26 giu |
| 14 | Riconciliazione decomposizione a 2 assi ↔ finding della primitiva unica | 26 giu |
| 15–16 | Script campagna 2ª GPU + **A100 conferma lo shift del knee L2** (validità esterna cross-arch) | 26 giu |
| 17 | Versione **HPEC 2026** a 6 pagine + A100 cross-arch + audit zero-difetti | 27 giu |
| 18 | Fix tipografico del titolo | 27 giu |
| 19 | **Paper-2 "cura"**: backend GPU validati + tecniche memory-centric, submission-ready | 30 giu |
| 20 | Upstream: refresh delle due PR Triton vive + stato registrato | 14 ago |
| 21 | Infra: runner GPU generico su **Modal** guidato da sessione cloud | 15 ago |
| 22 | Fix: lo script di setup non deve `cat` la CA del proxy (non esiste ancora) | 15 ago |
| 23 | Questo ledger + modello operativo local-first | 15 ago |
| 24 | Memoria di progetto dentro la repo (`docs/memory/`) + ri-verifica PR da locale + Modal pronto sul portatile | 15 ago |
| 25 | Trappola del credential helper di git + Modal verificato end-to-end su T4 | 15 ago |

⚠️ **Buco nel registro delle PR, non nel lavoro:** tra la #19 (30 giu) e la #20 (14 ago) ci sono **53
commit su `main` senza PR** — sottomissione TACO (desk-reject), pivot CGO/PPoPP, generalizzazione della
cura, prima ondata upstream. Sono arrivati via `dev` e il merge `a3400ae` (12 ago). Chi cerca quel
periodo nelle PR non lo trova: sta in `git log`, in `docs/TACO_*`, `docs/PPOPP_PLAN.md`,
`docs/SUBMISSION_ASPLOS.md`.

---

## C. Dove sta il valore, onestamente

1. **Paper** — pacchetto ASPLOS 2027 (ciclo Fall, **9 set 2026 AoE**) pronto e committato
   (`paper2/gpufsm_asplos.{tex,pdf}`, versione named, abstract ASCII, nota di resubmission, disclosure AI).
   Unico bloccante: **il sito HotCRP di settembre non è ancora aperto**. Non è nostro.
2. **Upstream** — 1 merge reale (docs, valore tecnico ≈ 0 ma precedente + contatto) e **#10766 viva**,
   che è quella con sostanza. Prossima mossa: aspettare, non ritoccare i branch.
3. **Ricerca** — i contributi veri stanno nel repo, non nelle PR: two-faces dell'abstraction regret, il
   2×2 (il regret segue il **paradigma di esecuzione**, non l'altezza dell'astrazione), l'ablation causale
   (cliff 16×), la capability→cost map con la primitiva mancante nominata (scalar-gather-in-tile), la
   validazione cross-arch su A100 e i risultati negativi tenuti (worklist compatto, shared-mem inerte,
   cost model non predittivo per Triton).
