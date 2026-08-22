# Ledger delle PR — stato verificato il 2026-08-22 (mattina, da locale, `gh api` su tutti i thread)

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

Score: **2 mergiate (#11311, #11324), 2 aperte (#10766, #11325), 5 chiuse + 1 RFC chiusa**, piu' 2 issue
aperte da noi (#11326, #11328). **17 ago 10:21: primo scambio tecnico con un maintainer.** Jokeren ha chiuso #11323 come
"micro optimization" dopo averne discusso con @jeffniu-openai. La patch era corretta (verificato in
`WarpSpecializeUtility.cpp:555-559`: le due barriere sono incondizionate), il valore no — e lo
dicevamo noi stessi nella PR. **#11324 e' stata mergiata il 20 ago**; #11325 resta aperta e `blocked`.

**Aggiornamento 22 ago.** Due cose sono cambiate e nessuna delle due era visibile dal ledger vecchio:
**#10766 ha la CI verde** (approvata e girata il 20 ago 15:47Z, 10 check su 10, `integration-tests-nvidia`
su a100/h100/gb200 inclusi) e **#11325 e' ferma per decisione esplicita**, non per silenzio. In piu'
l'issue #11328 ha prodotto una collaborazione con due contributori esterni, raccontata sotto nella
sezione "La catena 11328, 11393, 11396".

| PR | Titolo | Aperta | Stato oggi | Chi ha deciso |
|----|--------|--------|-----------|---------------|
| [#11325](https://github.com/triton-lang/triton/pull/11325) | `[Membar] Key subslice offset comparison on the value they came from` | 16 ago | **APERTA, ma ferma per decisione**: 20 ago 15:03Z Jokeren *"This is just a workaround to conservatively reject these cases so I don't think it's the right fix and will keep this PR pending without merging"* | Jokeren, in sospeso |
| [#11324](https://github.com/triton-lang/triton/pull/11324) | `[Membar] Treat warp_yield as a CTA sync point` | 16 ago | ✅ **MERGIATA il 20 ago 13:17Z** da Jokeren, commit `37a4b78fc` — −30 barriere nel PTX; misurata su H100 il 16 ago: **nessun guadagno di velocita'**, dato pubblicato sulla PR | chiusa |
| [#11323](https://github.com/triton-lang/triton/pull/11323) | `[Membar] Treat warp_specialize entry as a CTA sync point` | 16 ago | **CHIUSA** 17 ago — "micro optimization" (Jokeren + jeffniu-openai). Patch corretta, valore marginale: non cambiava il codice generato, e lo dicevamo noi | Jokeren |
| [#11311](https://github.com/triton-lang/triton/pull/11311) | `[DOCS] examples/plugins: make the Example 4 python block parse` | 14 ago | **MERGIATA** 15 ago | Jokeren |
| [#10766](https://github.com/triton-lang/triton/pull/10766) | `[TRITON] Fold split(join(a,b)) -> (a,b) and join(split(x)) -> x` | 30 giu | **APERTA**, `mergeable:true` / `blocked`, **CI VERDE** su `e60dc6ff7` (10/10, 20 ago 15:47Z). Manca solo una review che approvi | in attesa (review chiesta 22 ago) |
| [#10780](https://github.com/triton-lang/triton/pull/10780) | `[DOCS] Precompute the stages-inspection key/hash in the plugin examples` | 2 lug | **APERTA**, contestata, CI da approvare | CRobeck ha obiettato |
| [#10788](https://github.com/triton-lang/triton/pull/10788) | `[FRONTEND] Fix fp8 block-pointer load with padding_option="zero"` | 4 lug | CHIUSA 4 lug | ThomasRaoux |
| [#10785](https://github.com/triton-lang/triton/pull/10785) | `[TritonGPU] Fail cleanly when lowering global_scratch_alloc without an offset` | 3 lug | CHIUSA 3 lug | ThomasRaoux |
| [#10774](https://github.com/triton-lang/triton/pull/10774) | `[NVIDIA] Add verified opt-in per-lane loop retirement pass (RFC #10773)` | 1 lug | CHIUSA (era draft) | ritirata da noi |
| [#10773](https://github.com/triton-lang/triton/issues/10773) *(issue)* | `[RFC] Per-lane retirement for divergent while-loops` | 1 lug | CHIUSA "completed" 3 lug | — |


### Le tre nuove del 16 ago — la differenza che conta

#11323 e #11324 tolgono barriere **di troppo**; #11325 ne aggiunge una **mancante**. Solo la terza e'
un bug di correttezza, ed e' l'unica che non rischia il verdetto "trivial"
([[triton-rejects-trivial-prs]]). Dettagli e riproduzioni:
`docs/upstream/MEMBAR_SUBSLICE_COORDS.md`, `MEMBAR_CALL_BOUNDARY.md`, `CLUSTER_BARRIER_VIEW.md`.

Le due issue (#11326 confine di chiamata, #11328 vista multicast) sono aperte **come issue e non come
PR** di proposito: in un caso la correzione e' una scelta di disegno interprocedurale, nell'altro la
correzione ovvia fa abortire la compilazione. In entrambe e' offerta l'implementazione.

### La catena 11328, 11393, 11396: la prima volta che qualcuno raccoglie una nostra issue

Aperta il 16 ago come issue e non come PR di proposito (la correzione ovvia fa abortire la
compilazione), **#11328 e' stata raccolta da due contributori esterni fra il 20 e il 21 ago**:
`layahaasini` (*"started looking into this"*, 20 ago 23:42Z) e `whutsunxu`, che ha letto
`AllocationAnalysis` e ha portato un caso minimo.

Come e' andata, nell'ordine, perche' e' il primo scambio tecnico paritetico che abbiamo avuto qui:

1. **21 ago 10:20Z** whutsunxu propone una causa e un repro minimo.
2. **10:25Z** rispondiamo che il suo caso **non** riproduce su `4d641eca`. Sbagliato: avevamo usato
   la pipeline di pass sbagliata.
3. **11:16Z** ritiriamo, con la RUN line giusta presa da `tritonnvidiagpu_to_llvm.mlir` l'errore
   riproduce. Retrattazione esplicita, non silenziosa.
4. **13:52Z** whutsunxu scorpora [#11393](https://github.com/triton-lang/triton/issues/11393)
   perche' i due bug hanno fix dipendenti.
5. **17:07Z** pubblichiamo cinque arm misurati sullo stesso base `4d641eca2b`, **cinque build
   separate con md5 diverso** perche' un binario stantio ci aveva gia' ingannati una volta
   ([[a-missing-binary-is-not-a-red-test]]).
6. **19:00Z** whutsunxu apre [#11396](https://github.com/triton-lang/triton/pull/11396); riportiamo
   li' le misure gia' fatte cosi' non vanno rifatte.
7. **19:11Z** Jokeren mette `CHANGES_REQUESTED`: togliere l'assert e' sbagliato, l'eccezione va
   ristretta a `LocalAtomicScatterRMWOp`.
8. **19:54Z** ritiriamo il nostro endorsement pubblicamente: *"Jokeren is right and my endorsement
   above was wrong"*. Avevamo verificato che il crash sparisse, non cosa succedeva all'invariante
   per tutti gli altri op ([[judge-a-fix-by-the-invariant-not-the-symptom]]).
9. **23:01Z** Jokeren chiede a whutsunxu di spostare il test in `membar-cluster`. La palla e' sua.

**Stato al 22 ago:** #11396 aperta, `CHANGES_REQUESTED`, head `fb9cbec0d`. **Il nostro fix di #11328
dipende da questa**, quindi qui non si spinge: si aspetta e si misura quando serve. #11393 aperta.

Due lezioni gia' pagate in questa catena: la misura vale come contributo anche su una PR che non e'
nostra, e **una ritrattazione veloce costa meno di una difesa**. Entrambe le volte che abbiamo avuto
torto, dirlo entro l'ora ha tenuto vivo lo scambio.

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

### #10766 — la PR che vale davvero (viva, CI verde, in attesa di una review)
Codice nel dialect (`lib/Dialect/Triton/IR/`), non docs. Ping del 14 ago con lo use case misurato che
Raoux chiedeva dal 1 luglio: helper `@triton.jit` che ritorna `tl.join(cheap, tl.dot(x,w))` + chiamante
che splitta e usa solo la metà economica → senza il fold il round-trip tiene vivo il `tt.dot` morto **e
il suo staging in shared memory**: 16 KB smem, 78 vs 40 registri, 224 vs 72 istruzioni SASS, 31,5 → 23,3 µs
(1,35× su 4070, 3 run entro l'1%). Caveat tenuto nel testo: se la metà morta è solo una load, ptxas la
toglie da sola e il SASS è identico → rivendicare **solo** il caso shared-memory.
Sound vs #10749 (che rilassa `JoinOp::verify` a `ignoreRegBroadcast=true`): i nostri fold confrontano i
tipi per uguaglianza **esatta**, quindi sono strettamente più conservativi; declinano sui round-trip che
differiscono solo per register broadcast.
**Branch `fold-split-join` da congelare.**

**Stato al 22 ago** (`gh api`, head `e60dc6ff7`): `state: open`, `mergeable: true`,
`mergeable_state: "blocked"`, **CI VERDE**. Il 19 ago ThomasRaoux ha chiesto di riscrivere la
descrizione perché stesse in piedi da sola come messaggio di commit; fatto, e il 20 ago **qualcuno ha
approvato la run**: 10 check su 10 verdi alle 15:47Z, `pre-commit`, `build-macos`,
`integration-tests-nvidia` su a100/h100/gb200, `integration-tests-amd` su gfx90a/gfx942/gfx950, e
`proton-tests-amd`. Il ledger precedente diceva "zero check-runs, nessun verde": era vero il 15 ago e
**falso da due giorni**, e non ce ne siamo accorti perché nessuno guarda i check-runs se non li
chiede. `mergeable_state: blocked` adesso vuol dire una cosa sola: **manca una review che approvi**,
non manca la CI. Le uniche review sul thread sono un `COMMENTED` di peterbell10 del 2 luglio e due
nostri. Chiesta la review a ThomasRaoux e peterbell10 il 22 ago.

⚠️ Regola che questo caso stabilisce: **quando chiedi a un maintainer di approvare la CI, il giro non
finisce quando lui risponde, finisce quando guardi i check-runs.** Qui la risposta non e' mai
arrivata come commento, e' arrivata come dieci job verdi.

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

## B. Interne `alepot55/gpufsm` — 61 PR, **tutte mergiate**, nessuna aperta (al 22 ago 2026)

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

## B-bis. Upstream `llvm/llvm-project` — 1 mergiata, 6 aperte (stato: 22 ago 2026, mattina)

Bersagli scelti col criterio di [[pick-uncontested-bugs-not-design-changes]]: **solo** bug a cui
risponde una macchina. Su **14 rilievi ricevuti, ZERO toccano la sostanza** di una correzione: sono
tutti nomi e commenti. Il criterio regge. Controprova nello stesso giorno: su Triton la PR #11323
(un'*ottimizzazione*) e' stata chiusa con "this seems like a micro optimization".

**Il collo di bottiglia oggi non e' la review, e' il landing.** Tre PR su sei sono approvate e verdi
e nessuna delle tre e' atterrata: non abbiamo commit access, quindi ogni merge richiede che un
revisore lo faccia a mano, e va **chiesto**. Su #216854 la richiesta non era mai partita.

| PR | cosa | stato al 22 ago |
|---|---|---|
| [#216851](https://github.com/llvm/llvm-project/pull/216851) | mem2reg crash su `memref<0xf32>` | ✅ **MERGIATA il 19 ago 2026 15:59Z**, commit `0ed130af5`, landing fatto da FedericoBruzzone dopo 3 approvazioni (lui, gysit, Jianhui-Li). **Prima patch LLVM atterrata.** |
| [#216605](https://github.com/llvm/llvm-project/pull/216605) | affine LICM ignora valori catturati da regioni | `LGTM % nits` di FedericoBruzzone, 6 suggerimenti applicati il 18 ago (`5a4138d5a`). **Nessuna approvazione formale, ferma da 4 giorni.** Non pingata il 22 per non mandare tre richieste allo stesso revisore in un colpo |
| [#216853](https://github.com/llvm/llvm-project/pull/216853) | coalescing SCF fonde iter_arg diversi | correzione nido imperfetto spinta (`48097f51b`); revisori richiesti su nostra indicazione. **Zero approvazioni**, nessun movimento dal 19 ago |
| [#216854](https://github.com/llvm/llvm-project/pull/216854) | `multi_reduction` non valida `reduction_dims` | ✅ **2 approvazioni** (banach-space 19 ago, FedericoBruzzone 20 ago), **entrambe dopo l'ultimo push**, CI verde su `9482db30f`. Federico aveva scritto *"we might wait a day before landing it"*: il giorno e' passato, **landing chiesto il 22 ago** |
| [#216852](https://github.com/llvm/llvm-project/pull/216852) | SCF non dichiara `cf` dipendente (1 riga) | ⚠️ **unica questione di sostanza aperta**: `Hardcode84` obietta che quella canonicalizzazione non dovrebbe creare `cf`. Concesso, palla a loro. Fermo dal 17 ago |
| [#216947](https://github.com/llvm/llvm-project/pull/216947) | VectorToSCF asserisce senza `AutomaticAllocationScope` | ✅ **2 approvazioni** (FedericoBruzzone 18 ago, banach-space 19 ago 14:33Z), CI verde su `5e8cc0051`. ⚠️ L'ultimo push e' delle 15:49Z, quindi **posteriore a entrambe**: applicava pero' solo la riformulazione di un commento che Federico aveva suggerito due minuti prima. **Landing chiesto il 22 ago**, dichiarando cosa conteneva quel push |

**APERTA il 19 ago 2026 come [#217392](https://github.com/llvm/llvm-project/pull/217392)** (era la riserva),
**APPROVATA da `matthias-springer` il 20 ago 06:54Z**, CI verde (11 pass) su `86bbca689`, landing chiesto
il 20 ago 09:51Z. ⚠️ **Non e' ancora atterrata due giorni dopo**: ri-pingata il 22 ago. E' il caso che
ha fatto capire che il landing va inseguito, non dato per fatto una volta chiesto. Per
l'issue [#203858](https://github.com/llvm/llvm-project/issues/203858)
`scf::loopUnrollByFactor` asserisce `expected constant loop bound` (`Utils.cpp:404`). Il difetto e'
piu' largo di come lo descrive la segnalazione: `constantTripCount` risponde su **tre** strade in cui
gli estremi NON sono costanti, e la funzione le legge tutte come costanti.

1. `lb == ub` (stesso Value) -> 0 iterazioni;
2. `lb == 0` e `ub == step` -> 1 iterazione;
3. `ub` offset costante da un `lb` non costante (via `computeUbMinusLb`, richiede `nsw`).

Fix (10 righe): si imbocca il ramo costante solo se **tutti e tre** gli estremi lo sono, altrimenti si
cade sul percorso dinamico gia' esistente, che li gestisce correttamente. Gli altri due chiamanti di
`getStaticTripCount` in quel file usano solo il conteggio, mai gli estremi: il difetto e' confinato.

- Worktree `~/.cache/llvmwt/wt-unroll`, ramo `scf-unroll-nonconstant-bounds`, **ribasato il 19 ago su
  `0ed130af5`** (284 commit assorbiti senza conflitti), commit `86bbca689`. Test: 3 casi in `mlir/test/Dialect/SCF/loop-unroll.mlir`, uno per strada.
- Verificato ai due estremi **allo stesso ref**, rifatto il 19 ago su `0ed130af5`: i tre repro,
  **isolati uno per file**, danno `rc=134` sul baseline e `rc=0` con la patch. Tutte e **10 le RUN line**
  di `loop-unroll.mlir` passano. (Isolarli conta: eseguendo il file intero si misura tre volte lo stesso
  aborto, non le tre strade.)
- Regressione: 343 test scoperti (SCF 45, Affine 72, Vector 101, MemRef 33, Transforms 92), tutti verdi.
  L'unico XFAIL (`parallel-loop-invalid.mlir`) e' identico sul baseline, quindi preesistente.
- `clang-format` pulito sul file intero. Nessuna PR duplicata (ricontrollare comunque prima di aprire).

⚠️ **GATE APERTO il 19 ago 2026**: #216851 e' atterrata, quindi la riserva #203858 si puo' aprire. Prima di
aprirla pero' va rifatta la verifica: il ramo `scf-unroll-nonconstant-bounds` e' **284 commit indietro** e la
prova "ai due estremi allo stesso ref" descritta qui sotto vale per `da1fb5cf9`, non per main di oggi.

Regola: **non aprire la settima finche' una delle sei non atterra** ([[llvm-pr-register-short-and-staggered]]).
Il collo di bottiglia e' l'attenzione dei revisori, non la nostra produzione.

### Infrastruttura, per non ricostruirla

- Worktree locali: `/home/alepot55/.cache/llvmwt/wt-{mem2reg,scfdialect,coalesce,multired,vecscf}`,
  piu' `llvmsrc` stesso sul ramo `affine-licm-region-capture` (#216605). Base comune `d4e78d7f5`.
  ⚠️ `llvmsrc` e i worktree sono **sparse checkout**: `git apply --check` fallisce li' per file
  assenti, non per la patch.
- Modal: albero `main` = baseline pulita (`7cb5d8961`), albero `fix` = tutte le patch insieme
  (`$SCRATCH/combined4.patch`). Build: `scripts/modal_llvm.py build --tree fix --ref d4e78d7f5 --patch <p>`.
  `python3` del venv `gpufsm`, non quello di sistema.
- `clang-format` **c'e'**, in `~/Desktop/projects/gpufsm/.venv/bin/` (gli agenti non lo trovano).
  Uso: `git-clang-format --diff d4e78d7f5`.
- Monitor: `~/.cache/watch_upstream.sh` (6 PR) + tool `Monitor` che ne fa `tail -F` sul log, con
  guardia sull'eta' di `~/.cache/upstream_state`. Vedi [[never-poll-a-job-you-cant-see]].
- Triton **non e' nostra**: la segue un altro agente.

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

---

## 17-18 ago 2026 — la review si apre

Verificato con `gh api` il 18 ago 08:00 CEST.

| PR | diff | stato | ultima mossa |
|---|---|---|---|
| [#11324](https://github.com/triton-lang/triton/pull/11324) | +22/-1 | ✅ **MERGIATA** 20 ago 13:17Z, `37a4b78fc` | 4 richieste di Jokeren evase, poi approvata il 19 e mergiata dopo un sollecito |
| [#11325](https://github.com/triton-lang/triton/pull/11325) | +115/-3 | aperta | riscritta sull'obiezione di Jokeren |
| [#10766](https://github.com/triton-lang/triton/pull/10766) | +75/-1 | aperta, sbloccata | test di #9147 ripristinato |
| [#11323](https://github.com/triton-lang/triton/pull/11323) | — | **chiusa** | "this seems like a micro optimization" |
| [#11326](https://github.com/triton-lang/triton/issues/11326), [#11328](https://github.com/triton-lang/triton/issues/11328) | — | issue aperte | nessun triage |

### #11324 — quattro giri di review

Da +81/-4 a **+2/-1** di codice (+22/-1 col test). Ogni giro ha tolto qualcosa:

1. *"You can just include `WarpYieldOp` and `WarpReturnOp` in `containsLocalBarrier` and remove
   comments"* → la sua versione era **migliore e la nostra incompleta**: `WarpReturnOp` emette la
   stessa barriera e non la coprivamo. Effetto **raddoppiato**: 2016 → 1956 bar.sync (era 1986).
2. *"Why adding an empty new line?"* → rimossa, il secondo hunk sparisce.
3. *"Remove this test as well"* → rimosso, segnalando cosa restava scoperto.
4. *"your test checks nothing"* → **aveva torto**: mostrato l'IR prima/dopo dove la barriera sparisce.
   Ha risposto *"OK, let's add back the test ... and leave a comment before the test"*. Rimesso.

**Numeri corretti in pubblico**: i 2356/2326 pubblicati su #11323 erano mal contati. E il benchmark
era sul kernel sbagliato (`matmul_tma_ws_kernel`, dove la WS gira una volta per programma);
rifatto su `matmul_tma_persistent_ws_kernel`: sempre rumore (−0.48%..+0.26%).

### #11325 — riscritta tre volte

1. confronto di **forme** → mezza correzione: un `reinterpret` che cambia solo il tipo elemento
   lascia la forma uguale (trovato da un red-team, **verificato**: 0 barriere prima e dopo).
2. identita' del **valore** sorgente → Jokeren: *"I don't think this is the right fix"*, due
   `reinterpret` identici sono valori diversi → **falso positivo** su zone disgiunte. Confermato.
3. **tipo** della sorgente → corregge entrambi i bug **e** tiene la precisione nel suo caso, che e'
   ora un test. Costo: **0 barriere aggiunte** su 107 kernel (2203 = 2203).

### #10766 — perche' era ferma sei settimane

Due difetti nostri, non loro: la PR aveva **svuotato** il test di regressione di #9147 (4 `CHECK`
sostituite da `CHECK-NOT`), e il thread di review di peterbell10 era stato **chiuso da noi** senza
mai rispondergli — dal suo lato risultava sistemato.

Il 17 ago avevo anche scritto che *non era possibile* salvare il test. Falso: nome SSA invalido nel
mio IR di prova (`%0b`), errore su stderr soppresso. Il 18 ago, con un nome valido, funziona:
`arith.addf` fra `join` e `split` blocca il fold, le 4 verifiche tornano, passa con e senza la PR.
Vedi `docs/memory/empty-output-is-not-a-result.md`.
