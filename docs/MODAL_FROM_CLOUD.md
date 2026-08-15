# GPU da una sessione Claude cloud, via Modal (senza Remote Control)

Obiettivo: lanciare gli esperimenti GPU **da qualsiasi sessione Claude** (web, mobile, `claude --cloud`)
senza tenere accesa la macchina con la 4070 e senza pagare una GPU esterna. La sessione resta CPU-only
e fa da driver: il lavoro gira su una GPU affittata al secondo da **Modal** (free tier $30/mese).

RunPod resta documentato in `A100_RUNBOOK.md` (pod interattivo). Questo file copre il path Modal, che
è quello automatizzabile da un agente perché è tutto HTTPS.

## 1. Requisito bloccante: aprire l'egress verso Modal

Le sessioni cloud girano dietro un proxy di sicurezza. Con la policy di rete **Trusted** (il default)
`api.modal.com` è **bloccato**: il tunnel CONNECT risponde `403`. Verificato in sessione — vale anche
per le connessioni dirette, non c'è bypass.

Fix, una volta sola:

1. Su [claude.ai/code](https://claude.ai/code), clicca l'icona cloud col nome dell'ambiente (riga sopra
   il box del messaggio). Non esiste una pagina impostazioni né un URL diretto.
2. Passa sopra l'ambiente → icona ingranaggio → si apre il dialog con nome, network access, variabili
   d'ambiente e setup script.
3. **Network access**: `Full`, oppure `Custom` con `*.modal.com` e la casella *"Also include default
   list of common package managers"* spuntata (serve a tenere PyPI e GitHub).
4. **Environment variables** (formato `.env`, una coppia per riga), da
   [modal.com/settings/tokens](https://modal.com/settings/tokens):

   ```text
   MODAL_TOKEN_ID=ak-...
   MODAL_TOKEN_SECRET=as-...
   ```

5. **Setup script** (opzionale, evita di reinstallare a mano ogni volta):

   ```bash
   pip install 'modal[api-proxy-support]'
   ```

⚠️ Due avvertenze oneste:

- Le variabili vengono copiate **all'avvio della sessione**: dopo la modifica serve una **sessione
  nuova**, quella in corso non le rilegge.
- Il token Modal finisce dentro la sandbox, leggibile da qualunque comando della sessione. È un token
  con pieni poteri sul workspace Modal: se non ti va, tieni il path RunPod/locale e ruotalo quando vuoi.

L'extra `api-proxy-support` (tira dentro `python-socks` + `aiohttp-socks`) non è cosmetico: la sandbox
imposta `HTTPS_PROXY`, e senza quell'extra il client Modal non lo onora.

## 2. Preflight

```bash
python scripts/modal_gpu.py --preflight
```

Stampa PASS/FAIL su: modal installato, `api.modal.com` raggiungibile, credenziali presenti, supporto
proxy. Ogni FAIL riporta il comando esatto per sistemarlo. Da eseguire per primo in ogni sessione nuova.

## 3. Lanciare lavoro

`scripts/modal_gpu.py` è il runner generico: scegli GPU, passi i comandi, dichiari i file da riportare
indietro. Il container si spegne da solo a fine job.

```bash
# smoke test (~1 minuto di GPU, spicci)
python scripts/modal_gpu.py --gpu A100

# la validazione cross-arch, su H100 invece che A100
python scripts/modal_gpu.py --gpu H100 \
    --cmd "python experiments/cure/p3_cross_arch.py" \
    --fetch "paper2/data/cross_arch/*"

# più comandi in un solo affitto di GPU (paghi un solo avvio)
python scripts/modal_gpu.py --gpu A100-80GB \
    --cmd "python scripts/sweep_dfa.py" \
    --cmd "python scripts/regret_multiseed.py" \
    --fetch "paper/data/*.csv"
```

Opzioni: `--pip` (pacchetti dell'immagine, default `torch numpy triton`), `--timeout` (per comando,
default 1800 s), `--dry-run`. I file tornano scritti al loro path relativo nel repo, pronti da committare.
I file >8 MB non vengono riportati: sono elencati come `NOT fetched` invece di far fallire il job.

I job one-off già esistenti (`modal_a100.py`, `modal_cure_a100.py`, `modal_m5_decomp.py`) continuano a
funzionare senza modifiche una volta aperto l'egress; `modal_gpu.py` serve a non doverne scrivere un
quarto per ogni esperimento nuovo.

## 4. Costi

Free tier Starter: **$30/mese**, nessuna waitlist, si paga al secondo solo il tempo di GPU.

| GPU | ~$/h | ore/mese col free tier |
|---|---|---|
| T4 | ~1.1 | ~27 |
| L40S | ~2.0 | ~15 |
| A100 80GB | ~2.5 | ~12 |
| H100 SXM | ~4.3 | ~7 |

Metro di paragone: il run A100 cross-arch è costato ~$0.25 su RunPod, quindi il free tier ne copre
un centinaio. Il costo vero non è l'ora di GPU, è dimenticare acceso qualcosa — e qui non succede,
il container muore a fine funzione.

## Stato di verifica

Verificato in sessione cloud: il blocco `403` su `api.modal.com` (proxy e diretto), la costruzione
dell'app Modal, il preflight, il dry-run della CLI. **Non** verificato end-to-end: l'esecuzione remota
richiede l'egress aperto e i token, che stanno nelle impostazioni dell'account.
