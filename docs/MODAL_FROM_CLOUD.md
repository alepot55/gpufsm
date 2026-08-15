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
4. **Setup script** — gira a ogni avvio di sessione, che è esattamente ciò che serve: il container è
   effimero, quindi `~/.modal.toml` va riscritto ogni volta. Due righe, in quest'ordine:

   ```bash
   pip install -q 'modal[api-proxy-support]'
   modal token set --token-id ak-... --token-secret as-... --no-verify
   ```

   In alternativa al comando `modal token set`, le due env var `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`
   nel campo **Environment variables** (formato `.env`) fanno la stessa cosa: il client legge entrambe
   le forme. I token si generano su [modal.com/settings/tokens](https://modal.com/settings/tokens).

Perché così:

- **`api-proxy-support`** (tira dentro `python-socks` + `aiohttp-socks`): la sandbox impone
  `HTTPS_PROXY` e senza quell'extra il client Modal non lo onora — solleva direttamente un
  `ImportError` che lo dice.
- **`--no-verify`**: il default di `modal token set` è `--verify`, che fa una chiamata di rete vera. In
  un setup script è fragile (la sessione che parte con la rete non ancora aperta fallirebbe lo script);
  la verifica la fa `--preflight`, con una diagnosi migliore.
- **La CA del proxy non va messa qui.** Serve (vedi sotto), ma `/root/.ccr/agent-proxy-ca.crt` **non
  esiste ancora** quando gira il setup script — il proxy viene materializzato dopo, e un `cat` su quel
  path fa fallire il setup con `No such file or directory`. Ci pensa `scripts/modal_gpu.py`, che se la
  aggiunge da solo (idempotente, solo se `HTTPS_PROXY` è impostato e il file esiste).

### La trappola della CA

Il proxy ri-termina il TLS, e Modal costruisce il suo SSL context da `certifi.where()` sia sul canale
gRPC (via grpclib) sia sul client blob: `SSL_CERT_FILE`, che l'ambiente imposta già, viene **ignorato**.
Senza la CA dentro il bundle certifi si prende `CERTIFICATE_VERIFY_FAILED: self-signed certificate in
certificate chain` anche con l'egress aperto. Verificato su una connessione reale, prima e dopo.

Se serve farlo a mano (fuori da `modal_gpu.py`), da dentro la sessione:

```bash
cat /root/.ccr/agent-proxy-ca.crt >> "$(python -c 'import certifi;print(certifi.where())')"
```

⚠️ Due avvertenze oneste:

- La configurazione dell'ambiente viene letta **all'avvio della sessione**: dopo la modifica serve una
  **sessione nuova**, quella in corso non la rilegge.
- Il token Modal finisce dentro la sandbox, leggibile da qualunque comando della sessione (e in chiaro
  nel setup script). È un token con pieni poteri sul workspace Modal: se non ti va, tieni il path
  RunPod/locale e ruotalo quando vuoi.

## 2. Preflight

```bash
python scripts/modal_gpu.py --preflight
```

Stampa PASS/FAIL su: modal installato, `api.modal.com` raggiungibile, credenziali presenti, supporto
proxy, CA del proxy dentro il bundle certifi. Ogni FAIL riporta il comando esatto per sistemarlo. Da
eseguire per primo in ogni sessione nuova.

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
dell'app Modal, il preflight (entrambi gli esiti del check CA), il dry-run della CLI, e il fix della CA
misurato su una connessione reale attraverso il proxy (`CERTIFICATE_VERIFY_FAILED` prima, handshake OK
dopo). **Non** verificato end-to-end: l'esecuzione remota richiede l'egress aperto e i token, che stanno
nelle impostazioni dell'account.
