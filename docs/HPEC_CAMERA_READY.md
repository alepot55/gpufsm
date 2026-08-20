# HPEC 2026 — accettato, e cosa resta da fare

**Paper 133**, *The Two Faces of Abstraction Regret*. Esito CMT: **`Accept-Oral-Xplore`**
(notifica 19 ago 2026). Presentazione **orale** e pubblicazione su **IEEE Xplore**.

## Lo slot

**Mercoledi 16 settembre**, sessione **3-3: General Purpose GPU Computing**, 14:15–15:30 EDT,
cioe' **20:15–21:30 ora italiana**. Conferenza interamente **virtuale**. Cinque paper in sessione:
il nostro e' l'ultimo dell'elenco, quindi circa **15 minuti** a testa piu' domande.

## Scadenze

| cosa | quando | chi |
|---|---|---|
| Camera-ready su CMT | **4 set 2026** | serve la firma dell'utente |
| IEEE Copyright Form | insieme al camera-ready | **solo l'utente** |
| Registrazione al tasso ridotto | **entro il 4 set** | **solo l'utente**, e' un pagamento |
| Rimborso non piu' possibile | dopo il 1 set | — |

Tariffe: studente IEEE/SIAM **$140**, studente non socio **$180** (dopo il 4 set: $180 / $220).
**L'autore che presenta deve pagare la quota piena**: non e' opzionale.

## Le due review

**Reviewer #1 — Accept.** Very Good, **Very Novel**, High Confidence. Un solo rilievo: *"NFA and
DFA seem to be used without introduction"*. Vero: non erano espansi in nessun punto del paper.

**Reviewer #3 — Neutral.** Good, Reasonably Novel, Good Confidence. Tre rilievi, il secondo di
sostanza:

1. **Troppo denso**, i risultati chiave restano sepolti. Suggerisce di snellire su uno o due
   messaggi, oppure di espandere in un articolo di rivista.
2. **Obiezione di metodo**: i kernel "structurally mirrored line-for-line" non sarebbero un
   confronto equo, perche' la struttura ottima in un linguaggio non e' quella ottima in un altro.
   Chiede una giustificazione esplicita.
3. **Predittivita'**: si puo' sapere in anticipo se un DSL raggiungera' CUDA, senza prima scrivere
   il CUDA?

## Cosa e' gia' stato fatto nel `.tex`

- **R1**: `NFA` e `DFA` espansi alla prima occorrenza, sia nell'abstract sia nel corpo.
- **R3.2**: aggiunta la giustificazione del protocollo. L'argomento e' che il regret e' *definito*
  ad algoritmo fisso, quindi lasciare che ogni DSL scelga il proprio misurerebbe la riprogettazione
  algoritmica, non cio' che l'astrazione preclude; e la messa a punto per linguaggio **non** e'
  esclusa, perche' dentro ogni DSL variamo la *tecnica*. Il limite e' di scopo, non di parzialita'.
- **R3.3**: reso esplicito che la mappa capability→cost si applica **prima** che il port esista, e
  che il CUDA a mano *calibra* il regret invece di rilevarlo. La risposta era gia' nel paper, ma
  sepolta: il fatto che il revisore l'abbia persa e' una conferma del rilievo 1.
- Compattato il rimando all'artifact per rientrare nel limite.

**Vincolo di pagine, verificato:** il limite e' **6 pagine di corpo**, referenze escluse, e una
pagina extra costa $200. Dopo le modifiche il corpo occupa **esattamente 6 pagine**: su pagina 7
resta solo l'intestazione `REFERENCES`. Build pulita, **0 overfull box**, nessun riferimento non
definito. I 12 *underfull* sono cosmetici e c'erano anche prima.

## Cosa resta aperto

- **R3.1, la densita'**: non affrontato. In 6 pagine non si snellisce senza tagliare un risultato,
  ed e' una scelta che spetta all'utente. La strada suggerita dal revisore, la versione estesa da
  rivista, esiste gia' come `paper/gpufsm.tex` (8 pagine).
- **Caricare il camera-ready** su CMT, firmare il **Copyright Form**, **registrarsi**: tutte azioni
  a nome dell'utente, nessuna delle quali va fatta da un agente.
- **Preparare il talk**: ~15 minuti, 16 set sera ora italiana.
