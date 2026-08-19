# Verso l'esterno: niente em dash, e la lunghezza e' quella di chi legge

Il 19 ago 2026 l'utente ha corretto a mano un commento che avevo postato su
`triton-lang/triton#10766`, cambiando `Done — rewrote...` in `Done, rewrote...`. Chiarimento suo
subito dopo: **ha toccato solo l'em dash per fare in fretta, ma il problema vero era il registro.**

## 1. L'em dash: solo verso l'esterno

Vietato in tutto cio' che esce: commenti upstream, body delle PR, prosa dei paper. E' uno dei segnali
con cui si riconosce il testo generato, e su una repo di terzi la credibilita' e' l'unica valuta
([[llvm-pr-register-short-and-staggered]]).

**Internamente e' libero**: in questa conversazione e nei doc della repo si usa quello che si vuole.
L'utente lo ha detto esplicitamente. Non uniformare `MEMORY.md`, che usa l'em dash come separatore
dell'indice in 25 righe.

## 2. Il registro: e' qui che avevo sbagliato

Thomas aveva scritto nove parole: *"Looks good, can you write a meaningful description for this PR"*.
Gli ho risposto con un periodo di sessanta, che elencava tutto quello che avevo fatto.

Quello che andava scritto:

> Done. Rewrote it to stand alone as the commit message, dropped the review-thread material. Tell me
> if you meant something else.

Regola: **prima di postare, guardare la lunghezza dell'ultimo messaggio dell'interlocutore e stare
in quell'ordine di grandezza.** Elencare il proprio lavoro non e' informazione per chi legge, il
diff lo mostra gia'.

| chi | come scrive | come rispondergli |
|---|---|---|
| `ThomasRaoux` (Triton) | una riga, asciutto, zero convenevoli | una o due righe, il fatto |
| `FedericoBruzzone` (LLVM) | caldo, emoji, molti ringraziamenti | breve e cordiale, senza imitare le emoji |
| `Jianhui-Li` (LLVM) | nit tecnici secchi | conferma tecnica e basta |

Il contrario vale allo stesso modo: a un rilievo tecnico lungo e argomentato si risponde nel merito,
non con un "done". La brevita' non e' mai un motivo per tacere un fatto che cambierebbe la decisione
di chi legge ([[verify-by-running-not-by-agent-verdict]]).

## Seconda botta, lo stesso giorno

Dopo aver registrato la lezione qui sopra ho riscritto la descrizione di `triton#10766` portandola
da 4900 a 3205 caratteri, con tabella e quattro sezioni, e l'ho considerata "concisa". Thomas ha
risposto:

> please make a reasonable description that is meaningful and concise. If you are asking for
> reviewers to spend time you need to also show that you are not sending random patches.

Due cose da capire in quella frase. La prima e' che **3205 caratteri non sono concisi**: il mio
metro era ancora sbagliato di un fattore tre, la versione buona sta a 880. La seconda e' il
sottotesto, che e' piu' importante: **una descrizione gonfia non legge come completezza, legge come
patch generata in serie.** In un momento in cui i maintainer sono sommersi di contributi automatici,
il volume di prosa e' un segnale negativo, non una prova di lavoro.

Regola operativa: una descrizione di PR dice **cosa cambia, perche' conta con un numero, quale
vincolo non ovvio ha il codice, dove sono i test.** Tutto il resto va tolto. Se serve una tabella,
probabilmente il posto e' il thread di review, non il corpo della PR, che diventa il messaggio di
commit permanente.
