# Niente em dash, e scrivi nel registro di chi ti legge

Il 19 ago 2026 l'utente ha corretto a mano un commento che avevo postato su
`triton-lang/triton#10766`. L'unica modifica: `Done — rewrote the description...` diventa
`Done, rewrote the description...`. Poi la richiesta esplicita: **non usare em dash, e adattarsi
allo stile di comunicazione dell'interlocutore.**

## 1. L'em dash non va usato, punto

Il divieto era gia' in `CLAUDE.md` ma limitato alla prosa dei paper. **Vale ovunque**: commenti
upstream, messaggi di PR, risposte all'utente, note come questa. Sostituti: virgola, due punti,
parentesi, oppure due frasi. Non e' una preferenza estetica, e' una firma: l'em dash e' uno dei
segnali con cui si riconosce il testo generato, e su una repo di terzi la nostra credibilita' e'
l'unica valuta ([[llvm-pr-register-short-and-staggered]]).

## 2. Il registro e' quello di chi legge, non il nostro

Su una PR non si scrive con un registro solo. Osservato sugli stessi thread:

| chi | come scrive | come rispondergli |
|---|---|---|
| `ThomasRaoux` (Triton) | una riga, minuscolo, nessun convenevole: *"are there practical use cases?"*, *"Looks good, can you write a meaningful description for this PR"* | una riga, il fatto, stop |
| `FedericoBruzzone` (LLVM) | caldo, emoji, molte grazie: *"Thanks a lot for all of your patches :D"* | breve e cordiale, senza imitare le emoji |
| `Jianhui-Li` (LLVM) | nit tecnici asciutti | conferma tecnica e basta |

Il mio default e' denso e formale, ed e' sbagliato quasi sempre: a un maintainer che scrive sei
parole non si risponde con un paragrafo strutturato. **Prima di postare, rileggere l'ultimo
messaggio dell'interlocutore e scendere al suo livello di lunghezza.** Vale anche per il contrario:
a un rilievo tecnico dettagliato si risponde nel merito, non con un "done".

Corollario che resta valido: la brevita' non e' un motivo per omettere un fatto che cambierebbe la
decisione di chi legge ([[verify-by-running-not-by-agent-verdict]]).
