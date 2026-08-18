# "Chi ha parlato per ultimo" non dice di chi è la palla

Il 18 ago 2026, per capire in fretta su quali PR upstream fossimo in debito di una risposta, ho usato
come criterio *l'autore dell'ultimo messaggio*: se non era `alepot55`, la palla era nostra. Su tre PR
LLVM segnalate così, **tre erano falsi positivi**. Leggendo il merito:

- **#216851** — l'ultimo messaggio di FedericoBruzzone era *"I'd wait a reasonable amount of time to
  ensure @Jianhui-Li has seen it before landing"*. Non una richiesta: un'attesa. La PR aveva già due
  approvazioni e CI verde.
- **#216854** — l'ultimo era *"I left the comment as **non-blocking** just to leave a record"*. Il
  rilievo era esplicitamente marcato come non bloccante, e il nit era già stato applicato.
- **#216853** — l'ultimo era un ringraziamento più *"I requested the review, don't hesitate to ping me
  next week"*. La palla era ai revisori richiesti, non a noi.

Il criterio giusto costa una lettura in più e suona così: **è aperta solo una domanda o un
cambiamento richiesto da un umano, a cui non si è replicato dopo, e che non è marcato non-blocking.**
Un ringraziamento, un'attesa di un terzo revisore e un rilievo non bloccante chiudono il turno, non lo
aprono.

Vale la pena tenerlo a mente perché il costo è asimmetrico e invisibile: un falso positivo qui non
produce un errore, produce una **bozza di risposta a una domanda che nessuno ha fatto** — cioè
rumore mandato a un revisore che sta già facendo il suo lavoro, sulla repo di terzi dove la nostra
reputazione è l'unica valuta ([[llvm-pr-register-short-and-staggered]]).

Stessa famiglia di [[verify-by-running-not-by-agent-verdict]]: la scorciatoia che *sembra* un segnale
non è un segnale. Lì era il verdetto di un agente, qui è un timestamp.
