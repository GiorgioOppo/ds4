# Worker/Serving

È il dispatcher principale dei frame ricevuti da una connessione coordinator.

## Flusso

`DistWorker+Serving.swift` invia `HELLO`, quindi gestisce offerta/ricezione file,
assegnazioni, comandi KV, richieste expert e `WORK`. Per la pipeline orizzontale
esegue la slice sotto `DistGate`, inoltra lo stato al worker seguente oppure
restituisce il risultato al return listener.

## Dipendenze

Compone tutte le aree del worker e usa [`Transport`](../../Transport/README.md)
e [`Protocol`](../../Protocol/README.md). Non implementa codec wire propri.

## Estensione

Ogni nuovo `MsgType` deve avere uno stato del lifecycle in cui è ammesso e una
risposta di errore deterministica. Validare sessione, assegnazione, slice,
posizione, contesto e shape prima di toccare il motore.
