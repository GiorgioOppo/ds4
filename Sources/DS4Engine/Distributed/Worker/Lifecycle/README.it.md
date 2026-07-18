# Worker/Lifecycle

Possiede il ciclo di vita del listener e lo stato di sessione del worker.

## Componente

`DistWorker+Lifecycle.swift` crea la directory dello shard, persiste l'usage,
avvia/ferma `NWListener`, accetta connessioni, costruisce `HELLO` e risolve il
modello locale o trasferito.

## Flusso e dipendenze

`start` apre il listener; ogni connessione passa a
[`Serving`](../Serving/README.md). `admit` impedisce che turni concorrenti
resettino la KV attiva. `stop` chiude listener, task e risorse persistenti.

## Estensione

Start e stop devono essere idempotenti. Non mantenere una sessione dopo errori
terminali e non usare un percorso modello ricevuto senza fallback al file
sanitizzato nell'archivio gestito.
