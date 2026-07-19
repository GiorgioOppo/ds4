[English](README.md) | **Italiano**

# Controller del server

`ServerController.swift` possiede lo stato rivolto all'utente di host, porta,
API key, CORS, avvio/arresto e status. Crea `LocalServer` solo dopo che il
motore Chat condiviso è pronto e mappa le callback del server in aggiornamenti
della UI sul main actor.

L'arresto è asincrono al confine del servizio. Mantieni il lease
`EngineActivityGate` del server finché `LocalServer.stop()` non ha drenato le
richieste accettate e il lavoro in background del motore; solo allora fai
uscire la UI dallo stato di arresto e rilascia il lease.

Mantieni l'implementazione di listener ed endpoint in `Services`, `Networking`
e `API`. Non caricare mai un modello da questo controller.
