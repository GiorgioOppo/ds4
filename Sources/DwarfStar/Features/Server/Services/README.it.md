# Servizi del Server

`LocalServer.swift` possiede il listener `Network.framework`, il riferimento
all'engine condiviso, la configurazione del server e lo stato comune di
routing. Estensioni mirate nelle directory sorelle `Networking/` e `API/`
implementano il comportamento di trasporto e degli endpoint.

Il servizio è `@unchecked Sendable`; la sincronizzazione deve quindi restare
esplicita. Lo stato del listener e ogni task di connessione/richiesta
accettata sono posseduti sotto un unico lock di ciclo di vita. `stop()` è una
barriera async: chiude il listener, cancella e attende le richieste accettate,
poi mette in quiescenza il lavoro GPU/I/O dell'engine. Preserva
quell'ordinamento, la validazione della API key e la serializzazione a engine
singolo quando modifichi lo stato condiviso.
