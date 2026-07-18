# Controller distribuiti

`DistributedController.swift` possiede il ciclo di vita di coordinator e
worker, la configurazione delle connessioni, i log e il riferimento al
coordinator condiviso con Chat e Benchmark. Adatta le API distribuite di
`DS4Engine` allo stato della UI sul main actor.

Mantieni il protocollo di rete, il framing, il trasferimento dei file e la
logica di esecuzione in `DS4Engine/Distributed`. Le modifiche qui devono
limitarsi a coordinare quelle API e devono preservare una pulizia ordinata di
arresto/disconnessione.
