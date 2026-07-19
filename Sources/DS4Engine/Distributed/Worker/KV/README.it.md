[English](README.md) | **Italiano**

# Worker/KV

Gestisce i frame di query, restore e salvataggio della cache KV dello shard.

## Componente e flusso

`DistWorker+KV.swift` decodifica la richiesta tramite
[`Protocol/KV`](../../Protocol/KV/README.it.md), delega a `DistEngine` e risponde
con lunghezze o ack. I checkpoint sono separati per modello e intervallo di
layer, così uno shard non può ripristinare dati di un'altra assegnazione.

## Estensione

Eseguire import/export sotto il gate di calcolo, mantenere l'I/O persistente
streaming e restituire un fallimento esplicito quando il motore non è pronto o
il prefisso non coincide esattamente.
