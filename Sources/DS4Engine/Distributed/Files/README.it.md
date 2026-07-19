[English](README.md) | **Italiano**

# Distributed/Files

Gestisce identità e conservazione locale dei grandi artefatti distribuiti.

## Componenti

- `DistFileHash`: SHA-256 completo, catena di checkpoint e cache persistente
  validata con dimensione e data di modifica.
- `DistFileStore`: directory gestita, sanitizzazione nomi, manifest e verifica
  rapida dei file già ricevuti.

## Flusso e dipendenze

Il coordinator calcola una volta hash e catena; il worker usa il manifest per
evitare trasferimenti già verificati. I messaggi corrispondenti sono in
[`Protocol/Files`](../Protocol/Files/README.it.md), la ricezione in
[`Worker/Files`](../Worker/Files/README.it.md). Dipende da Foundation e CryptoKit.

## Estensione

Non fidarsi di nomi inviati in rete, non promuovere `.part` senza hash finale e
invalidare la cache quando cambiano dimensione o mtime. Un nuovo sidecar richiede
un nuovo `Kind` nel protocollo e una politica esplicita di risoluzione.
