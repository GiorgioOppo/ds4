[English](README.md) | **Italiano**

# Protocol/KV

`DistKV.swift` serializza i messaggi di continuità KV per ogni slice.

## Operazioni

- encode/decode delle liste di token per query e restore;
- lista delle lunghezze di prefisso disponibili;
- richiesta di salvataggio con indicazione cold/continued;
- ack con esito e messaggio diagnostico.

## Flusso e dipendenze

Il coordinator interseca le lunghezze restituite da tutti i worker e sceglie un
prefisso comune; ogni worker inoltra l'operazione al proprio `DistEngine` e
[`DiskKVStore`](../../../Persistence/KV/README.it.md).

## Estensione

Un restore è valido soltanto se tutti gli shard confermano la stessa frontiera.
Imporre limiti al numero di token e non trattare un ack parziale come successo
globale.
