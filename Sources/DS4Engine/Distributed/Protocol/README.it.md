# Distributed/Protocol

Raccoglie tutti i tipi serializzati sul wire. Le strutture non aprono socket e
non conoscono lo stato di coordinator o worker.

## Aree

- [`Core`](Core/README.md): versione, tipi messaggio, flag e limiti.
- [`Framing`](Framing/README.md): header comune dei frame.
- [`Serialization`](Serialization/README.md): primitive little-endian.
- [`Handshake`](Handshake/README.md): identità e assegnazione.
- [`Files`](Files/README.md): offerta e trasferimento artefatti.
- [`KV`](KV/README.md): controllo dei checkpoint shard.
- [`Work`](Work/README.md): stati hidden, route e risultati.
- [`Experts`](Experts/README.md): parallelismo verticale MoE.
- [`Codec`](Codec/README.md): compressione delle attivazioni.

La sequenza completa è in [`../PROTOCOLLO.md`](../PROTOCOLLO.md).

## Regole

Ogni `encoded()` deve avere un `decode` simmetrico che controlla tutti i limiti
prima di costruire il valore. Non usare `MemoryLayout` come formato wire:
campi, ordine e endianess devono essere espliciti. Una modifica incompatibile
richiede il bump della versione in `Core/Dist.swift`.
