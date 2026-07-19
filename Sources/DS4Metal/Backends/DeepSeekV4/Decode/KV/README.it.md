[English](README.md) | **Italiano**

# DeepSeekV4/Decode/KV

Esportazione e ripristino dello stato ricorrente del decoder.

## File principali

- [`KVSnapshot.swift`](KVSnapshot.swift): `CompSnapshot`, `KVLayerSnapshot` e
  `KVSnapshot`, più le estensioni di `StreamingDecoder` per capture/restore.
- `KVSnapshotError`: segnala incompatibilità di forma durante il ripristino.

## Flusso

Il decoder copia in uno snapshot CPU finestra raw KV, righe compresse e stato
dell'indexer. `DS4Engine` può conservarlo in memoria o codificarlo con
[`KVCFile`](../../../../../DS4Core/Formats/KVCheckpoint/README.it.md), quindi ripristinarlo
in un decoder con la stessa architettura.

## Regole di modifica

Uno snapshot deve essere autoconsistente e indipendente dai buffer temporanei.
Validare layer, dimensioni, contatori e capacità prima di scrivere sulla GPU.
Aggiornare insieme snapshot e formato KVC quando cambia lo stato persistito.
