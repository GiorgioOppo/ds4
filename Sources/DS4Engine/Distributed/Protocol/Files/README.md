# Protocol/Files

Definisce il trasferimento resumable di GGUF e sidecar.

## Tipi

- `DistFileEntry`: kind, nome, dimensione, SHA-256 e catena di checkpoint.
- `DistFileOffer`: manifest proposto dal coordinator.
- `DistFileNeed`: indici mancanti e offset validati di ripresa.
- `DistFileChunk` / `DistFileDone`: contenuto sequenziale e fine file.

`FILE_ACK` usa il formato ack condiviso dal flusso di controllo.

## Flusso e dipendenze

I metadati sono costruiti in [`../../Files`](../../Files/README.md), inviati dal
[`Coordinator`](../../Coordinator/README.md) e verificati in
[`Worker/Files`](../../Worker/Files/README.md).

## Estensione

Limitare numero di entry, dimensione dei chunk e lunghezze dei nomi/hash. Un
nuovo `Kind` deve specificare se è obbligatorio, come si calcola il percorso e
come viene attivato nell'assegnazione.
