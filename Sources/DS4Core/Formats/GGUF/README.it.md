[English](README.md) | **Italiano**

# Formats/GGUF

Lettura validata e a copia zero dei modelli GGUF.

## File principali

- [`GGUFTypes.swift`](GGUFTypes.swift): costanti, tipi valore, informazioni di
  quantizzazione ed errori del formato.
- [`GGUFCursor.swift`](GGUFCursor.swift): cursore interno per letture binarie con
  controllo dei limiti.
- [`GGUFModel.swift`](GGUFModel.swift): apre e mappa il file, indicizza metadati e
  tensor descriptor, espone viste e suggerimenti di prefetch.
- [`GGUFWriter.swift`](GGUFWriter.swift): l'inverso del reader — serializza un
  file GGUF v3 da metadati tipizzati ordinati (`GGUFMetadataValue`) e tensori,
  scrivendo i payload uno alla volta così che i modelli grandi non debbano
  stare in memoria tutti insieme.
- [`GGUFModel+Export.swift`](GGUFModel+Export.swift): helper di ri-lettura
  (`allMetadata`, `tensorData`) che passano un modello caricato al writer,
  chiudendo il round-trip lettura -> modifica -> scrittura.

## Flusso

L'inizializzazione verifica header e tabelle, calcola gli offset assoluti e
mantiene una sola mappatura. Tokenizer e loader pesi interrogano metadati e
tensori senza copiare l'intero modello; le pagine vengono materializzate dal
page cache del sistema quando accedute.

## Regole di modifica

Ogni aritmetica derivata dal file deve controllare overflow e intervalli. Non
trasformare le viste mmap in copie implicite. Nuovi tipi GGUF richiedono test con
file validi, troncati e metadati non riconosciuti.
