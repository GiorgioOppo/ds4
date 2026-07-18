# Formats

Formati binari e conversioni numeriche condivisi dal caricamento e dalla
persistenza del modello.

## Struttura

- [`GGUF/`](GGUF/README.md): parser mmap del file modello e dei metadati.
- [`KVCheckpoint/`](KVCheckpoint/README.md): formato persistente della KV cache.
- [`Quantization/`](Quantization/README.md): f16/f32 e Q8_0 -> Q4_K su CPU.

## Flusso e dipendenze

`GGUFModel` valida e mappa il modello; `DS4Metal` usa i descrittori risultanti per
creare viste sui pesi. `KVCFile` serializza lo stato ricorrente fuori dal percorso
GGUF. Le conversioni CPU sono impiegate nella preparazione delle cache
quantizzate. Tutta la cartella rimane indipendente da Metal.

## Regole di modifica

Validare sempre limiti, overflow, allineamenti ed endianess prima di leggere.
Non cambiare layout persistenti senza versione o migrazione. Evitare copie dei
payload di grandi dimensioni nel percorso GGUF.
