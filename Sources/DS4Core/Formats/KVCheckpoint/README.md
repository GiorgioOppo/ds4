# Formats/KVCheckpoint

Formato su disco per salvare e riutilizzare lo stato KV del decoder.

## File principali

- [`KVCFile.swift`](KVCFile.swift): header, flag di estensione, naming SHA-1,
  punteggio di eviction e codifica/decodifica del payload.
- `DSV4PayloadHeader`: descrive la forma specifica dello snapshot DeepSeek-V4.

## Flusso e dipendenze

Il livello di persistenza converte un `KVSnapshot` Metal in payload e header,
scrive il checkpoint e in seguito lo valida prima del ripristino. Qui è definito
solo il contratto binario: policy della cache e I/O orchestrato appartengono a
`DS4Engine`.

## Regole di modifica

Mantenere magic, versione, flag e ordine dei campi compatibili. Verificare forma
e dimensione prima di allocare o decodificare. Un'estensione del payload deve
usare un flag/versione riconoscibile dai lettori precedenti.
