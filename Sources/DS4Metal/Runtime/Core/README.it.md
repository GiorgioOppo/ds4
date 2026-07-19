[English](README.md) | **Italiano**

# Runtime/Core

Astrazioni minime sopra Metal e memoria unificata Apple Silicon.

## File principali

- [`MetalRuntime.swift`](MetalRuntime.swift): seleziona il device, crea queue e
  libreria, concatena i kernel incorporati e memorizza le compute pipeline.
- [`GPUTensor.swift`](GPUTensor.swift): buffer con lunghezza logica, byte offset,
  viste zero-copy, allocazioni residenti/mmap e lock best-effort delle pagine.

## Flusso

Il runtime viene creato una volta per decoder. Loader e scratch costruiscono
`GPUTensor`; i wrapper kernel associano `buffer` e `byteOffset` agli encoder. Le
viste mmap condividono il page cache, mentre subview e staging condividono un
buffer Metal già allocato.

## Regole di modifica

Ogni binding deve rispettare `byteOffset`, non solo il buffer base. Usare buffer
non inizializzati solo quando la scrittura copre l'intero intervallo prima della
lettura. Non trattenere puntatori CPU oltre la vita del buffer/mmap e non creare
una pipeline nel percorso per-token se può essere memorizzata.
