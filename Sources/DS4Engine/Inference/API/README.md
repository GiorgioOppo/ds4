# Inference/API

Contiene i tipi pubblici e `Sendable` usati dai client del motore.

## Tipi principali

- `ChatRole`: ruolo logico di un turno.
- `DS4ThinkMode`: modalità di reasoning esposta dall'applicazione.
- `SamplingParams`: temperatura, top-k/top-p/min-p, seed e penalità ripetizione.
- `ModelInfo`: descrizione sintetica del modello caricato.
- `GenEvent`: stream di reasoning, testo, tool call e avanzamento.
- `InferenceError`: errori applicativi presentabili al chiamante.

## Dipendenze e flusso

I tipi dipendono da Foundation e, dove necessario, dai modelli portabili di
`DS4Core`. Sono prodotti da [`Service`](../Service/README.md) e consumati da
GUI, server e benchmark senza esporre oggetti Metal.

## Estensione

Mantenere i tipi indipendenti da SwiftUI e da `Metal`. Un nuovo evento deve
avere semantica chiara nello stream e tutti i consumer devono gestirlo in modo
esplicito; evitare di usare stringhe di stato quando serve un caso enum.
