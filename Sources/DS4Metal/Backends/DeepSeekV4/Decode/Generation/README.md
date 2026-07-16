# DeepSeekV4/Decode/Generation

Operazioni che trasformano token in input del decoder e stato finale in logits.

## File principali

- [`StreamingDecoder+Generation.swift`](StreamingDecoder+Generation.swift):
  embedding, normalizzazione/output head e primitive usate dal ciclo di
  generazione.

## Flusso e dipendenze

L'id del token seleziona la riga di embedding; il risultato entra nel forward.
All'ultimo layer, output norm e matrice di output producono logits leggibili dal
[`Sampler`](../../../../../DS4Core/Generation/README.md). Il ciclo e le policy di stop
sono orchestrati da `DS4Engine`.

## Regole di modifica

Mantenere coerenti dimensione vocabolario, dtype e scala dell'output. Evitare
readback intermedi: solo i logits necessari devono attraversare il confine GPU/CPU.
