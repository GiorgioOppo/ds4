# Inference/Service

Implementa l'actor centrale che carica il modello e serializza ogni operazione
sul decoder.

## File principali

- `InferenceService.swift`: dipendenze, stato, executor e inizializzazione.
- `InferenceService+Conversation.swift`: rendering, turni e continuità KV.
- `InferenceService+Generation.swift`: prefill, sampling, stream e checkpoint.
- `InferenceService+Agents.swift`: selezione del profilo agente.

## Flusso

Il chiamante configura conversazione e strumenti, quindi avvia `send`,
`sendWithHistory` o `complete`. Il servizio rende i token, riusa o ricostruisce
la KV, esegue prefill e decode, e restituisce un `AsyncThrowingStream<GenEvent>`.
Il flusso dettagliato è in [`../FLUSSO-INFERENZA.md`](../FLUSSO-INFERENZA.md).

## Dipendenze

Dipende da `DS4Core`, `DS4Metal`, [`Persistence/KV`](../../Persistence/KV/README.md),
[`Agents`](../../Agents/README.md) e [`Tools`](../../Tools/README.md).

## Estensione

Conservare qui soltanto le responsabilità del ciclo di inferenza. Benchmark,
tuning e sub-agent hanno cartelle proprie. Ogni nuovo percorso deve gestire
cancellazione, limite contesto e transizioni di `kvDirty`.
