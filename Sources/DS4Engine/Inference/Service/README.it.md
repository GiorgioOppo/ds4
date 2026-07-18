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

Prima di tokenizer, configurazione DeepSeek e Metal, l'inizializzatore passa da
[`RuntimeBackendFactory`](../../Runtime/README.md). Qwen viene riconosciuto ma
rifiutato esplicitamente come backend non ancora implementato; il decode
DeepSeek continua a usare il tipo concreto `StreamingDecoder`.

Prima del rendering, ogni campo fornito da utenti, storico, tool o API viene
neutralizzato rispetto ai token strutturali del GGUF. Solo il framing prodotto
da `ChatRenderer` può quindi diventare BOS, ruolo o delimitatore DSML atomico.

## Dipendenze

Dipende da `DS4Core`, `DS4Metal`, [`Persistence/KV`](../../Persistence/KV/README.md),
[`Agents`](../../Agents/README.md) e [`Tools`](../../Tools/README.md).

## Estensione

Conservare qui soltanto le responsabilità del ciclo di inferenza. Benchmark,
tuning e sub-agent hanno cartelle proprie. Ogni nuovo percorso deve gestire
cancellazione, limite contesto e transizioni di `kvDirty`.
