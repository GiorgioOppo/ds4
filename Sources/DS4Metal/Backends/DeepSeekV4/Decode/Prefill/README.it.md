# DeepSeekV4/Decode/Prefill

Ingestione efficiente del prompt con elaborazione layer-major e batching.

## File principali

- [`StreamingDecoder+Prefill.swift`](StreamingDecoder+Prefill.swift): suddivide i
  token in chunk, orchestra le fasi e sceglie i percorsi FFN/matrix-matrix.
- [`PrefillStage.swift`](PrefillStage.swift): staging buffer per un gruppo di token.
- [`PrefillGather.swift`](PrefillGather.swift): gather expert in background con
  consegna sincronizzata al chiamante.

## Flusso

Ogni chunk viene trasformato in uno staging contiguo; per ciascun layer si
calcolano route e unione degli expert, sovrapponendo I/O del gruppo successivo e
FFN GPU corrente. `DS4_PREFILL_CHUNK`, `DS4_PREFILL_ROUTE_BATCH`,
`DS4_PREFILL_FFN_BATCH` e `DS4_PREFILL_MM` controllano le varianti documentate
nella configurazione principale.

## Regole di modifica

Il risultato deve essere equivalente a una sequenza di forward singoli. Un
worker iniziato deve essere sempre atteso anche nei percorsi di errore. Limitare
la memoria temporanea al chunk e misurare separatamente token/s, picco RAM e
pressione SSD.
