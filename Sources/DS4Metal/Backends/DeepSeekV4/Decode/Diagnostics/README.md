# DeepSeekV4/Decode/Diagnostics

Misure aggregate del percorso caldo di inferenza.

## File principali

- [`DecodeProfile.swift`](DecodeProfile.swift): accumula tempi di embedding,
  route/attention, gather expert, FFN e output head; registra hit/miss, byte I/O
  e produce un report per token.

## Flusso e dipendenze

Il decoder aggiorna il profilo ai confini di command buffer e operazioni I/O.
Con `DS4_PROFILE_ROUTE=1` scompone ulteriormente la fase di routing; il servizio
legge il report al termine del turno. Non è una telemetria globale persistente.

## Regole di modifica

Specificare se una misura include attese GPU o solo encoding CPU. Evitare
sincronizzazioni aggiuntive nel profilo normale; i contatori diagnostici non
devono cambiare i risultati numerici e vanno interpretati sullo stesso numero di
forward.
