# DeepSeekV4/Decode/Diagnostics

Misurazioni aggregate del percorso caldo di inferenza.

## File principali

- [`DecodeProfile.swift`](DecodeProfile.swift): accumula i tempi di embedding,
  route/attenzione, gather degli esperti, FFN e output head; registra
  hit/miss e byte di I/O e produce un report per token.

## Flusso e dipendenze

Il decoder aggiorna il profilo ai confini dei command buffer e delle
operazioni di I/O. Con `DS4_PROFILE_ROUTE=1` scompone ulteriormente la fase di
routing; il servizio legge il report alla fine del turno. Non è telemetria
globale persistente.

## Regole di modifica

Dichiara se una misurazione include le attese GPU o solo l'encoding su CPU.
Evita sincronizzazioni extra nel profilo normale; i contatori diagnostici non
devono cambiare i risultati numerici e devono essere interpretati sullo stesso
numero di forward.
