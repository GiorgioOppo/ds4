# Inference/Tuning

Raccoglie e persiste il profilo di instradamento degli esperti usato per
scaldare la slot-cache e osservare hit/miss.

## Componente

`InferenceService+Tuning.swift` gestisce file di usage per coppia modello/agente,
seleziona un profilo iniziale, espone `TuningInfo`, `ModelInfo` e stima la memoria
della KV in base alla configurazione attiva.

## Flusso e dipendenze

Il profilo viene caricato durante l'inizializzazione del
[`Service`](../Service/README.md), aggiornato dalla generazione e salvato a fine
turno. I contatori provengono da `DS4Metal`; i file sono dati applicativi in
Application Support.

## Estensione

Versionare formati persistenti incompatibili, separare sempre profili di modelli
o agenti diversi e non usare queste statistiche per cambiare silenziosamente la
qualità numerica del modello.
