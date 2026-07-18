# Inference

Questa area espone l'API applicativa e coordina il ciclo completo di inferenza.
L'actor `InferenceService` è l'unico proprietario dello stato mutabile del
decoder, della conversazione e della continuità KV.

## Struttura

- [`API`](API/README.md): DTO pubblici, parametri di sampling ed eventi.
- [`Service`](Service/README.md): caricamento, conversazione, prefill e decode.
- [`Benchmark`](Benchmark/README.md): misurazione e warm-up.
- [`Diagnostics`](Diagnostics/README.md): ispezione di tokenizer e template.
- [`Subagents`](Subagents/README.md): contesti isolati per lavori delegati.
- [`Tuning`](Tuning/README.md): profilo d'uso degli esperti e metriche cache.

La descrizione passo per passo è in [FLUSSO-INFERENZA.md](FLUSSO-INFERENZA.md).

## Dipendenze

`DS4Core` fornisce tokenizer, rendering, tool call e sampling; `DS4Metal`
fornisce runtime e `StreamingDecoder`. La persistenza KV è incapsulata in
[`Persistence/KV`](../Persistence/KV/README.md).

## Regole di estensione

- Aggiungere un DTO pubblico in `API` solo se deve attraversare il confine del
  servizio.
- Aggiungere una responsabilità coesa come estensione di `InferenceService`
  nella cartella pertinente.
- Non accedere al decoder fuori dall'isolamento dell'actor.
- Un'interruzione deve lasciare `kvDirty` coerente, così il turno seguente può
  ricostruire lo stato senza produrre output numericamente corrotto.
