[English](README.md) | **Italiano**

# Generation

Selezione CPU del token successivo a partire dai logits.

## File principali

- [`Sampler.swift`](Sampler.swift): argmax, RNG xorshift64*, temperature,
  top-k, top-p, min-p e penalità di ripetizione.

## Flusso e dipendenze

Il backend produce i logits; il servizio passa parametri, token recenti e stato
RNG a `Sampler.sample`; l'id scelto torna al tokenizer e al ciclo di generazione.
L'implementazione usa soltanto Swift/libm ed è indipendente dalla GPU.

## Regole di modifica

La riproducibilità con lo stesso seed e la parità con il riferimento C sono
requisiti funzionali. Conservare l'ordine dei candidati e il comportamento dei
fallback per logits non finiti; accompagnare nuove strategie con test statistici
e deterministici separati.
