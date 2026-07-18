# Model/Quantization

Descrizione dei formati quantizzati accettati dai kernel MoE.

## File principali

- [`MoEQuant.swift`](MoEQuant.swift): enum delle quantizzazioni supportate e
  proprietà derivate, tra cui dimensione blocco, byte per riga e parametri di
  dispatch.

## Flusso e dipendenze

Il loader traduce il tipo GGUF in `MoEQuant`; gather, cache e kernel usano lo
stesso valore per calcolare intervalli gate/up/down e scegliere la pipeline.
Le conversioni CPU risiedono in
[`DS4Core/Formats/Quantization`](../../../DS4Core/Formats/Quantization/README.md).

## Regole di modifica

Un nuovo caso richiede supporto coordinato nel parser GGUF, calcolo layout,
wrapper Swift e kernel `.metal`. Non dedurre il tipo dal solo rapporto byte/elementi;
fallire su combinazioni non validate.
