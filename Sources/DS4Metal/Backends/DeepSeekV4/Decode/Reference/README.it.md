# DeepSeekV4/Decode/Reference

Decoder leggibile e conservativo usato come oracolo di correttezza.

## File principali

- [`DSV4Decoder.swift`](DSV4Decoder.swift): decoder di riferimento con
  attention densa e `OutputHeadWeights` espliciti.

## Flusso e dipendenze

Carica le stesse forme e gli stessi pesi del backend ottimizzato, ma
privilegia passaggi diretti e verificabili. I test confrontano output
intermedi o logits per distinguere gli errori matematici dai problemi di
streaming/scheduling.

## Regole di modifica

Non introdurre qui ottimizzazioni che renderebbero il riferimento dipendente
dal percorso sotto verifica. Le modifiche matematiche devono derivare dalla
specifica del modello ed essere accompagnate da test con tolleranze
giustificate.
