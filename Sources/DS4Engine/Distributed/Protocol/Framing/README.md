# Protocol/Framing

`DistFrameHeader.swift` definisce l'involucro di ogni messaggio: magic, tipo e
numero di byte del payload.

## Flusso e dipendenze

[`Transport`](../../Transport/README.md) legge prima l'header, valida magic e
tipo e poi richiede esattamente la lunghezza dichiarata. La serializzazione usa
le primitive di [`Serialization`](../Serialization/README.md).

## Estensione

L'header deve restare piccolo e deterministico. Un cambiamento di layout è
sempre incompatibile e richiede bump di versione più test per frame vuoti,
troncati, sovradimensionati e con magic errato.
