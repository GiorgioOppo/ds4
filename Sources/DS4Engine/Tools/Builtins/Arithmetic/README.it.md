# Builtins/Arithmetic

Strumenti locali, deterministici e senza side effect esterni.

## Tool

- `Add.swift`, `Subtract.swift`, `Multiply.swift`: operazioni binarie condivise
  dall'helper `binaryTool`.
- `Calculator.swift`: espressioni con operatori, parentesi, costanti e funzioni.
- `Clock.swift`: data e ora ISO-8601 (`now`).

## Dipendenze e flusso

Le closure ricevono JSON, usano gli helper di
[`Tools/Core`](../../Core/README.md) e restituiscono `ToolOutput`. Non accedono a
progetto, rete o decoder.

## Estensione

Imporre domini numerici e messaggi chiari per input non validi. Le funzioni
aggiunte al parser devono avere arità deterministica e test per precedenza,
parentesi, NaN/infinito e divisione per zero.
