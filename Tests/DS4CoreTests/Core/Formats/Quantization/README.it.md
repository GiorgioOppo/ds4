[English](README.md) | **Italiano**

# Test di quantizzazione

- `HalfTests.swift` verifica la conversione in mezza precisione e i casi
  limite.
- `QuantizeTests.swift` valida gli helper di quantizzazione/dequantizzazione e
  i limiti di errore.

I test numerici devono dichiarare se richiedono uguaglianza bit a bit o un
confronto basato su tolleranza. Includi, quando applicabile, casi con zero,
segno, limiti di intervallo, valori non finiti e blocchi incompleti.
