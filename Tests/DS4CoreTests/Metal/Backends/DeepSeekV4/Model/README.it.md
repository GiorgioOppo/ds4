# Test del modello DeepSeek-V4

- `GGUFLoaderTests.swift`: validazione dei metadati del modello e del loader.
- `GGUFWeightMapTests.swift`: mappatura da nome tensore a peso e forme.
- `MetalIOCircuitBreakerTests.swift`: decisioni di fallback per finestre di
  I/O Metal lente o fallite.

Usa fixture GGUF sintetiche dove possibile. I test non devono richiedere il
modello scaricato dall'utente, e i test del circuit breaker dovrebbero
iniettare i tempi invece di dipendere dalla velocità attuale dell'SSD.
