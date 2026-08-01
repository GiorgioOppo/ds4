[English](README.md) | **Italiano**

# DeepSeekV4/Experts

Percorsi I/O ottimizzati direttamente dal GGUF per gli expert routed.

## File principali

- [`MetalIOCircuitBreaker.swift`](MetalIOCircuitBreaker.swift): disabilita MetalIO
  quando prestazioni/errori indicano un percorso non affidabile.

## Flusso

La cache legge gli slab direttamente dal GGUF con pread split parallele.
`DS4_PREAD_SPLIT=3` è il default misurato.

## Regole di modifica

Mantenere i range disgiunti e verificare output byte-identico quando cambia l'I/O.
