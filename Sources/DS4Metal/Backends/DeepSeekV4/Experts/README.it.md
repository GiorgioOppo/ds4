[English](README.md) | **Italiano**

# DeepSeekV4/Experts

Formato sidecar e percorsi I/O ottimizzati per gli expert routed, i pesi più
costosi da recuperare durante la generazione.

## File principali

- [`ExpertBundle.swift`](ExpertBundle.swift): stato e header del sidecar `.expbundle`.
- `ExpertBundle+Builder.swift` e `ExpertBundle+Layout.swift`: scelta percorso,
  validazione layout e costruzione atomica del bundle.
- `ExpertBundle+Read.swift`: letture CPU/pread delle slab contigue.
- `ExpertBundle+MetalIO.swift`: caricamenti diretti tramite MetalIO quando disponibili.
- `ExpertBundle+Usage.swift`: informazioni d'uso e diagnostica del bundle.
- [`MetalIOCircuitBreaker.swift`](MetalIOCircuitBreaker.swift): disabilita MetalIO
  quando prestazioni/errori indicano un percorso non affidabile.

## Flusso

Il builder riordina gate/up/down di ciascun expert in regioni contigue. A runtime
la cache richiede una lista di expert e il bundle riempie slot disgiunti con una
lettura batch; in assenza di sidecar valido o MetalIO, il factory ricade sul
gather dal GGUF senza cambiare il layout consegnato ai kernel.

## Regole di modifica

Verificare identità del GGUF, dimensioni, offset e completezza prima di usare il
sidecar. La costruzione deve pubblicare solo file completi. MetalIO deve avere
fallback CPU corretto e non può cambiare i byte; aggiornare versione/layout se
cambia l'ordine delle slab.
