# Kernels/MoE

Wrapper del router e dei feed-forward routed/shared del Mixture-of-Experts.

## File principali

- [`MetalRouter.swift`](MetalRouter.swift): logits, probabilità, top-k e pesi del router.
- [`MetalMoE.swift`](MetalMoE.swift): matvec per expert selezionati e riduzione.
- [`MetalMoEFused.swift`](MetalMoEFused.swift): gate/up SwiGLU e down-sum fusi.

## Flusso e dipendenze

Il router produce id e pesi; la cache traduce gli id modello in slot residenti o
il loader raccoglie slab contigui. I kernel MoE applicano gate/up, attivazione e
down, poi sommano i contributi pesati nello stato residuo.

## Regole di modifica

Distinguere sempre expert id, slot id e indice nell'unione di prefill. Controllare
numero attivo, padding a peso zero e layout gate/up/down. Le fusioni devono
restare confrontabili con i tre passaggi separati e non alterare la selezione.
