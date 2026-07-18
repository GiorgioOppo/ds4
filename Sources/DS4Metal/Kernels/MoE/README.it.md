# Kernels/MoE

Wrapper per il router e per i feed-forward routed/shared del
Mixture-of-Experts.

## File principali

- [`MetalRouter.swift`](MetalRouter.swift): logit, probabilità, top-k e pesi
  del router.
- [`MetalMoE.swift`](MetalMoE.swift): matvec per gli esperti selezionati e
  riduzione.
- [`MetalMoEFused.swift`](MetalMoEFused.swift): SwiGLU gate/up fusa e
  down-sum.

## Flusso e dipendenze

Il router produce id e pesi; la cache traduce gli id di modello in slot
residenti oppure il loader raccoglie slab contigue. I kernel MoE applicano
gate/up, attivazione e down, poi sommano i contributi pesati nello stato
residuale.

## Regole di modifica

Distingui sempre l'id di esperto, l'id di slot e l'indice nell'unione di
prefill. Controlla il conteggio attivo, il padding a peso zero e il layout
gate/up/down. Le fusioni devono restare confrontabili con i tre passaggi
separati e non devono alterare la selezione.
