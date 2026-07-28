[English](README.md) | **Italiano**

# Backend Laguna S 2.1 (a stadi)

Spazio riservato al backend Metal di Laguna S 2.1. Oggi contiene solo le parti
validabili senza hardware Apple:

- [`TensorSchema/`](TensorSchema/README.it.md): il contratto esatto dei
  tensori GGUF delle due ricette di quantizzazione pubblicate (signal path
  Q8_0, legacy Q4_K/F16) e del file misto con esperti instradati Q2_K/Q3_K,
  portato da `weights_validate_laguna_layout` del branch di riferimento
  `laguna-s2.1`, più la directory payload-free `LagunaWeightMap`.
- [`Reference/`](Reference/README.it.md): oracoli CPU scalari del percorso di
  decode (norm/RoPE, attention GQA gated, router, blocchi FFN).
- [`Kernels/`](Kernels/README.it.md): wrapper di validazione sui kernel
  portati in `metal/laguna/`, giudicati su hardware contro gli oracoli.
- [`Engine/`](Engine/README.it.md): l'unico interruttore di abilitazione del
  runtime, attualmente spento.

Il decoder vero e proprio — `metal/laguna.metal`, il grafo di attention
GQA/SWA, il gather MoE e il compagno speculativo opzionale DFlash — non è
ancora portato; il piano chiavi in mano è in `docs/PORTING-GAPS.md`. Finché il
gate di parità dei logits non passa su hardware, un GGUF Laguna viene
riconosciuto, validato e rifiutato con un errore esplicito di
non-implementato.
