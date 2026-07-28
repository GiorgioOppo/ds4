[English](README.md) | **Italiano**

# Contratto del modello Laguna S 2.1

Questa directory possiede la geometria portabile del modello Laguna S 2.1 e la
validazione dei metadati GGUF. Non ha alcuna dipendenza da Metal e non registra
né seleziona un backend di runtime.

`LagunaConfiguration` accetta solo `general.architecture = "laguna"` e l'esatta
forma Laguna S 2.1 a 48 blocchi implementata dal branch di riferimento
`laguna-s2.1` di `antirez/ds4` (`DS4_SHAPE_LAGUNA_S21`): GQA con 8 teste KV e
un'alternanza per-layer delle teste query (48 teste full-attention ogni quarto
blocco, 72 teste sliding-window altrove), RoPE YaRN con una base di frequenza
indipendente per la sliding-window, un blocco denso iniziale e 256 esperti
instradati con 10 attivi più un esperto condiviso. Le varianti Laguna
sconosciute falliscono al caricamento invece di essere eseguite con dimensioni
incompatibili.
