[English](README.md) | **Italiano**

# Motore Laguna

`LagunaResidentModel` è il motore residente di primo taglio: ogni tensore
validato della ricetta ufficiale Q8_0-signal / esperti Q4_K viene caricato
una volta in `MTLBuffer` shared (upstream Laguna richiede residenza completa
— niente streaming SSD, sidecar o cache esperti), ogni layer possiede una KV
cache ad anello F16 (512 righe sui blocchi sliding-window, il contesto
configurato sui blocchi full-attention) e il grafo per-token rispecchia
`laguna_graph_forward_token`: RMSNorm → matvec Q8_0 accoppiati (Q/K, V/gate)
→ norm/RoPE per-testa → store sull'anello → attention GQA gated → proiezione
di output e residuo → FFN denso o instradato. Dispaccia le primitive GLM
condivise dove l'upstream le condivide (`kernel_glm52_rms_norm_f32`,
`kernel_glm52_matvec_pair_sg`, `kernel_glm52_router_select` con 10 esperti
attivi, i matvec K-quant di `glm52_moe`) più i kernel Laguna accanto. La
selezione del router viene riletta sull'host per indirizzare gli slab degli
esperti, come il chained decode GLM.

Limiti di scope deliberati di questo taglio, rifiutati con errori distinti al
caricamento: la ricetta legacy F16/Q4_K e il file misto Q2_K/Q3_K (i loro
percorsi matvec non sono cablati), lo streaming SSD e il prefill batched (i
prompt passano token-per-token dal percorso di decode — corretto, non
veloce). `LagunaResidentModelOptions.layerCount` tronca lo stack dal fronte
per le prove di bring-up.

`LagunaRuntimeGate.enabled` resta `false` finché questo motore non passa la
parità end-to-end dei logits contro il motore C di riferimento su pesi reali;
selezione, disponibilità a catalogo e dispatch della demo dipendono tutti da
quella sola costante.
