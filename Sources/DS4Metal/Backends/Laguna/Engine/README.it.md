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

Gli esperti instradati possono essere Q2_K, Q3_K o Q4_K per layer (coerenti,
come garantisce lo schema), quindi girano sia il file ufficiale Q4_K_M sia il
misto RoutedQ2_K-Last27Q3_K; gli helper di dot Q3_K vivono accanto agli altri
K-quant in `metal/glm5.2/glm52_quant.metal`. Limiti di scope deliberati di
questo taglio, rifiutati con errori distinti al caricamento: la ricetta
legacy F16/Q4_K (i suoi percorsi matvec non sono cablati) e il prefill
batched (i prompt passano token-per-token dal percorso di decode — corretto,
non veloce). `LagunaResidentModelOptions.layerCount` tronca lo stack dal
fronte per le prove di bring-up.

`LagunaResidentModelOptions.expertStreaming` è una divergenza opt-in e
dichiarata dall'upstream (che per Laguna impone la residenza completa): il
percorso "segnale" Q8_0 resta residente (~5 GiB) e gli slab dei 10 esperti
instradati selezionati vengono copiati dal mmap per token dopo la rilettura
del router sull'host — stessi kernel, stessi byte, ~1,6 GB di letture per
token. Esiste perché le macchine da 32 GB possano eseguire il file da
45 GiB; aspettati pochi tok/s. `DecodeProfile` (`profileReport()`) riporta
il costo per fase, contando le letture degli slab come gather IO.

`LagunaRuntimeGate.enabled` resta `false` finché questo motore non passa la
parità end-to-end dei logits contro il motore C di riferimento su pesi reali;
selezione, disponibilità a catalogo e dispatch della demo dipendono tutti da
quella sola costante.
