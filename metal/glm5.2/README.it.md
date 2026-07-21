[English](README.md) | **Italiano**

# Kernel GLM 5.2

Sorgenti dei kernel Metal per il backend GLM 5.2 (`glm-dsa`), un file per
famiglia — lo stesso layout di `metal/deepseek/`:

| File | Contenuto |
|---|---|
| `glm52_router.metal` | selezione del router sigmoid top-8 |
| `glm52_quant.metal` | helper di dot K-quant/IQ2_XXS/Q8_0 (varianti per-thread e simdgroup vettorizzate) — deve precedere `glm52_moe` nell'ordine di concatenazione |
| `glm52_kv.metal` | store/normalizzazione della KV-LoRA compatta, store delle chiavi indexer, store delle righe compatte |
| `glm52_indexer.metal` | scoring dell'indexer DSA |
| `glm52_attention.metal` | core della compact-attention a stadi (`qk_lowrank`, `attention_indexed` sulle righe di cache selezionate, `value_project`; F32 e Q8_0) |
| `glm52_moe.metal` | FFN esperti: kernel di riferimento per-thread, varianti simdgroup, la coppia MoE batched (tutti gli esperti instradati in due dispatch) e il trio fase-B del prefill multi-token (pesi esperti letti una volta per tile di 4 token, bit-esatto vs il percorso per-applicazione) |
| `glm52_rope.metal` | RoPE della coda query/K e del prefisso indexer |
| `glm52_misc.metal` | primitive del grafo di decode residente (RMSNorm, matvec F32, add) |

Ogni famiglia di kernel ha un wrapper Swift isolato e un oracolo CPU sotto
`Sources/DS4Metal/Backends/GLM52/` (più i test di parità in
`Tests/DS4CoreTests/Metal/Backends/GLM52/`), e il decoder incatenato
(`GLM52ChainedDecode`) li guida in produzione con lo stato hidden residente
sulla GPU.

Tutti i file compilano in UNA libreria nell'ordine fissato da
`MetalRuntime.kernelFiles` (prima le tabelle condivise di `metal/common/`).
Il flusso di lavoro di modifica e l'embedding (`make embed-kernels`) sono
documentati in [`../README.md`](../README.it.md).
