[English](README.md) | **Italiano**

# Kernel Laguna S 2.1

Port testuale di `metal/laguna.metal` dal branch di riferimento `laguna-s2.1`
(head `448d569`), più una definizione locale di `block_q6_K` (il port non ha
struct quant condivise con `QK_K`). I kernel coprono solo ciò che l'API Metal
condivisa DeepSeek/GLM non rappresenta:

- `kernel_laguna_head_rms_norm_rope_neox` / `_qk_…`: RMSNorm per-testa in
  stile Qwen seguita dalle coppie rotanti NeoX sul prefisso della testa (i
  parametri YaRN arrivano dal grafo; i blocchi sliding-window passano
  `ext_factor 0`).
- `kernel_laguna_store_kv_f16` (+ varianti rows/stage/commit): le scritture
  della KV cache ad anello F16.
- `kernel_laguna_attention_decode_gqa_f16` (+ varianti `gqa3_split`, `rows`)
  e la famiglia `attention_prefill_gqa*`: attention GQA gated — softmax di
  `q·k/sqrt(128)` sulla finestra ad anello, miscela dei value, poi il gate
  softplus per-testa.
- `kernel_laguna_flash_attn_reduce_gate_f32`: la reduce flash-attention fusa
  col gate usata dal percorso di prefill a contesto lungo; consuma gli
  args/function-constant del vec-reduce da `deepseek/flash_attn.metal`.
- `kernel_laguna_q6_K_matmul_f32`: proiezione densa Q6_K per le proiezioni
  down e la testa di output della ricetta legacy.

L'ordine di concatenazione conta: questo file richiede prima
`deepseek/dsv4_rope.metal` (`rope_yarn`, `rope_yarn_corr_dims`) e
`deepseek/flash_attn.metal`; `MetalRuntime.kernelFiles` e
`scripts/embed_kernels.sh` lo tengono dopo la riga GLM. Gli oracoli CPU
contro cui questi kernel vengono giudicati vivono in
`Sources/DS4Metal/Backends/Laguna/Reference/`.
