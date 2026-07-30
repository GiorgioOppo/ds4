[English](README.md) | **Italiano**

# Kernel Laguna S 2.1

Port testuale di `metal/laguna.metal` dal branch di riferimento `laguna-s2.1`
(head `448d569`), più una definizione locale di `block_q6_K` (il port non ha
struct quant condivise con `QK_K`) — spezzato in un file per area nello stile
GLM (`laguna_quant`, `laguna_rope`, `laguna_kv`, `laguna_attention`). I kernel
coprono solo ciò che l'API Metal condivisa DeepSeek/GLM non rappresenta:

- `laguna_rope.metal` — `kernel_laguna_head_rms_norm_rope_neox` / `_qk_…`:
  RMSNorm per-testa in stile Qwen seguita dalle coppie rotanti NeoX sul
  prefisso della testa (i parametri YaRN arrivano dal grafo; i blocchi
  sliding-window passano `ext_factor 0`).
- `laguna_kv.metal` — `kernel_laguna_store_kv_f16` (+ varianti
  rows/stage/commit): le scritture della KV cache ad anello F16. Definisce
  anche la struct args del prefill condivisa con `laguna_attention.metal`.
- `laguna_attention.metal` — `kernel_laguna_attention_decode_gqa_f16`
  (+ varianti `gqa3_split`, `rows`) e la famiglia `attention_prefill_gqa*`:
  attention GQA gated — softmax di `q·k/sqrt(128)` sulla finestra ad anello,
  miscela dei value, poi il gate softplus per-testa. Più
  `kernel_laguna_flash_attn_reduce_gate_f32`: la reduce flash-attention fusa
  col gate usata dal percorso di prefill a contesto lungo; consuma gli
  args/function-constant del vec-reduce da `deepseek/flash_attn.metal`.
- `laguna_quant.metal` — `kernel_laguna_q6_K_matmul_f32`: proiezione densa
  Q6_K per le proiezioni down e la testa di output della ricetta legacy
  (auto-contenuto, porta con sé il `block_q6_K` locale).

L'ordine di concatenazione conta: questi file richiedono prima
`deepseek/dsv4_rope.metal` (`rope_yarn`, `rope_yarn_corr_dims`) e
`deepseek/flash_attn.metal`, e `laguna_kv` deve precedere `laguna_attention`
(struct args del prefill condivisa); `MetalRuntime.kernelFiles` e
`scripts/embed_kernels.sh` tengono la famiglia dopo la riga GLM. Gli oracoli
CPU contro cui questi kernel vengono giudicati vivono in
`Sources/DS4Metal/Backends/Laguna/Reference/`.
