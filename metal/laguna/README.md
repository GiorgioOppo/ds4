**English** | [Italiano](README.it.md)

# Laguna S 2.1 kernels

Verbatim port of `metal/laguna.metal` from the reference `laguna-s2.1` branch
(head `448d569`), plus a local `block_q6_K` definition (the port has no shared
`QK_K` quant structs). The kernels cover only what the shared DeepSeek/GLM
Metal API does not represent:

- `kernel_laguna_head_rms_norm_rope_neox` / `_qk_…`: Qwen-style per-head
  RMSNorm followed by NeoX rotary pairs on the head prefix (YaRN parameters
  arrive from the graph; sliding-window blocks pass `ext_factor 0`).
- `kernel_laguna_store_kv_f16` (+ rows/stage/commit variants): the F16 ring
  KV cache writes.
- `kernel_laguna_attention_decode_gqa_f16` (+ `gqa3_split`, `rows` variants)
  and the `attention_prefill_gqa*` family: gated GQA attention — softmax of
  `q·k/sqrt(128)` over the ring window, value mix, then the per-head
  softplus gate.
- `kernel_laguna_flash_attn_reduce_gate_f32`: the fused flash-attention
  reduce + gate used by the long-context prefill path; it consumes the
  vec-reduce args/function-constants from `deepseek/flash_attn.metal`.
- `kernel_laguna_q6_K_matmul_f32`: dense Q6_K projection for the legacy
  recipe's down projections and output head.

Concatenation order matters: this file needs `deepseek/dsv4_rope.metal`
(`rope_yarn`, `rope_yarn_corr_dims`) and `deepseek/flash_attn.metal` before
it; `MetalRuntime.kernelFiles` and `scripts/embed_kernels.sh` keep it after
the GLM row. The CPU oracles these kernels are judged against live in
`Sources/DS4Metal/Backends/Laguna/Reference/`.
