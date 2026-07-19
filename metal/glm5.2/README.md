**English** | [Italiano](README.it.md)

# GLM 5.2 kernels

Metal kernel sources for the GLM 5.2 (`glm-dsa`) backend, one file per
family — the same layout as `metal/deepseek/`:

| File | Contents |
|---|---|
| `glm52_router.metal` | sigmoid top-8 router selection |
| `glm52_quant.metal` | K-quant/IQ2_XXS/Q8_0 dot helpers (per-thread and vectorized simdgroup variants) — must precede `glm52_moe` in the concatenation order |
| `glm52_kv.metal` | compact KV-LoRA store/normalization, indexer-key store, compact-row store |
| `glm52_indexer.metal` | DSA indexer scoring |
| `glm52_attention.metal` | staged compact-attention core (`qk_lowrank`, `attention_indexed` over the selected cache rows, `value_project`; F32 and Q8_0) |
| `glm52_moe.metal` | expert FFN: reference per-thread kernels, simdgroup variants, and the batched MoE pair (every routed expert in two dispatches) |
| `glm52_rope.metal` | query/K-tail and indexer-prefix RoPE |
| `glm52_misc.metal` | resident decode-graph primitives (RMSNorm, F32 matvec, add) |

Every kernel family has an isolated Swift wrapper and a CPU oracle under
`Sources/DS4Metal/Backends/GLM52/` (plus parity tests in
`Tests/DS4CoreTests/Metal/Backends/GLM52/`), and the chained decoder
(`GLM52ChainedDecode`) drives them in production with the hidden state
resident on the GPU.

All files compile into ONE library in the order fixed by
`MetalRuntime.kernelFiles` (shared tables from `metal/common/` first).
Editing workflow and embedding (`make embed-kernels`) are documented in
[`../README.md`](../README.md).
