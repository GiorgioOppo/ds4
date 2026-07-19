**English** | [Italiano](CONTRATTO-GGUF.it.md)

# GLM 5.2 GGUF Contract

The backend accepts only the verified GLM 5.2 geometry. It does not try to
interpret future variants whose constants happen to be compatible "by
chance".

## Essential metadata

| Field | Value |
|---|---:|
| `general.architecture` | `glm-dsa` |
| `glm-dsa.block_count` | 79 |
| `glm-dsa.context_length` | 1,048,576 |
| `glm-dsa.embedding_length` | 6,144 |
| `glm-dsa.vocab_size` | 154,880 |
| `glm-dsa.attention.head_count` | 64 |
| `glm-dsa.attention.key_length` | 576 |
| `glm-dsa.attention.value_length` | 512 |
| `glm-dsa.attention.q_lora_rank` | 2,048 |
| `glm-dsa.attention.kv_lora_rank` | 512 |
| `glm-dsa.rope.dimension_count` | 64 |
| `glm-dsa.expert_count` / `expert_used_count` | 256 / 8 |
| `glm-dsa.leading_dense_block_count` | 3 |
| `glm-dsa.nextn_predict_layers` | 1 |
| `glm-dsa.attention.indexer.head_count` | 32 |
| `glm-dsa.attention.indexer.key_length` | 128 |
| `glm-dsa.attention.indexer.top_k` | 2,048 |

RMS epsilon, RoPE frequency, router scale/norm, FFN dimensions and MLA
fields are also checked against `GLM52Shape.v5_2`.

## Tensor directory

The three global tensors are the Q8_0 embedding, F32 output norm and Q8_0
output head. Every block contains RMS, Q-LoRA, KV-LoRA, MLA projections,
indexer and FFN. Blocks 0–2 have a dense Q8_0 FFN; blocks 3–78 contain the
router, 256 routed experts and one shared expert. Block 78 additionally holds
the `nextn` tensors and is not executed in the normal autoregressive path.

Routed quantizations allowed by the reference graph contract:

- gate/up: IQ2_XXS, Q2_K, Q4_K or Q5_K, with identical type within the same
  layer;
- down: IQ2_XXS, Q2_K, Q4_K, Q5_K or Q6_K;
- control/dense weights: Q8_0 or F32 depending on the individual tensor.

Validation checks names, rank, shapes, types and gate/up consistency before
allocating GPU resources. The future streaming loader must additionally
verify that every expert slice respects the quantization block and the file
bounds.

## Tokenizer and wire format

- GPT-2 byte-level BPE with `glm4` pretokenizer;
- initial sequence `[gMASK]<sop>`;
- roles `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`;
- thinking `<think>…</think>`;
- tool calls
  `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`;
- tool results under
  `<|observation|><tool_response>…</tool_response>`.

`<|tool|>` does not exist in the verified vocabulary and must not be
introduced as a synthetic role.
