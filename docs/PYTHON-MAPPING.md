# Swift ↔ Python reference mapping

The Swift port mirrors the upstream Python in `Reference/inference/`
and `Reference/encoding/` as closely as possible. This document is the
cross-walk: when you open a Swift file, look here to find the Python
line range it implements; when you read the Python, find the
corresponding Swift.

The Python files are the **source of truth** for any architectural
question.

## Reference/inference/model.py

| Python | Lines | Swift |
|---|---|---|
| `ModelArgs` dataclass | 34–81 | `Sources/DeepSeekKit/Config.swift` (`ModelConfig`) |
| `ParallelEmbedding` | 83–105 | `Sources/DeepSeekKit/Model.swift` (`ParallelEmbedding`) |
| `linear` dispatch fn | 108–120 | `Sources/DeepSeekKit/Layers/Linear.swift` (`Linear.callAsFunction`) |
| `Linear` class | 123–152 | same file |
| `Column/RowParallelLinear` | 155–180 | not ported (single-rank only) |
| `RMSNorm` | 183–196 | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| `precompute_freqs_cis` (YaRN) | 199–229 | `Sources/DeepSeekKit/YaRN.swift` |
| `apply_rotary_emb` | 232–244 | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Kernels/rope.metal` |
| `rotate_activation` (Hadamard) | 247–251 | `Sources/DeepSeekKit/Layers/Hadamard.swift` + `Kernels/hadamard.metal` |
| `get_window_topk_idxs` | 254–265 | `Sources/DeepSeekKit/Layers/AttentionIndices.swift` (`slidingWindow`) |
| `get_compress_topk_idxs` | 268–276 | same file (`compressed`) |
| `Compressor` | 279–377 | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| `Indexer` | 380–433 | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| `Attention` (MLA) | 436–543 | `Sources/DeepSeekKit/Layers/Attention.swift` |
| `Gate` | 546–584 | `Sources/DeepSeekKit/Layers/MoE.swift` (`Gate`) |
| `Expert` | 587–606 | same file (`Expert`) |
| `MoE` | 609–644 | same file (`MoEFFN`) |
| `Block` (with HC pre/post) | 647–700 | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| `Block.hc_pre` / `hc_post` | 673–686 | `Sources/DeepSeekKit/Layers/HyperConnections.swift` |
| `ParallelHead` | 703–735 | `Sources/DeepSeekKit/Model.swift` (`ParallelHead`) |
| `MTPBlock` | 738–766 | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| `Transformer` | 769–809 | `Sources/DeepSeekKit/Model.swift` (`Transformer`) |

## Reference/inference/kernel.py

| Python | Lines | Swift |
|---|---|---|
| `act_quant_kernel` (FP8) | 40–102 | `Sources/DeepSeekKit/Kernels/act_quant.metal` (`act_quant_fp8`) |
| `act_quant` host fn | 105–125 | `Sources/DeepSeekKit/Layers/ActQuant.swift` |
| `fp4_quant_kernel` | 128–183 | `Kernels/act_quant.metal` (`act_quant_fp4`) |
| `fp4_act_quant` host fn | 186–200 | `Layers/ActQuant.swift` |
| `fp8_gemm_kernel` | 203–254 | `Kernels/fp8_gemm.metal` (`gemm_fp8_to_f32`) |
| `fp8_gemm` host fn | 257–273 | called via `Linear.fp8Forward` |
| `sparse_attn_kernel` | 277–352 | `Kernels/sparse_attn.metal` + `Layers/SparseAttention.swift` |
| `hc_split_sinkhorn_kernel` | 371–427 | `Kernels/hc_sinkhorn.metal` + `Layers/HCSinkhorn.swift` |
| `hc_split_sinkhorn` host fn | 430–438 | same |
| `fp4_gemm_kernel` | 441–515 | `Kernels/fp4_gemm.metal` (`gemm_fp8_fp4_to_f32`) |
| `fp4_gemm` host fn | 518–537 | called via `Linear.fp4Forward` |

## Reference/inference/generate.py

| Python | Lines | Swift |
|---|---|---|
| `sample()` Gumbel-max | 19–24 | `Sampler.sample(_:history:options:)` in `Sources/DeepSeekKit/Sampling.swift` |
| `generate()` loop | 27–69 | `Sources/deepseek/main.swift` (prefill + decode loop, lines 110–155) |
| `main()` loader path | 72–143 | `Sources/deepseek/main.swift` (config + tokenizer + model load) |

## Reference/inference/convert.py

| Python | Lines | Swift |
|---|---|---|
| `FP4_TABLE` constant | 11–14 | `Sources/converter/main.swift` (`deqE2M1` switch) |
| `cast_e2m1fn_to_e4m3fn` | 17–52 | NOT PORTED. Path `--expert-dtype fp8` is documented but falls back to relabel-only |
| `mapping` table | 55–79 | `Sources/converter/main.swift` (`leafMapping`) |
| `main()` walk + write loop | 82–151 | `Sources/converter/main.swift` (top-level statements after arg parsing) |
| `cast_e2m1fn_to_e4m3fn` call site | 142–149 | NOT PORTED (relabel-only path active) |
| wo_a fusion logic | 137–141 | `fuseFP8ToNative` in `Sources/converter/main.swift` |

## Reference/encoding/encoding_dsv4.py

| Python | Lines | Swift |
|---|---|---|
| Special token constants | 17–35 | `Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` (top of file) |
| `system/user/assistant_msg_template` | 42–47 | implicit in `encodeMessages` |
| `response_format_template` | 49–51 | NOT PORTED — caller can prepend to system message manually |
| `tool_call_template` etc. | 52–58 | `EncodingDSV4.encodeToolCalls` |
| `REASONING_EFFORT_MAX` | 64–67 | `EncodingDSV4.reasoningEffortMax` |
| `TOOLS_TEMPLATE` | 70–95 | `EncodingDSV4.toolsBlock(toolSchemasJSON:)` |
| `to_json` / `tools_from_openai_format` / etc. | 101–137 | format helpers not ported; caller passes pre-serialised JSON |
| `encode_arguments_to_dsml` | 139–180 | `EncodingDSV4.encodeArguments` (inline) |
| `encode_messages` | 506–575 | `EncodingDSV4.encodeMessages` (partial — see ROADMAP.md) |
| `parse_message_from_completion_text` | 687–744 | `EncodingDSV4.parseCompletion` |

## Reference test inputs

`Reference/encoding/tests/test_input_1.json` through `test_input_4.json`
plus the matching `test_output_*.txt` are golden tests for the
encoder. The Swift `EncodingDSV4Tests` would consume them when the
port reaches feature parity (currently only the plain chat path is
ported).
