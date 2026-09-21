# Qwen3.8 Flash Next

Native Swift decoder for the `qwen4exp` GGUF architecture at upstream commit
`acf5c16bb6a01f8c4f027a0db265ae270d8907ab`. The 48-layer text trunk includes
four-stream hyper-connections, recurrent GDN, the original BF16 PLE n-gram
lookup and dilated convolution, pooled sparse attention, routed and shared
experts, and the output head. The optional MTP layer is not executed.

The model owns its context and recurrent state. `evaluate` appends a nonempty
causal suffix and returns the last token's vocabulary logits. After a failure
or cancellation during evaluation, callers must reset and replay their prefix.
All submitted Metal work is drained before evaluation returns or throws.

Dense tensors are individual read-only mapped GPU views. PLE reads fetch only
the 16 required BF16 rows per token using `pread`. Routed matrices are never
registered as whole Metal buffers: selected experts are compacted into three
layer-scoped slabs, together bounded to 1 GiB. Production Q2/Q4 prefill uses
32-token chunks, native dense GEMM, expert-grouped matrix kernels, and matrix
attention. Decode uses SIMD matvecs, GDN scan and split-softmax attention.

The published Q2 expert recipe uses IQ2_XXS gate/up and padded Q2_K down rows;
the Q4 recipe uses Q4_K gate/up and MXFP4 down rows. Weight geometry and byte
spans are checked before GPU allocation. Native context only is accepted;
no implicit YaRN extension or model capacity inflation is performed.

`metal/qwen38/qwen4.metal` is vendored from the pinned revision. Its small
prelude uses the common runtime's quantization tables and dense helpers.
Run `python3 metal/qwen38/embed.py` after editing the owned Metal source.
The copied vision shader is retained for the separate encoder integration;
the text decoder does not claim image support.

Tests are in `Tests/DS4CoreTests/Qwen38`. Set `DS4_TEST_QWEN38_GPU=1` only in the
serialized GPU validation job to run the synthetic 8-layer full-graph test.
It checks causal chunk/decode agreement, sparse-boundary behavior, reset,
cancellation, and bounds without downloading model weights. Unicode tokenizer
goldens were captured from a standalone extraction of the pinned C tokenizer.
