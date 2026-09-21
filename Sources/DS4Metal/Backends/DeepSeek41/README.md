# DeepSeek V4.1 native decoder

`DeepSeek41Model` implements the separate V4.1 graph in Swift and Metal:
delayed hyperconnection pre gates, per-layer SWA rings, four shared compressed
KV/index caches, index-source reuse, candidate blocks, BF16 and FP8/FP4
boundaries, the two SSD-backed Engram injections, routing and shared FFN.
This is not an alias for the V4 decoder.

The upstream shader is copied from GiorgioOppo/ds4 revision
`acf5c16bb6a01f8c4f027a0db265ae270d8907ab` with its MIT license. Local graph
adapters are in `metal/deepseek41/swift_graph.metal`; the root-owned
`scripts/embed_backend_kernels.py` regenerates embedded Swift sources.
The library uses the shared runtime's default math mode, as upstream does;
safe math is an explicit diagnostic override.

Prefill is layer-major in bounded chunks of up to 128 tokens. Dense F16,
Q8 and Q4_K projections use matrix kernels above eight rows; smaller batches
retain row reductions. Routed expert tensors and Engram tables are never
registered wholesale with Metal. The pager reads only selected expert
records into six reusable slots, groups all selected token rows by expert,
and scatters outputs back into the original six-slot reduction order. Dense
Q4_0 and routed Q5_K/Q6_K/MXFP4 have packed SIMD fallback kernels. The advanced
upstream 8K carry/suffix optimization and Metal 4 tensor API are not enabled.

`evaluateRows` provides precomputed image-embedding injection. Image rows mask
Engram n-grams and require the checkpoint's VL routing bias; image encoding is
a caller responsibility. Reset discards ring state and n-gram history. Any
partial evaluation failure/cancellation requires reset and prefix replay.

Synthetic tests exercise activation encodings, BF16 ties/offsets, pooling
across chunk boundaries, latest-block retention, top-k ties and long-position
RoPE. Attention fixtures cover 263 tokens across two 128-row ring wraps,
causal compressed-cache visibility, chunked versus scalar execution and a CPU
softmax oracle. Additional fixtures cover the production QA dimensions, causal
index scores, text/vision routing biases and delayed HC with BF16 boundaries.
They do not establish full-checkpoint logit parity or throughput.
