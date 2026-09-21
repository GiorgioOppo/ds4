# Bonsai 2 native decoder

`BonsaiModel` runs the complete standard Ternary Bonsai 2 27B text trunk in
Swift and Metal. `BonsaiConfiguration` verifies the Qwen35 geometry, tensor
layouts, all 401 forward-folded weights, inverse embedding transform, explicit
Hadamard signs and MRoPE sections before GPU allocation. The standalone Metal
library uses safe floating-point mode, independently of the DeepSeek library.

The kernels are vendored from GiorgioOppo/ds4 revision
`acf5c16bb6a01f8c4f027a0db265ae270d8907ab`; their upstream MIT notices are retained.
`metal/bonsai/bonsai.metal` is the source of truth. Run the root-owned
`scripts/embed_backend_kernels.py` when changing shader source.

Weights remain packed PQ2_0/PTQ1_0. Decode uses the specialized PQ2 GEMV,
fused gate/up and BF16 alpha/beta pair where their gates apply. Prefill runs
layers over chunks of up to 128 rows, using PQ2 matrix tiles, PTQ 4/8-row
kernels, convolution scans and D128 recurrent scans. Attention has causal
F32 KV caches and bounded score scratch. Session memory is admitted before
large allocations; context capacity may therefore be below model metadata.

`evaluateRows` accepts original-space embedding replacements and optional
per-row time/height/width positions. Image encoding and image-grid construction
belong to the caller. Subsequent text positions must include its MRoPE delta.
These hooks alone do not constitute a vision encoder or image UI integration.
A failed/cancelled evaluation invalidates state until `reset()` and prefix
replay. Cancellation drains any submitted GPU work before returning.

Tests under `Tests/DS4CoreTests/Bonsai` cover both packed codecs, Hadamard,
projection tails, convolution/GDN continuation and MRoPE. Full-checkpoint
quality and throughput still require a separately acquired supported GGUF.
