# Port from the `antirez/ds4:glm5.2` branch

The [`glm5.2`](https://github.com/antirez/ds4/tree/glm5.2) branch is the
functional reference, not a source to copy monolithically. Its GLM runtime is
integrated into the original large C/Objective-C/Metal graph; DwarfStar Swift
instead separates it by architecture, with CPU oracles and tests before wiring
each kernel into generation.

## What upstream introduces

- `glm-dsa` shape and loader for the 79 blocks, with 78 autoregressive layers;
- ChatGLM/GLM templates, High/Max reasoning and flat XML tool grammar;
- MLA/DSA with compact KV-LoRA cache, RoPE tail and 32×128 indexer;
- IndexShare: full indexer in layers 0, 1, 2, 6, 10, …, 74;
- full causal attention for the short rows and top-2048 selection for long
  context;
- sigmoid top-8 routing, bias used only for selection, normalized non-biased
  weights and scale 2.5;
- three dense FFN layers, then routed MoE plus shared expert;
- resident loading or SSD streaming of the experts and dedicated Metal kernels;
- separate paths for single token, batch prefill and indexed prefill.

The branch does not provide an equivalent production GLM CPU/CUDA/ROCm
backend, does not enable MTP in normal generation and does not contain
reliable benchmarks usable as a performance promise for this app.

## What is already ported here

| Area | DwarfStar Swift |
|---|---|
| Detector and backend gate | `glm-dsa` registered, explicit rejection until runnable |
| Metadata/shape | strict validation of the 31 fields used by the graph |
| Tensor directory | complete schema, payload-free map and top-8 read planner |
| Weight reads | bounded `pread` on descriptors and top-8 plans, and slot-cache LRU with batch pinning and byte-identical hits; MetalIO still to be wired |
| Tokenizer/chat/tool | implemented with golden tests on the real IDs |
| Router | CPU oracle and dedicated Metal kernel |
| DSA/IndexShare | layout, policy, CPU scorer/top-k, Metal primitives, multi-block GPU top-k, tail RoPE (MLA queries) and prefix RoPE (indexer queries/keys), per-row K rotation at attention |
| Compact attention | dual CPU oracle (expanded vs absorbed) and staged Metal kernels qk_lowrank/indexed/value_project (F32 and Q8_0) compared against the oracle |
| FFN/MoE/output head | F32-ref CPU oracle, reference K-quant dequant and validation Metal kernels for all stages (routed K-quant, dense/shared/output head Q8_0); per-quant optimized families missing |
| Cache | lazy F16 planner; KV-LoRA norm/store and indexer-K norm/RoPE/store isolated |
| Decoder | CPU oracle for first token AND decode step (exact wiring of `glm_graph_forward_token`: cache store before selection/attention, fill-range/top-k with `visible = pos+1`, IndexShare verbatim), GPU compositions of both on the validated kernels compared against the oracles; resident graph of the attention phase (weights loaded once, GPU cache in place, one command buffer in the fill-range path) at parity with the executor; real GLM52ResidentModel engine (weight loading from the GGUF, embedding row, token-by-token prefill, greedy decode; IQ2_XXS experts supported with kernels + reference dequant) |

## Deliberate choices that differ from upstream

1. **Separate backend.** No GLM branch in the DeepSeek types and no alias that
   could send a GGUF to the wrong decoder.
2. **Lazy capacity.** Upstream normally sizes the compact cache to the logical
   window; here the planner uses append-only slabs and budgets, so an empty
   100k window does not immediately reserve about 8.87 GiB.
3. **Frontend before the decoder.** Tokenizer and tool protocol are testable
   and selectable for diagnostics, but do not imply inference capability.
4. **Correctness before batch shortcuts.** The optimized prefill paths will be
   enabled only after logits comparison; upstream's short token-major variant
   is not assumed to be numerically equivalent.
5. **Current DwarfStar streaming.** The port must reuse the existing circuit
   breaker, MetalIO/pread and memory limits, instead of duplicating the
   upstream C subsystem.

## Quality gates

A green isolated kernel is not enough. Making GLM runnable requires at least:

- GPU tests actually run on Metal, not skipped for lack of a device;
- tensor-by-tensor comparison on one dense layer and one MoE layer;
- documented hash/tolerance for all 154,880 logits;
- equivalence between single-token decode and prefill for the same prefix;
- unchanged quality between resident and streaming;
- stop, reasoning and tool calls verified in the full chat loop;
- cancellation, screen switching and memory release verified in the GUI.
