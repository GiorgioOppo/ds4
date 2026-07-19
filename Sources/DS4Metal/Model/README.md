**English** | [Italiano](README.it.md)

# Model

Model types shared across the Metal backends. Architectural descriptions,
GGUF tensor schemas, streaming and family-specific caches live under
[`Backends/`](../Backends/README.md).

## Structure

- [`Quantization/`](Quantization/README.md): metadata for quantized MoE
  layouts.

## Flow

Code in this folder must not assume tensor names, KV cache shape, head count
or routing strategy of a single family. The DeepSeek-V4 backend keeps its own
model in [`Backends/DeepSeekV4`](../Backends/DeepSeekV4/README.md).

## Modification rules

Promote a type here only when at least two backends genuinely share its
semantics and layout. Fail explicitly on unsupported shapes or quantizations;
do not use fallbacks that produce plausible but wrong logits.
