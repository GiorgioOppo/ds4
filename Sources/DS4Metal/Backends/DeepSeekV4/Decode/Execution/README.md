**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/Execution

Main implementation of the streaming decoder and the per-layer forward.

## Main files

- [`StreamingDecoder.swift`](StreamingDecoder.swift): ownership of runtime,
  weights, cache, KV state and the knobs read at initialization.
- [`StreamingDecoder+Factories.swift`](StreamingDecoder+Factories.swift):
  resident/streaming factories, expert bundle, MetalIO and cache.
- [`StreamingDecoder+Forward.swift`](StreamingDecoder+Forward.swift): forward
  of a single token and coordination of the phases.
- [`StreamingDecoder+LayerExecution.swift`](StreamingDecoder+LayerExecution.swift):
  command buffers for attention, routing and FFN, including asynchrony.
- [`StreamingDecoder+TensorIO.swift`](StreamingDecoder+TensorIO.swift): upload,
  readback and controlled tensor access.
- [`DecodeLayer.swift`](DecodeLayer.swift): the mathematical sequence of a layer.
- [`CachedLayerProvider.swift`](CachedLayerProvider.swift): per-layer cache of
  `LayerWeights` loaded once and reused across tokens.
- [`SpecDecode.swift`](SpecDecode.swift): state and path for speculative decode.

## Flow

The factory builds the weight and cache strategies. `forward` prepares the
input and walks the layers; `DecodeLayer` uses the operations from
[`Graph`](../../../../Graph/README.md), writes KV and produces the next state.
The asynchronous path overlaps gather and FFN while respecting the
dependencies between command buffers.

## Change rules

Never read a buffer before the operation that writes it completes, and never
reuse in-flight staging. Knobs are acquired once except those declared live.
Separate numerical changes from scheduling changes and provide an A/B path for
experimental optimizations.
