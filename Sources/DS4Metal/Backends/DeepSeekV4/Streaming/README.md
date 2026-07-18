# DeepSeekV4/Streaming

Streaming of per-layer dense weights from SSD through a staging ring, with
optional resident caches to reduce per-token bandwidth.

## Main files

- [`DenseStreamer.swift`](DenseStreamer.swift): state, per-layer plan, file
  descriptors and ring initialization.
- `DenseStreamer+Types.swift`: entries, fields and internal types of the plan.
- `DenseStreamer+Pipeline.swift`: read-ahead and ordered delivery of slots.
- `DenseStreamer+Q4Cache.swift`: Q8_0 -> Q4_K requant and persistent cache.
- `DenseStreamer+CompressorQ8.swift`: optional conversion of the F16 compressors.

## Flow

At startup an aligned-region plan is built. During decode, while the GPU
processes the current layer, the ring reads future layers with
`pread/F_NOCACHE`; resident Q4 or compressor weights are excluded from the
stream. With `DS4_LAZY_IDX`, the indexer's scoring projections also stay out
of the plan: they are read once into resident buffers when the context
actually in use reaches the sparse threshold. On consumption, the slot's
subviews and any resident buffers populate a temporary `LayerWeights`.

## Modification rules

A slot must not be overwritten while the GPU is using it. Keep the number of
in-flight requests bounded and handle all workers in the error paths.
Keep lossless optimizations separate from lossy quantizations and invalidate
the cache whenever the model, format or conversion parameters change.
