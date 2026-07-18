# Graph/Operations

Graph operations grouped by mathematical phase of the model.

## Key files

- `GraphAttention.swift` and `GraphAttentionOutput.swift`: attention and output projection.
- `GraphRopeKV.swift`: RoPE, KV write and read.
- `GraphCompressor*.swift` and `GPUTensor+Compressor.swift`: NSA pooling, compressor, and indexer.
- `GraphRouter.swift`, `GraphMoE.swift`, and `GraphMoEMM.swift`: routing and routed FFN,
  including the matrix-matrix prefill variant.
- `GraphHyperConnections.swift`: HyperConnections reduce/expand.
- `GraphOutput.swift` and `GraphUnary.swift`: output head and simple transforms.

## Flow

The operations are extensions of `GraphContext`: they receive already-sized
tensors, select pipelines and quantization, encode the kernels, and return the
tensor/state for the next phase. The exact path is orchestrated by
[`Backends/DeepSeekV4/Decode/Execution`](../../Backends/DeepSeekV4/Decode/Execution/README.md).

## Attention decode: raw ring and split-K

`GraphAttention.swift` materializes for FlashAttention first the raw rows of
the SWA window and then the compressed rows. With `DS4_RAW_RING=1` the raw KV
is an `MTLBuffer` in Metal shared/unified memory, not a KV cache on SSD and
not an alias of the Disk-KV checkpoints. When the circular window wraps past
the end of the buffer, `kernel_dsv4_raw_ring_cpy_f32_f16` reorders it
chronologically and converts it F32→F16 with a single 2D GPU dispatch; the
contiguous case keeps the existing linear copy.

The adaptive split-K uses the total number of visible rows, that is
`totalRows = rawRows + compressedRows`, and chooses exactly:

```text
nwg = min(32, max(1, ceil(totalRows / 32)))
```

It no longer rounds `nwg` up to the next power of two: 128 rows use 4
workgroups, 129 use 5, 161 use 6. The vec/reduce kernels support all values
1…32; with `nwg == 1` the separate reduce is not needed. Setting
`DS4_ADAPTIVE_SPLITK=0` always restores 32 workgroups for an A/B comparison.

## Modification rules

A function must represent a recognizable mathematical responsibility. Do not
hide CPU waits or readbacks in apparently GPU-only helpers. When a fusion
changes the floating-point order, keep it toggleable and compare it against
the unfused path.
