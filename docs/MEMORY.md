# Memory model

How weights, activations, and KV caches live in memory. Open this
when you're trying to understand why something is fast / slow, why a
crash says "no space", or why the converter peaks at 4 GB during
fusion.

## At a glance

| Allocation | Where | Lifetime | Typical size |
|---|---|---|---|
| Checkpoint weights (mmapped) | `SafeTensorsFile` per shard | Process lifetime | 140 GB / 600 GB / 900 GB virtual |
| Activation tensors | `Tensor.empty` calls in forward | Per-token, freed after | MB to low GB |
| KV cache | `Assembly.swift` per layer | Process lifetime | GBs total across layers |
| Compressor state | `Assembly.swift` per layer | Process lifetime | small (`coff * ratio * coff * head_dim` per batch) |
| RoPE freqs | `RoPE.makeFreqs` per layer | Process lifetime | KB |
| Compute outputs (intermediate `Tensor`s) | inside kernels | One command-buffer worth | MB to low GB |

Process-lifetime allocations are made once at `Transformer.load` /
`Transformer.randomInit`. Per-token allocations are short-lived and
mostly automatic via Swift ARC.

## 1. Tensor as MTLBuffer + offset

The fundamental container is `Tensor` (`Sources/DeepSeekKit/Tensor.swift`):

```swift
public final class Tensor {
    public let shape: [Int]
    public let dtype: DType
    public let buffer: MTLBuffer
    public let offset: Int
}
```

The `buffer` is a Metal-backed allocation; `offset` is the byte
position of the tensor's first element inside that buffer. Multiple
tensors can share one `MTLBuffer` with different offsets.

Two construction patterns:

- **Fresh allocation**: `Tensor.empty(shape:dtype:)` calls
  `device.makeBuffer(length:options:)` with `.storageModeShared`. The
  bytes are CPU-writable and GPU-readable; no copy happens on dispatch.
- **From host bytes**: `Tensor.from(bytes:shape:dtype:)` calls
  `device.makeBuffer(bytes:length:options:)` which copies the source
  bytes into a new buffer. Used by tests and tiny config tensors.
- **From mmap**: returned by `SafeTensorsFile.load(_:)`. The buffer is
  shared across all tensors from the same shard (see §2).

All Apple Silicon GPUs use unified memory, so an `MTLBuffer` doesn't
imply a separate VRAM copy.

## 2. mmap-backed safetensors

`SafeTensorsFile.init(url:)` (`Sources/DeepSeekKit/SafeTensors.swift`)
walks like this:

1. `open(url.path, O_RDONLY)` — POSIX file descriptor.
2. `fstat` — get file size.
3. Round size up to a multiple of `sysconf(_SC_PAGESIZE)` (16 KiB on
   Apple Silicon).
4. `mmap(NULL, alignedSize, PROT_READ, MAP_PRIVATE, fd, 0)` — virtual
   address space reservation. The kernel does NOT load anything yet.
   Page residency is on-demand.
5. `device.makeBuffer(bytesNoCopy: mappedPtr, length: alignedSize,
   options: .storageModeShared, deallocator: { munmap($0, $1) })` —
   wrap the mmap as an `MTLBuffer`. The deallocator runs when the
   buffer is released (ARC).
6. Parse the JSON header from the first `8 + headerLen` bytes; build a
   `[String: Entry]` map for `load(_:)` lookups.

Once construction is done, calls to `load("layers.0.attn.wkv.weight")`
return:

```swift
Tensor(shape: e.shape, dtype: parseDType(e.dtype),
       buffer: sharedBuffer,
       offset: dataStart + e.dataOffsets[0])
```

Zero bytes copied. The kernel that consumes this tensor will trigger
page faults on first access, the OS will read from the SSD into RAM,
and subsequent accesses hit the page cache.

### Memory footprint after load

- **Virtual**: sum of all shard sizes (140 GB / 600 GB / etc.) +
  process overhead. `vm_stat` and Activity Monitor report this.
- **Resident**: starts near zero; grows as forward passes touch
  weights. Caps at "active working set" or "available RAM minus OS
  reservations", whichever is smaller.
- **Wired**: minimal. The mmapped pages can be evicted at will.

## 3. KV cache lifecycle

Each MLA layer owns a `kvCache: Tensor` allocated in
`Sources/DeepSeekKit/Assembly.swift` (`Transformer.randomInit` /
`Transformer.load`):

```swift
let kvCacheRows = config.windowSize +
    (ratio > 0 ? config.maxSeqLen / ratio : 0)
let kvCache = Tensor.empty(
    shape: [config.maxBatchSize, max(kvCacheRows, 1), config.headDim],
    dtype: .f32)
```

Layout per batch:

```
kvCache[b]:  [ window_size rows         ][ compress_rows                  ]
             [ sliding-window ring buf  ][ compressor.kvCache slice       ]
```

The Compressor instance gets a `Tensor` referencing the trailing slice
of the same buffer (offset = `windowSize * headDim * sizeof(Float)`):

```swift
comp.kvCache = Tensor(shape: [B, compRows, headDim], dtype: .f32,
                       buffer: kvCache.buffer,
                       offset: kvCache.offset + win * bytesPerRow)
```

This share-with-offset trick lets the MLA pass `kvCache` directly to
`SparseAttention` while the Compressor independently writes into the
trailing slice.

Per-layer KV cache size for V4-Flash (head_dim=512, maxBatch=4,
maxSeqLen=4096):

- window: 4 × 128 × 512 × 4 B = ~1 MB
- compressed (ratio=128): 4 × 32 × 512 × 4 B = ~256 KB
- compressed (ratio=4): 4 × 1024 × 512 × 4 B = ~8 MB

Total across all layers: low GB. Negligible vs the weight footprint.

## 4. Compressor state buffers

Beyond the KV cache slice, each Compressor instance has two state
buffers (`Sources/DeepSeekKit/Layers/Compressor.swift`):

```swift
public var kvState: Tensor     // [maxBatch, coff*ratio, coff*head_dim]
public var scoreState: Tensor  // same
```

Where `coff = 2` if overlap (ratio=4) else 1. These accumulate the
per-step writes during decode and are read out on checkpoint emit.

Size: a few MB per layer in the worst case.

The overlap state-shift (when emitting a compressed token) needs a
temporary buffer of `[B, ratio, coff*head_dim]` — see
`compressor_state_shift_copy_f32` kernel and its host-side blit-copy
back into `state[:, :ratio]`.

## 5. Per-token allocation pattern

During a single `model.forward` call, fresh `MTLBuffer`s are created
for each intermediate. Swift ARC frees them when the last strong
reference disappears (typically at the end of the function scope).

Notable inline allocations per token:

- Embed lookup: `[N, dim]` f32, ~1 MB at V4-Flash dim=4096.
- HC expand: `[N, hc, dim]` f32, ~4 MB (hc=4).
- Per-layer norm outputs, Q/KV projections, scratch tensors: ~16
  similar-sized buffers per block.
- MoE: `[T, dim]` gathered + `[T, dim]` outs + `[N, dim]` y, plus a
  fresh buffer per expert forward. ~tens of MB per layer.
- Sparse attention output: `[B, S, n_heads, head_dim]` f32 ≈ 8 MB at
  V4-Flash.

The MoE forward is the heaviest per-layer because it creates one
intermediate per active expert and re-allocates outs each time. Long
prompts compound this.

To investigate spikes, run with `os_log` enabled and watch the
`vm_pageout` activity in Instruments.

## 6. Page cache during inference

The mmap design means the OS controls what's in RAM. The forward pass
touches:

1. Embed table (read row `id`)
2. Layer 0 weights: attn_norm, wq_a, q_norm, wq_b, wkv, kv_norm, wo_a,
   wo_b, gate, n_active_experts * (w1, w2, w3), shared_expert
3. Layer 1 weights: same shape
4. … through layer N-1
5. Final norm
6. lm_head (read all `vocab` rows)

In strict file order if the converter sharded by layer (which it does
by default). The OS prefetcher learns this pattern after the first
forward and starts reading ahead.

**Cold first forward**: a 13 GB working set (V4-Flash with 13B active
params per token) gets paged in from SSD. At ~7 GB/s sequential read
on an Apple SSD, that's ~2 s of pure I/O, plus compute.

**Warm subsequent forwards**: as long as no other process evicts the
pages, you get RAM-speed reads. Steady-state token rate depends on
compute (the heavy matmuls) not I/O.

**Memory pressure**: when other apps consume RAM, the OS evicts the
LRU mmapped pages first (clean read-only mapping). If you go back to
those pages, they fault in again. Acceptable as long as the working
set fits.

## 7. Converter memory footprint

The converter (`Sources/converter/main.swift`) has a different
profile:

- **Indexing phase**: opens every input shard via `SafeTensorsFile`
  (mmap). Adds ~bytes-of-mmap virtual memory, ~0 resident.
- **Per-tensor fusion** (FP8/FP4 → BF16/F16): the parallel
  implementation slurps one weight tensor at a time into a `Data`
  buffer so worker threads can index it concurrently. Peak resident
  during fusion = `sizeof(input weight tensor)`. For a 64 × 14 GB
  shard, an individual weight is at most a few hundred MB.
- **Output BF16 buffer**: also held in `Data` while being written. ~2×
  the input weight size for BF16 (since BF16 is 16 bits vs FP8's 8).
  Total transient: ~3-4 GB peak per concurrent thread fusion.
- **Output safetensors write**: streamed via `SafeTensorsWriter` in
  64 MB chunks; no big buffer.

At 16-core M3 Max with `DispatchQueue.concurrentPerform` saturating
all cores, peak resident during the fusion phase is roughly
`n_cores * tensor_size`, capped at the larger of the two-axis bounds.

## 8. Pitfalls

### LFS pointer files masquerading as safetensors

`huggingface-cli download` without LFS configured downloads tiny (~1
KB) text files instead of the real binary blobs. `SafeTensorsFile`
would try to `mmap` these and parse a malformed JSON header. The
`WeightLoader` guard `size < 1024` skips them with a clear error.

### Buffer can't span the GPU's max length

`MTLDevice.maxBufferLength` is the upper bound on a single buffer.
Apple Silicon unified memory makes this generous (~the full unified
memory on M-series), so a 14 GB shard fits in one buffer. If you ever
need a single buffer over the device limit, you'd have to split. Not
a current concern.

### Tensor offset alignment

`makeBuffer(bytesNoCopy:length:)` requires the *pointer* to be
page-aligned. The `offset` within the buffer can be anything (each
tensor's offset comes from the safetensors header). This is fine for
compute reads, which don't care about element alignment beyond what
the dtype requires.

### KV cache size scales with `maxSeqLen`

For long-context inference, set `maxSeqLen` in your `config.json` to
the actual length you'll use; otherwise you waste a lot of RAM on
unused KV ring rows. V4-Flash supports `maxSeqLen` up to 1 M, but
`config.json`'s default is much smaller.

### Overlap state buffers are unbounded

The Compressor's overlap state grows as `coff * ratio * coff *
head_dim * maxBatch * sizeof(Float)`. With ratio=4 and head_dim=128
(Indexer), that's only a few MB per layer. Don't worry about it.

### Convert + run on the same SSD

If you `swift run -c release converter` to the same SSD you're then
reading from in `swift run deepseek`, you'll thrash both ways. Use
separate volumes if you can (e.g. input HF on one SSD, converted
output on another).

## 9. Sources

- `Sources/DeepSeekKit/Tensor.swift` — Tensor struct + DType enum
- `Sources/DeepSeekKit/SafeTensors.swift:30-100` — mmap setup
- `Sources/DeepSeekKit/Assembly.swift` — KV cache + Compressor state
  allocation
- `Sources/DeepSeekKit/Layers/Attention.swift:90-110` — KV cache
  buffer sharing with Compressor
- `Sources/DeepSeekKit/Layers/Compressor.swift:30-65` — state buffer
  shapes
- `Sources/converter/main.swift:260-310` — parallel slurp + fuse
  pattern

See also:
- [DTYPES.md](DTYPES.md) for what each byte/nibble means
- [PERFORMANCE.md](PERFORMANCE.md) for I/O-bound vs compute-bound
  measurements
