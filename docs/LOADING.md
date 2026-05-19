# Loading the model

How a 142 GB checkpoint becomes a usable `Transformer` on a 16 GB
Mac, what `LoadPlan` does before it commits, where the three loader
strategies differ, what the streaming pool buys you, and how
cross-restart KV-cache persistence is wired.

Companion docs:

- [`MODEL.md`](MODEL.md) — what the loader hands off to.
- [`MEMORY.md`](MEMORY.md) — mmap walkthrough, working-set
  estimates, page-cache behaviour. Some content overlaps; this doc
  focuses on the *control flow* (deciding, opening, streaming).
- [`MODULES.md`](MODULES.md) — per-file index.
- [`USAGE.md`](USAGE.md) — operational reference (flags, model
  picker).

> 🇮🇹 La versione italiana è [`LOADING.it.md`](LOADING.it.md).

---

## 1. The pipeline

```
~/Downloads/V4-Flash-HF/                       (or post-converter directory)
├── config.json
├── generation_config.json
├── tokenizer.json + tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00046.safetensors
├── …
└── model-00046-of-00046.safetensors

           Transformer.load(config:from:strategyOverride:forceLoad:warmupOnLoad:kvCacheFile:)
                                  │
                                  ▼
                  ┌─────────────────────────────────────┐
                  │ LoadPlan.decide(modelDir:override:)   │
                  │                                       │
                  │  • WeightLoader.discoverShards(...)   │
                  │  • SystemProbe.effectiveProcessBudget │
                  │  • pickStrategy(...) → preload/mmap/  │
                  │    streaming                          │
                  │  • shardTooLarge / kvCacheTooLarge    │
                  │    refusals (force-load bypasses)     │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ WeightLoader(plan: ...)               │
                  │                                       │
                  │  • mmap each shard → MTLBuffer        │
                  │  • parse headers → name → shard idx   │
                  │  • build shardLayers ownership table  │
                  │  • streaming? build StreamingPool     │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ ModelConfig.inferred(from: loader)    │
                  │ (patch missing/stale config fields    │
                  │  from real tensor shapes)             │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ KV cache: project bytes, refuse if    │
                  │ projectedKVCacheBytes > budget.       │
                  │ Optional: open KVCacheFile for         │
                  │ cross-restart persistence.            │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ Walk the canonical V4 weight tree:    │
                  │ embed, norm, head, hc_head_*,         │
                  │ per layer: attn, ffn, hc_attn_*,      │
                  │ hc_ffn_*, compressor, indexer, …      │
                  │ (Sources/DeepSeekKit/Assembly.swift)  │
                  │ Missing names → random init + warn.   │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                       Transformer (ready for forward)
```

Three phases worth understanding separately: **planning** (decide
without committing), **opening** (mmap + index), **assembly** (read
the canonical name tree, build the Swift objects). Streaming
overlays a fourth phase that runs *during* inference.

---

## 2. Planning: `LoadPlan.decide`

`Sources/DeepSeekKit/LoadStrategy.swift:193`.

Inputs: model directory + an optional strategy override
(`"auto"|"preload"|"mmap"|"streaming"` from `--load-strategy`) +
`forceLoad: Bool` (the `--force-load` CLI flag).

Outputs: a `LoadPlan` with the chosen strategy + every probe number
worth logging (total bytes, max shard bytes, available RAM,
physical RAM, MTL recommended working set, cpu cores) + a one-line
"why this strategy" reason.

### The three strategies

| Strategy | When | Behaviour |
|---|---|---|
| `.preload` | total ≤ 80% of effective budget | `pread(2)` every shard into a freshly allocated `MTLBuffer`. Fastest steady-state, highest cold-start cost. Concurrent (capped at 4 streams to avoid SSD contention). |
| `.mmap` | total > 80% of budget but ratio < 10× | `mmap(PROT_READ, MAP_PRIVATE)` per shard, wrap as `MTLBuffer(bytesNoCopy:)`. OS pages on demand. The default for "the checkpoint fits in RAM but only barely". |
| `.streaming` | ratio > 10× the budget, OR explicitly forced | Same mmap as above, plus a **StreamingPool** that rotates per-layer shards into a fixed-size pool of pinned slots. Bounded working set, slower per-token, the only way to run V4-Flash on a 16 GB Mac. |

### The decision matrix

`pickStrategy(total: available: totalOversubMultiplier: override:)`
at `LoadStrategy.swift:252`:

```swift
override == nil/"auto":
    if available == 0:                     → .mmap     (probe failed, safe fallback)
    if total > available * 10:             → .streaming (wildly oversubscribed)
    if total ≤ 0.80 × available:           → .preload   (fits comfortably)
    otherwise:                             → .mmap     (fits but tight)
override == "preload"/"mmap"/"streaming":  → that, with the override as the reason
```

The numerator/denominator pair (4/5 = 0.80) and the multiplier (10)
were locked in after testing on 16 GB, 64 GB, 128 GB, and 192 GB
hosts. See the inline comments in `LoadStrategy.swift:140-169` for
the rationale.

### The refusal guards

Two errors can be thrown *before* the strategy is even picked
(unless `--force-load` is on):

- **`shardTooLarge`** — the biggest single shard exceeds 50% of the
  effective budget. The kernel can't mmap a shard contiguously if it
  competes for unified memory with the GUI compositor and the GPU's
  command-buffer working set on every fresh hot-tensor fault. Refusing
  here leaves headroom. Resolutions in the error message: free
  memory, re-shard with a smaller `--shard-size-gb`, or use
  `--force-load`.
- **`kvCacheTooLarge`** — separate guard, fires *after* the loader
  has run but before block assembly. `ModelConfig.projectedKVCacheBytes`
  exceeds the budget. Streaming doesn't help here: KV caches are
  dense `storageModeShared` MTLBuffers, not mmap'd file pages, and
  the GPU writes to them during forward. `--force-load` doesn't help
  either. The error tells the user to lower
  `max_position_embeddings` / `max_batch_size` in `config.json` (the
  common trap: HF ships `max_position_embeddings = 1 M` which on a
  16 GB Mac wants ~50 GB of KV state).

### The unified-memory budget

`SystemProbe.effectiveProcessBudget()` returns
`min(processAvailableRAM, physicalRAM × 0.60)`. On Apple Silicon the
CPU and GPU share the same physical pages, so the "available" figure
(free + inactive + speculative pages from `host_statistics64`) is
optimistic — those inactive pages are file cache and other apps'
working sets the kernel only evicts under real pressure. Once we
start mmap'ing a multi-hundred-GB checkpoint the kernel is forced
to evict, the GPU's working set competes for the same pages, and
the whole system thrashes.

Treating 60% of physical as the ceiling keeps ~40% for the OS, the
GUI, the GPU command queue, and any other app open. Tunable via the
`physicalCap:` argument; the default 0.60 is what the refusal guards
use.

---

## 3. Opening: `WeightLoader.init`

`Sources/DeepSeekKit/WeightLoader.swift:53`. After the plan is
decided, the loader:

1. **Discovers shards**: `discoverShards(in:)` enumerates `*.safetensors`
   files in the directory, sorts by filename, drops anything < 1 KiB
   (LFS pointer stubs). Throws if none remain.
2. **Opens each shard**: `SafeTensorsFile(url:)` does `open` →
   `fstat` → round up to page size → `mmap(MAP_PRIVATE)` →
   `device.makeBuffer(bytesNoCopy: ..., deallocator: { munmap(...) })`.
   The deallocator runs when the MTLBuffer is released (ARC), so the
   `munmap` is automatic on process exit.
3. **Parses headers**: each shard starts with an 8-byte little-endian
   length + a JSON header listing every tensor's name, dtype, shape,
   and `[data_offset_start, data_offset_end]`. The loader builds a
   flat `[String: shardIndex]` map for `load(_:)` lookups.
4. **Classifies shards** (per-layer ownership): `buildShardLayers`
   walks every name. A shard "owns" layer K iff every tensor in it
   has a `layers.K.…` prefix; otherwise it's a "shared" shard
   (owner = -1), holding top-level tensors (embed, head, norm,
   `hc_head_*`) that get touched every forward pass.
5. **(streaming only) Builds the StreamingPool**:
   `StreamingPool(shards:shardLayers:...)`. Allocates one MTLBuffer
   for shared shards (mlock'd) and another `slotCount × slotSize`
   MTLBuffer for the rotating per-layer slots. Resolves every
   tensor's `TensorLocation` (slot + offset + shape + dtype) at
   construction so subsequent `load(name)` calls don't even need a
   dictionary lookup beyond the resolved location map.

### Why the streaming pool exists

When you wrap an mmap'd region as an MTLBuffer via `bytesNoCopy:`,
the Apple driver *pins* those pages whenever the buffer is
referenced. `madvise(MADV_DONTNEED)` returns success but `mincore()`
shows zero pages dropped — the kernel won't evict pages the driver
claims. With 147 GB of V4-Flash mmap'd through 45 such buffers, the
system runs at 100% memory pressure permanently. Either crashes or
makes inference unusably slow.

The streaming pool sidesteps this by allocating its own
`storageModeShared` MTLBuffers (which the kernel *can* evict from,
since we manage their contents explicitly) and using `pread(2)` to
copy shard data in / out of them on demand. The mmap stays for the
header parse; the actual weight bytes flow through the pool's
buffers.

### Tensor lookup

`WeightLoader.load(_:)`:

- **mmap / preload mode**: look up `index[name] → shardIdx`,
  delegate to `shards[shardIdx].load(name)` which returns a Tensor
  sharing the shard's MTLBuffer with the right offset.
- **streaming mode**: look up `pool.tensorLocation[name]`, return a
  Tensor pointing into `pool.sharedSlot` or `pool.rotatingSlot` at
  `loc.offsetInSlot`. The pool's backing data isn't valid until
  `ensureLayer(K)` has been called for the owning layer — the
  forward pass (`Transformer.forward`) ensures it before referencing
  the tensor.

`tryLoad(_:[fallbackNames])` exists for HF / converted alias pairs:
`"head.weight"` vs `"lm_head.weight"`, `"<base>.scale"` vs
`"<base>.weight_scale_inv"`, etc.

`shape(of name:)` / `shape(ofAny:)` query the on-disk shape without
loading bytes — used by `ModelConfig.inferred(from:)` to patch a
stale config.

### Missing names

`WeightLoader.missing: Set<String>` collects every name that came
back nil. Assembly.swift fills missing tensors with random init
(via `MiniRNG`) and prints a summary on stderr at the end. This is
intentional: it lets a partially-pruned checkpoint still produce a
forward pass (useful when porting, debugging, or iterating on layer
shapes before the release is final).

---

## 4. The StreamingPool

`Sources/DeepSeekKit/StreamingPool.swift:58`. Activated only in
`.streaming` strategy. Two MTLBuffer regions, both
`.storageModeShared`, allocated once at load time:

### Shared slot

Concatenated data of every "shared" shard (top-level tensors). Sized
exactly to fit the sum of their byte counts. `mlock`'d at construction
so the kernel can't evict — these tensors get touched every layer.

### Rotating slot

One MTLBuffer of size `slotCount × slotSize` bytes, carved into
`slotCount` sub-slots. Each per-layer shard K is **permanently
assigned** to sub-slot `K mod slotCount` — the slot index is a
property of K, not of access order.

```
rotatingSlot (one MTLBuffer):
[ sub-slot 0 ][ sub-slot 1 ][ sub-slot 2 ] ... [ sub-slot N-1 ]
  slotSize     slotSize       slotSize           slotSize

layer K=0  → sub-slot 0
layer K=1  → sub-slot 1
…
layer K=N  → sub-slot 0      (overwrites layer 0 in that slot)
layer K=N+1 → sub-slot 1
…
```

### Why modular assignment

`Tensor` captures `MTLBuffer + offset` at construction time
(Assembly.swift builds every block's weights upfront, before the
first forward). To avoid an indirection layer on every tensor access,
layer K's bytes must always live at the same address. With sub-slot
= `K mod N`, the address is
`rotatingSlot.contents() + (K mod N) * slotSize + inShardOffset` —
stable for the lifetime of the pool. Subsequent rotation just
overwrites the same address space.

### Sliding window

With N sub-slots and strictly sequential forward (layer 0, 1, ...,
L-1), the working set per layer is 1 and the prefetched window is
N-1. After computing layer K and `releaseLayer(K)`, the pool
schedules a background `pread` of layer K+N into sub-slot
`(K+N) mod N = K mod N` — i.e. the slot holding K, which is no
longer needed because the GPU finished with it before `releaseLayer`
ran (`cmdL.waitUntilCompleted` before the call).

By the time `ensureLayer(K+N)` is called, the prefetch is already
complete and the fast path returns without I/O.

### Pool sizing

`WeightLoader.computeStreamingSlotCount(...)` decides N at init:

```
sharedBytes = sum of shared-shard sizes
maxLayerBytes = max(per-layer shard size, rounded up to page)
slotSize = aligned(maxLayerBytes)
reserveBytes = 4 GiB (DEEPSEEK_STREAMING_RESERVE_GB env)
budgetCap = SystemProbe.effectiveProcessBudget()

rotatingBudget = budgetCap - sharedBytes - reserveBytes
N = min(layerShardCount, rotatingBudget / slotSize)
```

Capped at the total per-layer shard count (= "every shard
pre-loaded"). Lower bound is 1 (every layer transition pays a
blocking pread). The 4 GiB reserve covers the KV cache, activation
buffers, the Metal command queue, and headroom for the OS / GUI.

Override via `DEEPSEEK_STREAMING_SLOTS` or
`DEEPSEEK_STREAMING_RESERVE_GB` env vars.

---

## 5. KV cache projection refusal

After the loader is built but before block assembly,
`Assembly.swift` calls `config.projectedKVCacheBytes` and refuses if
it exceeds the budget:

```swift
let kvProjected = config.projectedKVCacheBytes
let kvBudget = SystemProbe.effectiveProcessBudget()
if kvBudget > 0, kvProjected > kvBudget {
    throw LoadStrategyError.kvCacheTooLarge(
        projected: kvProjected, available: kvBudget,
        maxSeqLen: config.maxSeqLen,
        maxBatchSize: config.maxBatchSize)
}
```

`projectedKVCacheBytes` walks every layer's `compress_ratio` and
sums:

- Attention KV cache: `maxBatchSize · (windowSize + maxSeqLen/ratio) · headDim · 4 bytes` (ratio > 0) or `maxBatchSize · windowSize · headDim · 4` (ratio == 0).
- Indexer KV cache: same formula, only on `ratio == 4` layers.
- Compressor state: `hcMult · ratio · maxBatchSize · headDim · 4 bytes` (ratio > 0).
- Each MTP layer's attention KV cache (~one batch's worth).

For the default V4-Flash config (`max_seq_len = 4096`, `max_batch = 4`)
this is < 100 MB total. For `max_seq_len = 1 M` it scales to tens of
GB — which is what the refusal protects against.

The error's resolution text recommends `jq`'ing `config.json` to
lower `max_position_embeddings` and `max_batch_size`.

---

## 6. Cross-restart KV cache persistence

Optional. When `Transformer.load(..., kvCacheFile:)` is passed a
`KVCacheFile`, every per-layer KV cache + every Compressor / Indexer
state tensor becomes a zero-copy slice of an mmapped backing file
instead of a fresh in-memory `MTLBuffer`. Closing the model and
reopening it later re-mmaps the same file; the tensors point at the
same offsets; the KV state is "automatically" preserved.

### `KVCacheFile`

`Sources/DeepSeekKit/KVCacheFile.swift:28`. On-disk format:

```
[ 4096-byte header ][ payload (page-aligned) ]
```

Header fields (`KVCacheFile.Header`):
- `magic` = `'KVC1'` (0x4B564331)
- `version`
- `payloadBytes`
- `prefilledTokens` (last checkpoint)
- `historyHashLow` / `historyHashHigh` (64-bit hash of the prompt
  history at the last checkpoint, to detect mismatched resumption)
- `modelPathHash` (which model produced this state, to refuse
  resumption against a different model)

API:
- `init(url:payloadBytes:modelPathHash:)` — creates the file if
  missing, sizes the payload, mmaps it.
- `readHeader()` / `resetHeader(...)` / `updateCheckpoint(...)`.
- `region(offset:length:)` — get an `MTLBuffer + offset` pair for
  a contiguous slice (used by `Assembly` to back individual KV
  tensors).
- `tensor(at: KVCacheLayout.Region, shape:, dtype:)` — convenience
  wrapper.
- `attemptResume(newTokens:)` — compare the saved `historyHash` to
  the new prompt's hash, find the common prefix, return
  `.fullMatch / .partialMatch(P) / .mismatch`.

### `KVCacheLayout`

`Sources/DeepSeekKit/KVCacheLayout.swift:25`. Computes the per-layer
byte offsets ahead of time from a `ModelConfig`. One
`LayerOffsets` per main layer + one per MTP layer; each contains
optional `Region` (offset + bytes) for `attnKVCache`,
`compressorKVState`, `compressorScoreState`, `indexerKVCache`,
`indexerCompressorKVState`, `indexerCompressorScoreState`.

Page-aligned to 16 KiB (Apple Silicon page size); costs up to 16 KiB
of slack per layer.

### Manifest

The KVCacheFile also stores a manifest (`<file>.manifest`) carrying
the *token sequence* the cache was built against. `readManifest()` /
`writeManifest(_:)` / `readManifestFull()` (which adds optional
`lastLogits` + `chunkAlignment`).

The manifest is what makes resume safe: when reopening, the host
re-tokenises the conversation history, compares against the saved
manifest's tokens (`commonPrefixLength`), and either:
- **Full match**: cache is exactly what we need, no prefill needed.
- **Partial match (length P)**: rewind the KV cache to position P
  (`Transformer.rewindKVTo(pos: P)`), then prefill only the delta.
- **Mismatch**: discard the cache, cold prefill from scratch.

`KVCacheFile.attemptResume(newTokens:)` returns the matching outcome.

### Where this is wired

The desktop app uses one `KVCacheFile` per conversation, stored
under `Application Support/.../conversations/<id>.kvcache`. The CLI
doesn't use it (always cold-prefills). See
`PersistencePaths.swift` for the file layout.

---

## 7. Optional warm-up

`Transformer.load(..., warmupOnLoad: true)` calls
`loader.warmupAllShards()` between the index + the block assembly.
This pre-faults every shard's pages into RAM so the first forward
doesn't pay per-tensor page-fault latency.

Auto-skipped when `model size > physical RAM × 1.5` (warm-up of a
file bigger than RAM is pointless — the OS will evict).

CLI exposes via `--warmup-on-load`; the GUI's Loading Settings tab
has a toggle.

---

## 8. Streaming-mode hooks in the forward pass

`Transformer.forward(...)` (`Sources/DeepSeekKit/Model.swift:214`)
checks `weightLoader?.streamingEnabled` and calls:

```swift
for (k, layer) in layers.enumerated() {
    loader?.ensureLayer(k)                   // (1) page in (or pool-load) shard K
    let cmdL = Device.shared.queue.makeCommandBuffer()!
    x = layer(x, startPos: ..., inputIds: ..., in: &cmdL)
    cmdL.commit(); cmdL.waitUntilCompleted()
    loader?.releaseLayer(k)                  // (2) MADV_DONTNEED or schedule prefetch
}
```

(1) `ensureLayer(K)` in pool mode kicks the rotating-slot `pread` if
the slot for `K mod N` doesn't currently hold layer K; in mmap mode
it's a `MADV_WILLNEED` hint.

(2) `releaseLayer(K)` in pool mode schedules a background prefetch
of layer K+N; in mmap mode it's `MADV_DONTNEED` on K-1's shard (one
layer back so the GPU's current reads aren't disturbed).

An earlier revision called `MADV_WILLNEED` on K+1 *before* computing
K. That backfired: the kernel started pulling K+1's pages while
K-1's `MADV_DONTNEED` hadn't been honoured yet → ~3 layers
simultaneously resident → OOM on 16 GB Macs. Letting the natural
page-fault path handle the next layer keeps residency strictly to
"the layer the GPU is currently reading".

---

## 9. Mapping CLI flags and GUI controls

```
swift run deepseek <model-dir> "prompt" \
    [--load-strategy auto|preload|mmap|streaming] \
    [--force-load] \
    [--warmup-on-load] \
    [--max-seq-len N] \
    [--max-batch-size N]
```

| Flag | Effect |
|---|---|
| `--load-strategy` | Override `LoadPlan.decide`'s automatic choice. |
| `--force-load` | Bypass the `shardTooLarge` refusal (does NOT bypass `kvCacheTooLarge` — that one is hard). |
| `--warmup-on-load` | Pre-fault every shard's pages. |
| `--max-seq-len` | Lower KV-cache rows per layer to fit in RAM. |
| `--max-batch-size` | Same, batch dimension. |

GUI counterparts (Settings → Loading tab):
- Strategy picker (`auto` / `preload` / `mmap` / `streaming`).
- Force-load toggle.
- Warm-up toggle.
- (Model Config tab) override fields write to
  `Application Support/.../config-overrides.json`.

---

## 10. Diagnostics

The loader prints a multi-line summary on stderr by default:

```
system: 16.00 GB unified (CPU + GPU share this pool)
        8.54 GB effective budget for this process
        8 cores · GPU rec. working-set 10.50 GB (same pool)
checkpoint: 46 shards, 142.30 GB total, largest 3.42 GB
oversubscription: 16.7× of effective budget
strategy: streaming (auto: total is 16.7× effective budget (cap 10×) — streaming with per-layer madvise hints)

Indexed 1843 tensors across 46 shard(s).
Projected KV cache: 0.04 GB at max_seq_len=4096, max_batch_size=4.
…
load:start → load:after-mmap → load:embed+head-built → load:layers-built → load:complete
```

Each transition is a `MemoryLogger.snapshot(...)` call gated by
`DEEPSEEK_MEM_LOG=1`. Useful when investigating "where did the
working set spike?" — every transition has the matching VM stat
dumped.

---

## 11. Source map

| Topic | File |
|---|---|
| Strategy decision + refusal guards | `Sources/DeepSeekKit/LoadStrategy.swift` |
| System probes (RAM, GPU, cores) | `Sources/DeepSeekKit/SystemProbe.swift` |
| Loader (mmap / preload mode) | `Sources/DeepSeekKit/WeightLoader.swift` |
| SafeTensors mmap reader | `Sources/DeepSeekKit/SafeTensors.swift` |
| Streaming pool | `Sources/DeepSeekKit/StreamingPool.swift` |
| Weight tree assembly | `Sources/DeepSeekKit/Assembly.swift` |
| Config inference from tensor shapes | `Sources/DeepSeekKit/Config.swift` (`inferred(from:)`) |
| KV cache file format | `Sources/DeepSeekKit/KVCacheFile.swift` |
| KV cache layout (byte offsets) | `Sources/DeepSeekKit/KVCacheLayout.swift` |
| Persistence path layout | `Sources/DeepSeekUI/Utility/PersistencePaths.swift` |
| MemoryLogger | `Sources/DeepSeekKit/MemoryLogger.swift` |
| CLI flag plumbing | `Sources/deepseek/main.swift` |
| GUI loading-related settings | `Sources/DeepSeekUI/Views/Settings/LoadingSettingsTab.swift` |

---

## 12. Failure modes and recovery

The recurring user-visible errors:

| Symptom | Cause | Resolution |
|---|---|---|
| `largest shard is X GB which exceeds the conservative cap of Y GB` | One shard exceeds 50% of budget | Free memory, re-shard the checkpoint smaller (`--shard-size-gb 2`), or `--force-load`. |
| `projected KV cache is X GB but only Y GB available` | `max_position_embeddings` × `max_batch_size` blows the budget | Edit `config.json`: lower one or both. No `--force-load` for this one. |
| `no .safetensors files in <dir>` | Wrong directory, or download failed | Verify the path; rerun `huggingface-cli download`. |
| `all safetensors files were LFS pointers` | git LFS not pulled | `git lfs pull` or re-download via huggingface-cli. |
| Loader logs "N tensor name(s) were not found … filled with random init" | Partial / pruned checkpoint | Verify download completed; check for `*-of-NNNNN` shards missing. The model still runs but produces garbage at the affected layers. |
| First token takes minutes | `.streaming` strategy on a 16 GB Mac, cold pages | Expected on first turn; subsequent tokens are much faster (rotating slot stays warm). Optionally `--warmup-on-load`. |
| "no Metal device" / Intel Mac | Hardware unsupported | Apple Silicon is required for local inference. Remote (OpenRouter) still works. |
| 100% memory pressure, system freezing | Multiple mmap'd buffers competing with GUI | Force `.streaming`, lower the working set via env vars, or restart with fewer apps open. |

The `force-load` flag bypasses **only** the `shardTooLarge` guard.
The `kvCacheTooLarge` guard is genuinely undefeatable (the runtime
allocates `storageModeShared` MTLBuffers up-front; there's no
streaming path that can help). Lower the seq-len / batch-size or
re-quantize.
