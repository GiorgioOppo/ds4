# DeepSeek V4 backend — engine architecture

This document describes the operational **DeepSeek V4 backend** behind
DwarfStar: how its GGUF is opened, how text becomes DeepSeek tokens, how tokens
flow through its Metal decode graph, how routed experts are streamed from SSD,
and how the service layer turns logits back into a chat stream. Architecture
detection and the rules shared with future backends are documented separately
in [`ARCHITETTURE-SUPPORTATE.md`](ARCHITETTURE-SUPPORTATE.md).

The goal of the port is behavioral fidelity to upstream `ds4.c` / `ds4_metal.m`,
while integrating cleanly with Swift, SwiftUI, actors, `Network.framework`, and
the macOS sandbox.

Cross-links:

- [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md) — app usage, panels, workflows.
- [`ARCHITETTURE-SUPPORTATE.md`](ARCHITETTURE-SUPPORTATE.md) — model-family
  detection, backend boundaries and current support matrix.
- [`PIPELINE-INFERENZA.md`](PIPELINE-INFERENZA.md) — request lifecycle,
  ownership of state, prefill and decode.
- [`BACKEND-METAL.md`](BACKEND-METAL.md) — runtime, wrappers, generated
  kernels and numerical-validation rules.
- [`INFERENZA-DISTRIBUITA.md`](INFERENZA-DISTRIBUITA.md) — horizontal and
  vertical topologies and protocol v11.
- [`../README.md`](../README.md) — repository overview and build commands.
- [`../README.md#configuration-reference`](../README.md#configuration-reference)
  — the authoritative list of every `DS4_*` knob mentioned below, with defaults
  and tuning guidance.
- [`../Sources/DS4Demo/README.md`](../Sources/DS4Demo/README.md) — CLI demo and
  runtime knobs.

## Contents

1. [End-to-End Pipeline](#1-end-to-end-pipeline)
2. [Model Loading: GGUF](#2-model-loading-gguf)
3. [Tokenizer](#3-tokenizer)
4. [Model Shape and Validation](#4-model-shape-and-validation)
5. [Metal Execution Substrate](#5-metal-execution-substrate)
6. [Decoder: Full Forward Pass](#6-decoder-full-forward-pass)
7. [Decode Layer Details](#7-decode-layer-details)
8. [NSA Compressor](#8-nsa-compressor)
9. [MoE: Router and Experts](#9-moe-router-and-experts)
10. [StreamingDecoder and Streaming Strategies](#10-streamingdecoder-and-streaming-strategies)
11. [Sampling: Logits to Token](#11-sampling-logits-to-token)
12. [Quantization](#12-quantization)
13. [Per-Layer Tensor Summary](#13-per-layer-tensor-summary)
14. [Tool Calling](#14-tool-calling)
15. [Distributed Inference](#15-distributed-inference)
16. [C-to-Swift Cross-Reference](#16-c-to-swift-cross-reference)

## 1. End-to-End Pipeline

```text
User text
  -> ChatRenderer / tokenizer chat template
  -> token ids
  -> embedding
  -> StreamingDecoder
       layer 0
       layer 1
       ...
       layer N-1
  -> output norm + output head
  -> logits
  -> Sampler
  -> token id
  -> detokenized bytes
  -> visible text / reasoning / DSML tool call
```

At the application level, `InferenceService` owns the decoder and exposes async
streams. The service tracks the exact committed token ids in the conversation.
When a user sends a new turn, only the suffix not already in KV is prefixed.
When tools are called, their results are appended as tool-result turns and the
decode continues from the same append-only state.

The CLI demo uses the same lower-level path without chat state: it opens a GGUF,
constructs a `StreamingDecoder`, optionally prefills a prompt, then greedily
decodes tokens.

## 2. Model Loading: GGUF

`Sources/DS4Core/Formats/GGUF/` contains the GGUF types, binary cursor, and
mapped model used to parse metadata and tensor descriptors. The
model file is opened with mmap so tensor data can be referenced without copying.

Core responsibilities:

- parse GGUF header and version;
- parse key/value metadata;
- parse tensor metadata: name, shape, type, offset, byte size;
- expose typed lookup by tensor name;
- support no-copy Metal mapping and expert slab gathering;
- provide tokenizer tables from metadata.

The loader is intentionally conservative. If the file does not contain expected
metadata or tensor names, the failure should happen before expensive decode work.
The CLI audit mode (`DS4_TYPES_ONLY=1`) prints representative tensor dtypes,
special token ids, and prompt ids for this reason.

### Loading Weights into `GPUTensor`

There are two broad weight categories:

| Category | Runtime treatment |
|---|---|
| Non-routed dense weights | No-copy mmap views by default, optionally streamed through a `pread + F_NOCACHE` staging ring with `DS4_DENSE_STREAM=1`, or copied to resident GPU buffers with `DS4_RESIDENT_DENSE=1` on RAM-rich systems. |
| Routed expert weights | Gathered per token/layer, optionally cached in expert slots, optionally read through `pread + F_NOCACHE`, optionally served from a contiguous expert-bundle sidecar. |

`GGUFWeights` builds `LayerWeights` values that reference the tensors required
by each layer. Expert tensors may remain unloaded as full tensors while still
being available for gather, because the gather path reads selected expert slabs
from the mapped GGUF tensor.

Mixed-precision routed expert layers are supported by storing quantization on
`LayerWeights` (`gateQuant`, `upQuant`, `downQuant`) instead of assuming one
global expert quantization for every layer.

## 3. Tokenizer

`Sources/DS4Core/Tokenization/Backends/DeepSeekV4/DeepSeekV4Tokenizer.swift` is
the pure-Swift tokenizer used by the current DeepSeek app, demo, server,
diagnostics, and tests. The historical public name `Tokenizer` remains an alias
for source compatibility; it is not the architecture-neutral tokenizer API.

### Components

- vocabulary and token bytes loaded from GGUF metadata;
- merge/rank data for BPE behavior;
- special token ids such as BOS, EOS, user, assistant, thinking markers, and
  DSML;
- fallback byte handling;
- chat-prompt rendering helpers manually aligned with the reference template;
  the GGUF Jinja text is exposed for diagnostics but is not interpreted at
  runtime.

### Public API

Typical calls:

- `Tokenizer(model:)` constructs from a `GGUFModel`;
- `encodeChatPrompt(system:prompt:think:)` renders a simple chat prompt;
- `tokenizeRenderedChat(_:)` tokenizes already-rendered template text;
- `tokenText(_:)` returns token bytes for detokenization.

The diagnostics panel opens the GGUF only for tokenizer metadata and prints the
same native tokenization path used by inference. No subprocess tokenizer remains.

## 4. Model Shape and Validation

`DS4Core/Model/Backends/DeepSeekV4/DeepSeekV4Configuration.swift` describes and
validates both known profiles. After validation,
`DS4Metal/Backends/DeepSeekV4/Architecture/DSV4RuntimeGeometry.swift` turns the
selected profile and its per-layer metadata into immutable instance-owned
dimensions. `DSV4Shape.swift` remains only as the source-compatible Flash
facade for legacy callers; model-aware decoder construction does not use it to
override a Pro file.

`ModelConfig(model:)` validates the metadata like the C loader
(`config_validate_model`): every shape-defining field (layer count, head dims,
LoRA ranks, expert counts, hash-layer count, indexer geometry, HC counts) must
match a known profile exactly. Engine-level metadata is cross-checked against
the selected profile — per-layer compression ratios against the expected
formula, per-layer SwiGLU clamps, RoPE scaling parameters,
`expert_weights_scale`, `expert_weights_norm`, and the
RMS/HC epsilons (both `1.0e-6`, matching the C reference). A mismatch refuses
the file at load instead of silently decoding with different math.

Shape selection distinguishes the Flash and Pro profiles by exact match. Flash
uses 43 layers, width 4096, 64 heads and 256 experts; Pro uses 61 layers, width
7168, 128 heads and 384 experts. The local service and CLI bind a
`DSV4RuntimeGeometry` before constructing weights, scratch, KV, cache and
decoder. The router accepts either 256 experts or a 512-lane bitonic dispatch
padded above Pro's 384 real experts, and applies the profile scale (1.5 or 2.5).
This is the supported local path for the single-file Pro Q2 GGUF. The Pro Q4
package remains download-only because it consists of two shards, while Pro
distributed execution remains under verification.

Tensor-level validation happens at load too: the hash-routing table
(`ffn_gate_tid2eid.weight`) is required on hash-routed layers with its layout
checked, the router bias layout is validated, and
`GGUFWeights.validateRoutedExperts` verifies routed expert tensors against the
detected quantization classes (mixed-precision layers are counted and decoded
per layer, bypassing the single-class expert cache).

The shape layer exists to keep constants centralized without making them
global runtime state. Decoder code reads dimensions from its geometry or
`DSV4Dims`; adding a static Flash fallback in a model-aware path is a regression.

## 5. Metal Execution Substrate

### `MetalRuntime`

`DS4Metal/Runtime/Core/MetalRuntime.swift` owns:

- the selected `MTLDevice`;
- command queue creation;
- embedded kernel source compilation;
- function lookup by kernel name;
- small runtime self-tests.

Kernels are embedded into `KernelSources.swift`, generated from `metal/*.metal`.
This makes the packaged app independent of a runtime kernel-source directory.

### `GPUTensor`

`GPUTensor` is the engine's basic data object:

- shape and element-count metadata;
- Metal buffer ownership or no-copy view;
- byte size and dtype interpretation;
- helper construction for mapped GGUF ranges and allocated scratch.

Swift ARC owns the object lifetime. This removes C/ObjC bridged-handle free
classes of bugs from the port.

### `GraphContext`

`GraphContext` wraps one command-buffer sequence and the transient state used by
kernel dispatches. In the streaming path, route/attention commits wait for
completion — the CPU must read back the selected expert ids — which also avoids
races where an expert slot is evicted while a previous command buffer is still
using it.

The routed-FFN command buffer is the exception: with `DS4_ASYNC_FFN` (default
on) it is committed *without* a CPU wait (`commitAsync`). The next layer's route
commit+wait lands on the same in-order queue, so correctness is guaranteed by
queue order while the CPU encode of layer i+1 overlaps the GPU execution of
layer i's FFN — the per-layer encode bubble (43 Flash layers or 61 Pro layers)
disappears. The in-flight buffer is explicitly drained at end of token (before
the output head readback / `readHC` / KV export) and on every error path,
including the prefill's. `DS4_ASYNC_FFN=0` restores the synchronous commit.

`DS4_PROFILE_ROUTE=1` deliberately adds extra phase boundaries to split
`route/attn` timing, and keeps the synchronous FFN wait so per-phase attribution
stays accurate. Those timings are diagnostic: ratios are useful, absolute
throughput is not representative because extra synchronization is introduced.

## 6. Decoder: Full Forward Pass

At a high level, one token forward pass does:

1. embed the token;
2. initialize or update Hyper-Connection state;
3. for each layer:
   - route / attention / compressor pre-work;
   - gather selected experts or hit expert cache;
   - run shared FFN and routed MoE;
   - apply residual and HC update;
4. output normalization;
5. output head projection;
6. return logits.

`StreamingDecoder.forward(token:pos:nKeys:)` drives this for decode.
`StreamingDecoder.prefill(tokens:)` uses a layer-major implementation for prompt
tokens to amortize weight I/O.

### Hyper-Connection (HC)

DeepSeek-V4 uses Hyper-Connection state in addition to normal residual flow.
DwarfStar carries HC tensors through the graph and across distributed worker
slices. In distributed mode, the coordinator embeds tokens, then workers pass HC
state through their layer ranges and return final state/logits as needed.

## 7. Decode Layer Details

Each layer is represented by `DecodeLayer` and associated graph helpers.

### Phase 1 — `decodeRoute`

This phase covers the work leading to routing:

- attention normalization;
- Q/KV projections;
- rotary position embedding;
- NSA compressor update;
- indexer / sparse attention selection;
- MLA attention;
- pre-FFN normalization;
- router logits and top-k expert ids.

### MLA Attention

The attention path is DeepSeek's multi-head latent attention variant. The Swift
port separates dense projections, normalization, RoPE, compressed KV, sparse
selection, and final attention output. Metal kernels are thin wrappers around the
same operations expected by the upstream graph.

### Phase 2 — `decodeExperts`

This phase executes:

- shared FFN branch;
- routed MoE gate/up/down projections;
- fused pair-SwiGLU where available;
- down-sum over active experts;
- residual update.

### Fused HC-Reduce

The Hyper-Connection reduce tail (split + collapse + RMSNorm) runs as ONE fused
dispatch instead of three (`DS4_FUSED_HC`, default on). It runs twice per layer,
so the fusion removes roughly 170 dispatches per token. The math is unchanged;
only the RMSNorm reduction order differs (±1 ulp class). `DS4_FUSED_HC=0`
restores the unfused three-dispatch path for A/B comparison.

The selected experts are the expensive SSD-streaming part. They can come from:

- direct mmap gather;
- `pread + F_NOCACHE`;
- contiguous expert-bundle slabs;
- expert slot-cache hit;
- expert slot-cache fill on miss.

## 8. NSA Compressor

The NSA compressor is recurrent. That matters for state management: a partially
generated sequence cannot be arbitrarily rewound unless the exact KV/compressor
state is restored or rebuilt from committed ids.

The compressor path maintains compressed rows and supports the sparse attention
selection used by decode. The raw sliding-window portion can optionally be stored
as a ring via `DS4_RAW_RING=1`, because the attention only reads the latest
`nSWA` raw rows. This ring remains an `MTLBuffer` in shared Apple-silicon memory;
it does not stream KV from the SSD and must not be confused with the Disk-KV
checkpoint store. Once the chronological window wraps, a dedicated 2D Metal
kernel reorders its rows and converts F32→F16 in one dispatch rather than
splitting the window into two copy dispatches.

Decode FlashAttention sizes adaptive split-K from every visible row (raw plus
compressed):

```text
nwg = min(32, max(1, ceil((nRaw + nCompressed) / 32)))
```

All workgroup depths from 1 through 32 are valid. There is no power-of-two
rounding, so the first compressed row after a full 128-row raw window selects 5
workgroups for 129 total rows instead of jumping from 4 to 8. Setting
`DS4_ADAPTIVE_SPLITK=0` restores the historical fixed depth of 32 for A/B tests.

With `DS4_DENSE_STREAM=1` the four NSA compressor projections are diverted out
of the staging ring and kept RESIDENT (`DS4_RESIDENT_COMP`, default on; the
~0.6 GB figure is for Flash): they are read every token on 41 of 43 Flash
layers and all 61 Pro layers, the single densest repeat-read in the dense
stream. Same bytes, identical numerics;
`DS4_RESIDENT_COMP=0` restores full streaming as a tight-RAM fallback.

### NSA Indexer

The indexer selects the relevant compressed positions used by attention. Flash
uses top-512 and Pro top-1024, both supplied by the runtime geometry. The
indexer and compressor are part of why the KV state is not a simple append-only
array that can be truncated without care.

In decode, attention stays DENSE over all compressed rows until their count
exceeds a sparse threshold, mirroring the C engine
(`metal_graph_decode_indexer_sparse_threshold`): around the ~2K frontier the
sparse path's score/top-k setup dominates the smaller attention scan. The
default is 1024, overridable with
`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (allowed values 64…4096, same as
the C). Activation is checked prospectively per layer: the prospective
compressed-row count must exceed both the sparse threshold and the indexer
top-K.

The top-K selection itself (`IndexerSelect.swift`) is heap-based — a binary
min-heap of the k best indices, O(n log k) instead of a full sort — because it
runs per ratio-4 layer per token once active.

Two resource-management consequences of the activation rule:

- **Lazy indexer-scoring staging** (`DS4_LAZY_IDX`, default on, requires
  `DS4_DENSE_STREAM=1`) follows the **used** context, not the configured
  `maxKeys`. While the live ratio-4 compressed-row count cannot exceed both the
  sparse threshold and top-K, the indexer *scoring* projections
  (`indexer.attn_q_b` + `indexer.proj`) are absent from the per-token staging
  plan. With the defaults (threshold 1024, top-K 512), live keys 4096 and 4099
  remain dense; key 4100 is the first boundary that can require scoring. At
  that boundary the scorer projections are loaded once into resident Metal
  buffers and reused for the rest of the session, instead of adding about
  360 MB of SSD reads to every earlier token. The indexer *compressor* pair
  keeps running from token one because its recurrent state must stay coherent.
  This is lossless: only the load time and storage location change. `=0`
  restores always-stage behaviour for A/B.
- **High-water attention scratch** starts from the rows needed by the live
  sequence and grows geometrically only when a new high-water mark is reached.
  The required row count is the current raw sliding window
  (`min(liveKeys, nSWA)`) plus the prospective ratio-4 compressed rows and a
  small safety margin; growth is capped by the configured maximum. Raising the
  context limit therefore no longer commits the maximum attention/indexer
  scratch on the first question. Existing buffers are retained between tokens,
  so steady-state decode does not allocate on every step.
- **Checkpoint restore bound-check**: restoring a KV/compressor checkpoint
  validates every length coming from the file against the live tensor
  capacities *and* against the emission schedule (`count <= maxKeys / ratio`,
  not the allocation, which carries slack rows) before any memcpy — a corrupt
  checkpoint can never restore a row count no legitimate run can reach.

## 9. MoE: Router and Experts

The router chooses active experts per token and layer. Flash and Pro both use 6
active experts. The routed expert tensors dominate model size and SSD I/O.

### Hash Routing and Selection Bias

Two routing details track the C reference exactly:

- The first `nHashLayer` (3) layers do not route from probabilities at all:
  they route by TOKEN ID through the `ffn_gate_tid2eid.weight` table
  (I32, `[6 x nVocab]`; the kernel clamps the token to `rows - 1`). The table
  is REQUIRED at load on those layers, with its layout validated. Because the
  selection depends only on the token id, these layers' expert I/O can always
  be resolved ahead of time (see the slot-cache look-ahead below).
- `exp_probs_b.bias` (F32 `[nExperts]`, optional per layer) shifts the router
  probabilities for SELECTION only — the routing weights applied to the expert
  outputs are computed from the unbiased probabilities, as in
  `layer_topk_selected_experts_from_probs`.

The router finalize path receives `nExperts` and `expertWeightScale` from the
active geometry. Flash dispatches a 256-thread bitonic sort with scale 1.5;
Pro dispatches 512 threads, pads lanes 384...511 to negative infinity and uses
scale 2.5. Both produce the same top-6 contract without reading beyond the
probability row.

### Fused MoE Kernels

Fused kernels reduce dispatch overhead and intermediate memory:

- pair-SwiGLU fusion;
- down projection plus sum over active experts;
- quantized expert formats q4_K, q2_K, iq2_xxs.

`DS4_FUSED_MOE=0` disables this path for debugging and A/B comparisons. It is not
a normal performance setting and can change rounding.

### Configurable Active Experts

`DS4_ACTIVE_EXPERTS=1...6` changes `DSV4Dims.activeExperts`. This intentionally
changes computation and output. It is useful for low-RAM experiments or profiling
the cost of expert I/O, not for quality-preserving optimization.

## 10. StreamingDecoder and Streaming Strategies

`StreamingDecoder` is the concrete decode engine used by both CLI and app. It
owns layer weights, KV state, usage statistics, profile counters, and optional
expert cache.

### Layer-Major Prefill

`prefill(tokens:chunk:)` processes prompt tokens in chunks of
`DS4_PREFILL_CHUNK` tokens (default 512). For each chunk, it loads each layer
once and applies it to all tokens in the chunk. This is the opposite of naive
token-major prefill, where every prompt token would reload the same layer
weights; layer-major prefill is essential for long prompts. A larger chunk
amortizes the per-chunk dense reload over more tokens at the cost of transient
activation memory.

Within one layer the work is split in two phases (`batchedExpertLayer`):

- **Phase A — routes.** Attention is causal within the layer (token j attends
  KV written by tokens 0..j), so routes stay token-sequential — but they no
  longer need a CPU round-trip each: runs of up to `DS4_PREFILL_ROUTE_BATCH`
  tokens (default 32) are encoded into ONE command buffer, each token's FFN
  inputs and router selection blit-copied GPU-side into a staging area, and
  the CPU reads all selections after a single wait. Indexer-active tokens
  (which need a CPU top-k mid-route) and `DS4_PROFILE_ROUTE` fall back to the
  per-token path.
- **Phase B — expert FFN.** Tokens are grouped so that each group's UNION of
  selected experts stays below `DS4_PREFILL_UNION` (default 192, never below
  the active-expert count); the union is gathered ONCE and every token's FFN
  runs over it with remapped ids. With `DS4_PREFILL_FFN_BATCH` (default on)
  all of a group's token-FFNs are encoded into ONE command buffer — one
  commit+wait per group instead of one per token, which removes tens of
  thousands of GPU syncs per chunk. Serial encoding keeps dispatch order and
  numerics identical; `=0` restores the per-token path for A/B parity checks.

`DS4_PREFILL_MM=1` (opt-in) additionally runs each group's routed FFN through
batched `mul_mm_id` matrix-matrix kernels instead of per-token matvecs, so
expert weights are read once per group rather than once per token. The
matrix-matrix accumulation order differs from the matvec path, which is why it
stays opt-in until validated A/B.

Numerically the batched pipeline is identical to the per-token path (a token's
FFN does not feed other tokens within the layer); only the expert I/O is
deduplicated — at most `min(6·tokens, 256)` expert reads per layer per group
instead of `6·tokens`.

### Expert Slot-Cache

`ExpertSlotCache` is a per-layer LRU cache for expert slabs. It supports:

- fixed slot budgets;
- minimum effective budget of 8 slots when enabled;
- usage-driven allocation across layers (uniform mode for A/B via
  `DS4_EXPERT_CACHE_UNIFORM=1`);
- pre-warming from persisted usage imatrix.

The cache is a wired-memory tradeoff. It helps only when routing is concentrated
enough that hot experts repeat. The Tuning tab exposes hit-rate and concentration
so this can be measured rather than guessed.

**Interleaved pool layout** (`DS4_POOL_INTERLEAVE`, default on): each slot holds
its expert's gate|up|down slabs CONTIGUOUS in one buffer — the same record
layout as the expert-bundle sidecar — so a miss served from the bundle becomes
ONE ~7 MB pread straight into the slot (1 syscall instead of 3, larger I/Os at
the same queue depth). Kernels do not change: gate/up/down are three views of
the same buffer and the stride between experts is the record size. `=0`
restores the historical three-narrow-buffer layout.

**Fill path**: on a miss without the bundle, the three slabs of an expert are
read CONCURRENTLY (three parallel jobs per miss), and `DS4_PREAD_SPLIT=N`
(default 1, max 8) further splits each slab into N disjoint ranges pread in
parallel on the `F_NOCACHE` path — decode misses are few per layer, and raising
the NVMe queue depth is what keeps the disk at its parallel ceiling.

**Concurrency**: operations are serialized per layer (one lock per layer), so
the decode thread's `acquire(layer: i)` can run while a background prefill
fills layer i+1. The demand path has priority: a running speculative prefill
fills in chunks and yields between chunks when a demand acquire is waiting, so
speculation can delay the critical path by at most about one fill chunk. A
speculative fill never evicts the slots of the last demand acquire (they may
still be read by an in-flight command buffer).

**Speculative look-ahead** (`kickLookahead`): at the start of layer i (and of
each token), the decoder resolves layer i+1's likely expert ids on the decode
thread and kicks their pool prefill on a background queue, so the fill I/O runs
under layer i's compute — the C engine's `begin_selected_load` trick. For the
hash-routed layers (0–2) the ids are EXACT (a `tid2eid` mmap read from the
token id), so their expert I/O is always hidden; for the other layers the guess
is the usage-prior top-N, opt-in with `DS4_EXPERT_LOOKAHEAD=N` (default 0; a
wrong guess wastes idle-window bandwidth only). Prefilled slabs do not count as
misses in the stats — their I/O ran off the critical path. Mixed-precision
layers (outside the cache's size class) are excluded.

### Expert Bundle Sidecar

`DS4_EXPERT_BUNDLE=1` adds a disk-side optimization for cache misses. The engine
looks for or builds a sidecar (`<gguf>.expbundle`, 4 KB-aligned records ordered
by layer then expert id) where each expert's gate, up, and down slabs are stored
contiguously. A miss can then be satisfied by one ~7 MB sequential read instead
of three scattered ~2 MB reads from the original GGUF tensor layout — and with
the interleaved slot pool the record layout matches the slot layout, so the
copy is a single pread into the slot.

The sidecar is not a new quantization and does not change math. It duplicates
the expert byte region on disk, is validated against the source model
(size/geometry plus per-layer content fingerprints), and is skipped when
writable space is insufficient. In sandboxed app builds, `DS4_BUNDLE_DIR`
points creation at Application Support; a readable sidecar next to the GGUF can
still be reused. Every load logs the bundle state, and use is proven at runtime
by a logarithmic heartbeat (first expert served, then 5k, 10k, 20k, …).

### Dense Streaming and Resident Dense Weights

Dense attention/shared weights are always needed, so page-cache eviction can
dominate low-RAM decode even when expert I/O is optimized. The engine supports
three strategies:

| Strategy | Knob | Notes |
|---|---|---|
| mmap/page cache | default | Minimal wired memory; best when RAM can keep hot dense pages resident. |
| dense streaming | `DS4_DENSE_STREAM=1` | Reads each layer's dense tensors into a small staging ring (`pread + F_NOCACHE`) one layer ahead of compute, so the SSD read of layer i+1 overlaps the GPU compute of layer i. ~300 MB of staging instead of ~6 GB resident. Takes precedence over resident dense. |
| resident dense | `DS4_RESIDENT_DENSE=1` | Copies dense weights into resident GPU buffers. Useful on RAM-rich systems, risky on 16 GB. |

`DS4_DENSE_AHEAD` controls staging read-ahead depth (default 1, the classic
2-slot ring; clamped to at most 3). Larger depths can improve overlap but also
compete with expert reads on the same SSD.

The dense stream hosts several carve-outs, weights that leave the ring because
streaming them every token is the wrong tradeoff:

- the **output head** (`output.weight`, ~560 MB Q8, read in full every token)
  is copied RESIDENT whenever `DS4_DENSE_STREAM=1` — mapped, it was re-read
  through a cold page cache every token. The embedding table stays mapped: the
  decode stages one ~8 KB row per token into a small staging buffer instead of
  wiring the whole table;
- the **NSA compressor projections** (`DS4_RESIDENT_COMP`, section 8);
- the **indexer scoring projections**, deferred by `DS4_LAZY_IDX` until the
  live sparse boundary and then kept resident (section 8);
- the **Q4-requantized projections** of `DS4_DENSE_Q4` / `DS4_SHARED_Q4`
  (section 12).

### Hot-Buffer Pinning

`DS4_MLOCK=1` requests best-effort `mlock()` on hot shared Metal buffers such as
expert-cache pools, dense-stream staging, resident output head, Q4 dense buffers
and the enabled raw-KV ring. This is a performance guard against macOS
memory-compressor churn, not a correctness requirement. Failure to pin is logged
or ignored depending on the call site; decode continues.

### Split Command Buffer Pattern

The streaming path is intentionally split around CPU-visible routing decisions:

1. run route/attention work;
2. read back selected expert ids;
3. gather/cache those experts;
4. run expert FFN work.

This makes SSD expert I/O explicit and measurable. It also means command-buffer
round-trip overhead can matter; `DS4_DIAG=1` measures empty command-buffer cost
in the CLI demo. Step 4 is the asynchronous half of the pipeline: the routed-FFN
buffer commits without a CPU wait (`DS4_ASYNC_FFN`, section 5) and the next
layer's route wait doubles as the join point, so only the route round-trip
remains synchronous.

All the `DS4_*` knobs in this section are documented with defaults in the root
[Configuration Reference](../README.md#configuration-reference).

### Constructors

The important construction families are:

| Constructor | Strategy |
|---|---|
| `fromGGUFExpertCachedMapped` | Fast SSD-streaming path: non-routed weights mapped, selected experts gathered, optional slot-cache. |
| expert-cache variants | Keep hot experts resident and fill on miss. |
| per-layer quant variants | Decode mixed expert quantization correctly by reading per-layer tensor types. |

## 11. Sampling: Logits to Token

`DS4Core/Generation/Sampler.swift` implements:

- greedy mode (`temperature = 0`);
- temperature sampling;
- top-k;
- top-p;
- min-p;
- repetition penalty.

The demo uses greedy sampling. The app exposes temperature and repetition
penalty. The service/server path uses `SamplingParams` so UI, HTTP, and
distributed coordinator can share behavior.

Repetition penalty is important for heavily quantized models because collapse
loops can occur after many generated tokens.

## 12. Quantization

The engine handles the quantized formats needed by the target GGUFs:

- Q8_0 dense matvecs;
- Q4_K routed experts;
- Q2_K routed experts;
- IQ2_XXS routed experts;
- F16/F32 scalar or normalization tensors where required.

`DS4_Q8_NSG` tunes dense Q8_0 matvec scheduling. It partitions work and partial
sums across simdgroups: the mathematical operation is unchanged, but the
floating-point reduction order can alter the last bits.

Mixed routed expert quantization is supported per layer. Uniform models keep the
same single-quantization-class selection path; out-of-class mixed layers bypass
that cache and select the gather/decode format declared for the individual
layer. This statement concerns format/path selection, not byte-identical logits
or generated tokens.

### Q4 Dense Requantization

`DS4_DENSE_Q4=1` (requires `DS4_DENSE_STREAM=1`) is a deliberate lossy speed
path. The engine requantizes the largest Q8 attention projections (`q_b`,
`output_a`, `output_b`) to Q4_K and keeps the reduced buffers resident. It
removes several GB of per-token dense traffic from the SSD path.
`DS4_SHARED_Q4=1` extends the same idea to the shared-expert FFN projections,
and `DS4_QKV_Q4=1` to the remaining mid-size attention projections (`q_a`,
`kv` — the last Q8 attention slabs still streamed, ~0.7 GB/token for ~0.35 GB
resident); both should be treated as separate A/B experiments. All three
knobs share the same `.q4dense` cache: records are matched per (layer,
tensor) key, so enabling a new knob reuses the existing cache and requantizes
only the tensors it adds.

The conversion is persisted in a **Q4 requant cache** (`<gguf>.q4dense`, about
1.4 GB for the base dense-Q4 trio and larger when QKV/shared records are
enabled): the first load pays the requant once, while later loads pread the
validated cache instead of converting the same tensors again; load time depends
on SSD, memory pressure and hardware. The requant **creates the cache file
empty up front** (valid header, zero records — a write preflight: permission,
path or disk-space problems surface in the log *before* any conversion work,
and the file's presence proves checkpoints have somewhere to land), then runs
in batches and **checkpoints the partial cache to disk between batches** (same
format, fewer records): a first load interrupted mid-requant (force quit,
crash, reboot)
resumes from the completed tensors instead of restarting from zero, and the
load bar advances per **MB of source converted** rather than per tensor, so a
long conversion is visibly progressing instead of looking hung. Validation is
against the model *bytes* (content fingerprints of each source tensor,
checked per record), not just the file size. A failed preflight or checkpoint
write is logged as `DS4 q4cache:`; writes use a temporary sibling plus rename,
so a torn file never replaces a valid cache. The current load safely continues
with converted tensors in memory, while a later load requantizes records that
were not persisted. The cache lives next to the model by
default (demo/CLI); the sandboxed app cannot write next to a picker-selected
file, so `DS4_Q4_CACHE_DIR` points it at Application Support. Reads try both
places, and a full cache found in the secondary location (e.g. produced by the
demo next to the GGUF) is PROMOTED into the primary one so demo and app share
a single conversion. Distributed layer slices read from the full cache but
write their own per-range cache name, so a slice requant can never clobber the
full cache.

Because this changes weights, it is not a parity mode. Use it for throughput,
not when comparing exact logits against a full-Q8 dense path.

## 13. Per-Layer Tensor Summary

Each decode layer uses tensor groups roughly corresponding to:

| Group | Examples |
|---|---|
| Attention norms/projections | `attn_norm`, `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a`, `attn_output_b` |
| Compressor/indexer | compressor and sparse-selection weights |
| Router | `ffn_gate_inp` or equivalent routed selection tensor |
| Shared FFN | shared gate/up/down tensors |
| Routed experts | `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` |
| HC | attention/FFN/output Hyper-Connection tensors |
| Output | final norm and output head |

`DS4_TYPES_ONLY=1` prints representative tensor dtypes for a GGUF so a mismatch
can be diagnosed before expensive decode.

## 14. Tool Calling

### Special Tokens

The tokenizer recognizes DS4 protocol tokens:

- user / assistant turn delimiters;
- think start/end;
- DSML tool-call token;
- end-of-sentence markers.

### DSML Format

DSML is the model-native tool-call format. DwarfStar's compiled `ChatRenderer`
implements the format against a reference DeepSeek-V4 template; it does not run
the GGUF's Jinja text. The compact GUI mode is enabled by default and sends a
shorter declaration for lower prefill cost, deliberately deviating from the
full training-oriented text; full mode stays closer to that reference schema.

### Rendering and Parsing

`ChatRenderer` renders:

- system prompt;
- user/assistant turns;
- tool declarations;
- tool results.

Tool-result payloads are wrapped in `<tool_result>…</tool_result>`. The
renderer deliberately does NOT HTML-escape the content (shell output and file
snippets must stay intact) but escapes the exact closing sentinel
(`escapeToolResult`, mirroring the C `bpe_tokenize_tool_result_text`): a
malicious or accidental `</tool_result>` inside a payload can therefore never
terminate the wrapper early and inject control tokens into the prompt.

`ToolCallParser` strips leaked markup and parses completed DSML blocks into
`ToolCall` values.

### Tool-Loop Orchestration

`InferenceService` streams assistant text, parses a complete DSML block and
emits `.toolCall`; it does not execute the requested tool. The application owns
the loop:

1. `ChatStore+ToolLoop` handles local chat, while `DistributedController`
   handles distributed chat;
2. the orchestrator dispatches built-ins/MCP through `ToolRegistry.executeAuto`
   and handles sub-agent or manual-result flows where supported;
3. it records the outputs and requests a continuation;
4. local chat calls `InferenceService.provideToolResults`, while distributed
   chat appends `toolResult` turns before calling the coordinator again.

### Registry and Tools

`ToolRegistry` owns built-ins and JSON schemas. Tools are split one per file for
reviewability. Project tools operate on `ProjectCache`, which is separate from
chat memory until tool results are inserted into the conversation.

### MCP Client

`DS4Engine/Tools/MCP` adds a Model Context Protocol client, so external MCP
servers appear to the model next to the built-ins:

- `MCPConfig` speaks the `{"mcpServers": …}` JSON interchange used by Claude
  Desktop / Cursor / VS Code, so configs import and export verbatim;
- `MCPTransport` implements the two spec transports: stdio (server spawned as
  a child process, newline-delimited JSON-RPC on stdin/stdout) and Streamable
  HTTP (POSTed frames, JSON or SSE responses, `Mcp-Session-Id` echoed);
- `MCPClient` owns one connection: initialize handshake, per-request timeouts,
  server pings, disconnect surfacing, and task cancellation (Stop settles an
  in-flight call immediately);
- `MCPManager` is the process-wide registry: it owns the clients and serves
  cached snapshots (statuses, namespaced `ToolSpec`s) to chat, agents, and
  distributed mode. Tool `read_file` on server `fs` becomes `mcp_fs_read_file`;
  the reverse mapping is an explicit index, never name parsing.

Tool loops never call this layer directly: `ToolRegistry.executeAuto(_:)`
dispatches built-ins first, then MCP.

### Sub-Agents

Sub-agents are implemented as an engine-side context switch:

- snapshot main KV;
- build or restore sub-agent content prefix;
- run tool loop in isolated context;
- return answer;
- restore main KV.

This lets the main conversation delegate work without ingesting every internal
file read or intermediate reasoning step.

## 15. Distributed Inference

Distributed inference splits contiguous layer ranges across workers.

### Decoder Slice API

Workers own `StreamingDecoder` instances for a layer slice. They allocate only
the weights and KV for that slice. The coordinator owns embeddings, sampling,
prompt rendering, and final orchestration.

### Protocol and Topology

The wire protocol (`Sources/DS4Engine/Distributed/Protocol/`) is
DwarfStar-native framing, currently
at version 10. The coordinator validates the version FIRST, so a mixed cluster
fails with a clear error instead of garbled frames. The version history doubles
as the feature list:

- **v2** — robustness: HELLO carries the version; WORK/RESULT carry a per-turn
  `session` id echoed by the workers, so a result left in a TCP buffer by a
  cancelled turn can never be mistaken for the next turn's reply.
- **v3** — the COORDINATOR defines each worker's job. Workers start idle
  (listening, no model loaded); the coordinator sends ASSIGN (gguf, context,
  layer slice, cache slots) and the worker replies READY once its engine is
  loaded.
- **v4** — distributed KV continuity: ASSIGN carries the usage imatrix (slot
  cache pre-warm) and a disk-KV token budget; KV control frames (`kvQuery` /
  `kvLengths` / `kvRestore` / `kvSave` / `kvAck`) let the coordinator
  checkpoint/restore each worker's shard; WORK gained `turnStart` so a turn
  can begin mid-context.
- **v5** — the coordinator DISTRIBUTES the files. After HELLO it sends a FILE
  OFFER (name, size, SHA-256 for gguf + sidecars); the worker answers with what
  it is missing (hash-verified against its store) and only that is streamed —
  the huge setup runs once, later connects verify hashes from cached manifests
  in milliseconds.
- **v6** — derived caches travel too: the offer can include the Q4 dense
  requant cache (`<gguf>.q4dense`), and ASSIGN carries the Q4 on/off decision.
- **v7** — WORK carries the chunk's token ids: the first 3 layers route experts
  by TOKEN ID (`ffn_gate_tid2eid`), so a shard covering them cannot route from
  the HC state alone.
- **v8** — RESUMABLE transfers: the offer carries a chained-hash checkpoint
  list per file (one SHA-256 every 256 MB, each folded over the previous, so
  `chain[k]` commits to the whole prefix); the worker keeps its `.part` across
  disconnects and sessions, validates it block-by-block, truncates to the last
  good checkpoint and answers FILE NEED with a per-file resume offset. The
  coordinator retries a broken peer setup up to 3 times.
- **v9** — ASSIGN carries the coordinator's PERFORMANCE KNOBS: a whitelisted
  set of `DS4_*` env vars (`Dist.perfKnobKeys`). The whitelist prevents the wire
  from setting arbitrary environment; it is not a numerical-parity contract.
  Allowed fusion, batching and prefill-MM paths may change floating-point
  reduction/accumulation order. `DS4_DENSE_Q4`, deliberately lossy, travels as
  a separate typed field with its cache. The worker applies the coordinator's
  job configuration before loading, without promising bit-identical results
  across hardware or execution paths.
- **v10** — expert parallelism: `expertAssign`, `expertWork` and `expertSum`
  let workers own masks over the 256 routed experts across every layer. The
  coordinator runs the dense backbone and combines remote partial FFN sums.
  Vertical chat and benchmark are wired in the app; the topology requires a
  low-latency wired link because it performs a network round-trip per routed
  layer.
- **v11** — model-owned distributed geometry: horizontal slices validate 43
  Flash or 61 Pro layers, while `expertAssign` carries a length-prefixed mask
  for 256 Flash or 384 Pro experts. `READY` reports the geometry actually
  loaded. The full Pro Q2 GGUF is accepted; split Pro Q4 still requires a
  multi-shard loader.

Peer setup runs IN PARALLEL: file transfer and engine load of every worker
proceed together in a task group, so route activation costs `max(worker
setup)` instead of the sum. Route order stays the peer-list order, and
contiguous coverage is still validated after assembly.

Each token or prefill chunk sends HC state through the route. In relay mode,
the coordinator round-trips through every worker. In forwarding mode, workers
pass state to the next worker and the terminal worker returns to the
coordinator listener.

Distributed benchmark reuses the already-connected coordinator. It must not run
while distributed chat is generating, because both share the same route and reset
worker KV.

The full operational flow, file-resume behavior and topology comparison are in
[`INFERENZA-DISTRIBUITA.md`](INFERENZA-DISTRIBUITA.md).

## 16. C-to-Swift Cross-Reference

| Upstream C Area | Swift Port |
|---|---|
| `ds4.c` model shape, tokenizer use, decoder flow | `DS4Core/Model/Backends/DeepSeekV4`, `DS4Core/Tokenization/Backends/DeepSeekV4`, `DS4Metal/Backends/DeepSeekV4/Decode`, `DS4Engine/Inference/Service` |
| `ds4_metal.m` runtime and kernels | `DS4Metal/Runtime`, `DS4Metal/Kernels`, `metal/*.metal` |
| GGUF parsing | `Sources/DS4Core/Formats/GGUF/` |
| SSD streaming expert model | `Sources/DS4Metal/Backends/DeepSeekV4/Weights/GGUFWeights.swift`, `StreamingDecoder` |
| Dense streaming / Q4 requant cache | `Sources/DS4Metal/Backends/DeepSeekV4/Streaming/DenseStreamer.swift` |
| Expert cache | `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Cache/`, usage imatrix in `StreamingDecoder` |
| Expert bundle sidecar | `Sources/DS4Metal/Backends/DeepSeekV4/Experts/ExpertBundle.swift` |
| KV store | `Sources/DS4Core/Formats/KVCheckpoint/KVCFile.swift`, `Sources/DS4Engine/Persistence/KV/DiskKVStore.swift` |
| Server | `Sources/DwarfStar/Features/Server`, `LocalServer` |
| Distributed runtime | `DS4Engine/Distributed` |
| MCP client (no C counterpart) | `DS4Engine/Tools/MCP` |
| CLI bring-up | `Sources/DS4Demo/Command/main.swift` |

The Swift port deliberately replaces C-specific memory ownership with ARC,
Foundation parsing where appropriate, Swift actors for service isolation, and
SwiftUI state models for the app. Kernel math and model-visible behavior remain
the parts that must track upstream most closely.
