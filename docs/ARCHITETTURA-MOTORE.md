# Engine Architecture — DeepSeek V4 (`DS4Core` + `DS4Metal`)

This document describes the pure-Swift engine behind DwarfStar: how a GGUF model
is opened, how text becomes tokens, how tokens flow through the Metal decode
graph, how routed experts are streamed from SSD, and how the service layer turns
logits back into a chat stream.

The goal of the port is behavioral fidelity to upstream `ds4.c` / `ds4_metal.m`,
while integrating cleanly with Swift, SwiftUI, actors, `Network.framework`, and
the macOS sandbox.

Cross-links:

- [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md) — app usage, panels, workflows.
- [`../README.md`](../README.md) — repository overview and build commands.
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

`DS4Core/Format/GGUF.swift` parses GGUF metadata and tensor descriptors. The
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

`DS4Core/Inference/Tokenizer.swift` is the pure-Swift tokenizer used by the app,
demo, server, diagnostics, and tests.

### Components

- vocabulary and token bytes loaded from GGUF metadata;
- merge/rank data for BPE behavior;
- special token ids such as BOS, EOS, user, assistant, thinking markers, and
  DSML;
- fallback byte handling;
- chat-prompt rendering helpers aligned with the GGUF template.

### Public API

Typical calls:

- `Tokenizer(model:)` constructs from a `GGUFModel`;
- `encodeChatPrompt(system:prompt:think:)` renders a simple chat prompt;
- `tokenizeRenderedChat(_:)` tokenizes already-rendered template text;
- `tokenText(_:)` returns token bytes for detokenization.

The diagnostics panel opens the GGUF only for tokenizer metadata and prints the
same native tokenization path used by inference. No subprocess tokenizer remains.

## 4. Model Shape and Validation

`DS4Core/Inference/ModelShape.swift` and `DS4Metal/Model/DSV4Shape.swift`
capture the DeepSeek-V4 Flash constants used by the port: layer count, hidden
dimensions, expert counts, attention dimensions, sliding-window size, and related
metadata.

`ModelShape.fromGGUF` validates shape metadata against known model families.
Tests cross-check shape selection with real GGUF metadata.

The shape layer exists to keep hardcoded constants centralized. Decoder code
should read dimensions from the shape or `DSV4Dims`, not scatter magic numbers
through graph stages.

## 5. Metal Execution Substrate

### `MetalRuntime`

`DS4Metal/Runtime/MetalRuntime.swift` owns:

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
kernel dispatches. In the current streaming path, commits wait for completion.
This makes the pipeline simpler and avoids races where an expert slot is evicted
while a previous command buffer is still using it.

`DS4_PROFILE_ROUTE=1` deliberately adds extra phase boundaries to split
`route/attn` timing. Those timings are diagnostic: ratios are useful, absolute
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
`nSWA` raw rows.

### NSA Indexer

The indexer selects the relevant compressed positions used by attention. Flash
uses a top-k indexer path (for example top-512 in the ported constants). The
indexer and compressor are part of why the KV state is not a simple append-only
array that can be truncated without care.

## 9. MoE: Router and Experts

The router chooses active experts per token and layer. By default Flash uses 6
active experts. The routed expert tensors dominate model size and SSD I/O.

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

`prefill(tokens:chunk:)` processes prompt tokens in chunks. For each chunk, it
loads each layer once and applies it to all tokens in the chunk. For routed FFN,
it gathers the union of experts required by that chunk.

This is the opposite of naive token-major prefill, where every prompt token would
reload the same layer weights. Layer-major prefill is essential for long prompts.

### Expert Slot-Cache

`ExpertSlotCache` is a per-layer LRU cache for expert slabs. It supports:

- fixed slot budgets;
- minimum effective budget of 8 slots when enabled;
- usage-driven allocation across layers;
- pre-warming from persisted usage imatrix;
- uniform allocation mode for A/B testing.

The cache is a wired-memory tradeoff. It helps only when routing is concentrated
enough that hot experts repeat. The Tuning tab exposes hit-rate and concentration
so this can be measured rather than guessed.

### Expert Bundle Sidecar

`DS4_EXPERT_BUNDLE=1` adds a disk-side optimization for cache misses. The engine
looks for or builds a sidecar where each expert's gate, up, and down slabs are
stored contiguously. A miss can then be satisfied by one sequential read instead
of three scattered reads from the original GGUF tensor layout.

The sidecar is not a new quantization and does not change math. It duplicates
the expert byte region on disk, is validated against the source model, and is
skipped when writable space is insufficient. In sandboxed app builds,
`DS4_BUNDLE_DIR` points creation at Application Support; a readable sidecar next
to the GGUF can still be reused.

### Dense Streaming and Resident Dense Weights

Dense attention/shared weights are always needed, so page-cache eviction can
dominate low-RAM decode even when expert I/O is optimized. The engine supports
three strategies:

| Strategy | Knob | Notes |
|---|---|---|
| mmap/page cache | default | Minimal wired memory; best when RAM can keep hot dense pages resident. |
| dense streaming | `DS4_DENSE_STREAM=1` | Reads each layer's dense tensors into a small staging ring one layer ahead of compute. Takes precedence over resident dense. |
| resident dense | `DS4_RESIDENT_DENSE=1` | Copies dense weights into resident GPU buffers. Useful on RAM-rich systems, risky on 16 GB. |

`DS4_DENSE_AHEAD` controls staging read-ahead depth. Larger depths can improve
overlap but also compete with expert reads on the same SSD.

### Hot-Buffer Pinning

`DS4_MLOCK=1` requests best-effort `mlock()` on hot shared Metal buffers such as
expert-cache pools, dense-stream staging, resident output head, and Q4 dense
buffers. This is a performance guard against macOS memory-compressor churn, not
a correctness requirement. Failure to pin is logged or ignored depending on the
call site; decode continues.

### Split Command Buffer Pattern

The streaming path is intentionally split around CPU-visible routing decisions:

1. run route/attention work;
2. read back selected expert ids;
3. gather/cache those experts;
4. run expert FFN work.

This makes SSD expert I/O explicit and measurable. It also means command-buffer
round-trip overhead can matter; `DS4_DIAG=1` measures empty command-buffer cost
in the CLI demo.

### Constructors

The important construction families are:

| Constructor | Strategy |
|---|---|
| `fromGGUFExpertCachedMapped` | Fast SSD-streaming path: non-routed weights mapped, selected experts gathered, optional slot-cache. |
| expert-cache variants | Keep hot experts resident and fill on miss. |
| per-layer quant variants | Decode mixed expert quantization correctly by reading per-layer tensor types. |

## 11. Sampling: Logits to Token

`DS4Core/Inference/Sampler.swift` implements:

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

`DS4_Q8_NSG` tunes dense Q8_0 matvec scheduling. It partitions work across
simdgroups; it does not change mathematical results.

Mixed routed expert quantization is supported per layer. Uniform models remain
byte-identical in behavior, while out-of-class mixed layers bypass single-class
expert cache and use correct gather/decode paths.

### Q4 Dense Requantization

`DS4_DENSE_Q4=1` is a deliberate lossy speed path. The engine requantizes the
largest Q8 attention projections (`q_b`, `output_a`, `output_b`) to Q4_K,
caches the result, and keeps the reduced buffers resident. It removes several GB
of per-token dense traffic from the SSD path. `DS4_SHARED_Q4=1` extends the same
idea to shared-expert FFN projections and should be treated as a separate A/B
experiment.

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

DSML is the model-native tool-call format. DwarfStar renders tool definitions
from the model's chat-template surface. The compact GUI mode sends a shorter
declaration for lower prefill cost; the full mode is closer to the training
schema.

### Rendering and Parsing

`ChatRenderer` renders:

- system prompt;
- user/assistant turns;
- tool declarations;
- tool results.

`ToolCallParser` strips leaked markup and parses completed DSML blocks into
`ToolCall` values.

### InferenceService Loop

The service loop:

1. streams assistant text until a DSML call is complete;
2. parses tool calls;
3. executes built-ins automatically;
4. collects manual results for unknown tools if needed;
5. appends tool results;
6. resumes generation.

### Registry and Tools

`ToolRegistry` owns built-ins and JSON schemas. Tools are split one per file for
reviewability. Project tools operate on `ProjectCache`, which is separate from
chat memory until tool results are inserted into the conversation.

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

The coordinator connects to workers, reads their hello frames, sorts them by
`layerStart`, and validates contiguous coverage. Each token or prefill chunk
sends HC state through the route. In relay mode, the coordinator round-trips
through every worker. In forwarding mode, workers pass state to the next worker
and the terminal worker returns to the coordinator listener.

Distributed benchmark reuses the already-connected coordinator. It must not run
while distributed chat is generating, because both share the same route and reset
worker KV.

## 16. C-to-Swift Cross-Reference

| Upstream C Area | Swift Port |
|---|---|
| `ds4.c` model shape, tokenizer use, decoder flow | `DS4Core`, `DS4Metal/Decode`, `DS4Engine/Service` |
| `ds4_metal.m` runtime and kernels | `DS4Metal/Runtime`, `DS4Metal/Kernels`, `metal/*.metal` |
| GGUF parsing | `DS4Core/Format/GGUF.swift` |
| SSD streaming expert model | `DS4Metal/Model/GGUFWeights.swift`, `StreamingDecoder` |
| Expert cache | `ExpertSlotCache`, usage imatrix in `StreamingDecoder` |
| KV store | `DS4Core/Format/KVCFile.swift`, `DS4Engine/Service/DiskKVStore.swift` |
| Server | `Sources/DwarfStar/Server`, `LocalServer` |
| Distributed runtime | `DS4Engine/Distributed` |
| CLI bring-up | `Sources/DS4Demo/main.swift` |

The Swift port deliberately replaces C-specific memory ownership with ARC,
Foundation parsing where appropriate, Swift actors for service isolation, and
SwiftUI state models for the app. Kernel math and model-visible behavior remain
the parts that must track upstream most closely.
