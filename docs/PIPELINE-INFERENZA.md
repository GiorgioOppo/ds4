**English** | [Italiano](PIPELINE-INFERENZA.it.md)

# Inference pipeline

This document follows a request from the GUI or the API all the way to the
token shown to the user. For the mathematical details of the individual layers
see [ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.md); for file locations see
[STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.md).

## Overview

```text
Chat / HTTP / benchmark
        |
        v
InferenceService (state, prefill/decode and event emission)
        |
        +--> ChatRenderer + Tokenizer
        |
        +--> KV prefix reuse or restore
        |
        +--> prefill of the new suffix only
        |
        +--> decode loop: forward -> logits -> sampler -> token
        |
        +--> text / reasoning / tool call / metrics events
                         |
                         v
        ChatStore+ToolLoop / DistributedController
        tool execution -> results -> inference resumption
```

The GUI, the server and the local benchmark share a single
`InferenceService` instance. This avoids loading two copies of the resident
buffers, the Q4 cache and the GPU scratch, an essential requirement on Macs
with 16 GB.

## 1. Model loading

1. `GGUFModel` opens the GGUF and validates header, metadata and tensor
   descriptors.
2. `ModelConfig` and `DSV4Shape` recognize the supported profile and reject
   incompatible shapes before decode starts.
3. `Tokenizer` reads vocabulary, merges and control tokens from the GGUF.
4. `GGUFWeights` prepares the mapped weight views and the layer providers.
5. `StreamingDecoder` picks the requested memory path: resident or streamed
   dense weights, expert cache, bundle, `pread` or MetalIO.
6. `InferenceService` publishes progress and logs and makes the engine
   available to chat, server and benchmark.

Toggles that change weight layout or residency are read at load time: after a
change the model must be reloaded. The exceptions that can be updated between
two prefills are documented in the
[Configuration Reference](../README.md#configuration-reference).

## 2. Conversation preparation

`ChatRenderer` turns system, turns, tool specs and results into DSML through a
Swift implementation built on the reference DeepSeek-V4 template. The runtime
does not interpret the Jinja embedded in the GGUF: that metadata is available
for diagnostics, while the default compact tool mode intentionally shortens
the declaration to reduce the prefill. `Tokenizer` converts the rendered text
into token ids. The shared types (`ChatTurn`, `ToolSpec` and `ToolCall`) live
in `DS4Core`, so the service does not introduce a second conversation format.

Before the prefill the service compares the rendered ids with the ones already
committed:

- if the new prompt exactly extends the current prefix, it processes only the
  suffix;
- if a compatible checkpoint exists, `DiskKVStore` restores it and continues
  from the saved prefix;
- after a stop or a divergence it rebuilds the state from the safe prefix.

The recurrence of the NSA compressor cannot simply be rewound. This is why the
visible conversation state and the committed KV state are kept separate until
the turn concludes cleanly.

## 3. Prefill

The prefill processes many already-known tokens. The path is organized by
layer and by chunk:

1. the route/attention stage prepares a group of tokens;
2. the layer's dense weights are loaded once per chunk;
3. the experts requested by the tokens are merged into groups bounded by
   `DS4_PREFILL_UNION`;
4. the FFN applies the experts to the tokens involved;
5. raw KV, compressed cache and recurrent state advance in order.

`DS4_PREFILL_CHUNK`, `DS4_PREFILL_UNION` and `DS4_PREFILL_ROUTE_BATCH`
respectively control dense-weight amortization, transient memory and the
number of synchronizations. The prefill is not a decode repeated in a loop: it
uses dedicated structures under
`DS4Metal/Backends/DeepSeekV4/Decode/Prefill`.

With `DS4_PREFILL_BATCH_ATTN` (default on) the route/attention stage runs ONE
multi-query FlashAttention per route-batch run — the C engine's prefill kernel
(8 query rows per threadgroup, simdgroup MMA) over the union KV span, with a
per-query mask reproducing exactly the per-token visibility (causal + SWA
window + compressed rows emitted up to each token) — instead of one
single-query vec dispatch per token. Same math over the same visible keys;
only the accumulation order differs, so outputs are close but not
bit-identical to the per-token path (`=0` restores it for A/B parity). Runs
that would overflow the `DS4_RAW_RING` raw cache fall back automatically.

On top of that, `DS4_PREFILL_DENSE_MM` (default on) turns EVERY dense
projection of the batched run — q_a, q_b, kv, the grouped low-rank attention
output, output_b, the router — plus both HyperConnection reduces into
matrix-matrix kernels over the whole run: dense weights are read once per
route batch instead of once per token, which is the C engine's layer-major
prefill shape. Per-token work shrinks to the recurrent NSA compressor state
update, the fp8 KV row store and the router top-6 finalize (the compressor
kv/score projections are batched too). Each projection dispatches on its
RESIDENT quant — Q8_0/F16 by default, Q4_K (`kernel_mul_mm_q4_K_f32`) when
`DS4_DENSE_Q4`/`DS4_QKV_Q4` requantized it — so the batched prefill stays
active in the decode-oriented resident-Q4 profile. `=0` restores the
per-token dense projections for A/B. The profile counters
`prefillFlashRuns`/`prefillDenseRuns` report how many runs took each path.

With `DS4_PREFILL_FULL_LAYER` (default on) and a chunk of at least
`DS4_PREFILL_FULL_LAYER_MIN` tokens (512), phase B streams the WHOLE routed
layer once — all experts, GLOBAL ids, no unions, no remap — and
double-buffers the next layer's slab during this layer's FFNs (the C
engine's prefill shape; ~2×1.7 GiB transient, released after the prefill).
Bytes/token become layerBytes/chunkTokens, so long chunks pay: measured
26.8 MB/token at a 4096-token chunk vs ~155 MB/token on the union path,
with the exposed gather time dropping to ~0. The MM paths' routed tail is
batched too (unit-weight collapse + rows-add + one HC expand per run,
bit-identical to the per-token tail). Best measured prefill env on the
2-bit Flash: `DS4_PREFILL_ROUTE_BATCH=128 DS4_PREFILL_CHUNK=4096
DS4_PREFILL_MM=1 DS4_EXPERT_PREAD=1 DS4_PREAD_SPLIT=4` — 26 tok/s on a
2.7k-token real prompt (M1 Pro), from 3.6 before this work.

## 4. Decoding one token

For each token the decoder runs all layers in order:

1. embedding and initial HC state;
2. normalization and Q/KV projections;
3. NSA compressor and attention;
4. MoE router or hash table of the first layers;
5. gather of the six selected routed experts;
6. routed and shared FFN, HC reduction and hand-off to the next layer;
7. final norm and output head;
8. sampling and detokenization.

`StreamingDecoder+LayerExecution.swift` orchestrates the layer, while
`Graph/*` composes smaller GPU operations. The wrappers under `Kernels/*`
must remain free of application policy.

## 5. Expert streaming

The full MoE model does not have to reside in RAM. For each layer only the
selected experts are read. The strategies can be combined:

- LRU slot cache pre-warmed from the usage imatrix;
- direct `pread` reads with `F_NOCACHE`;
- sidecar bundle with contiguous gate/up/down per expert;
- MetalIO with circuit breaker and fallback to `pread`;
- exact or history-guided prefetch;
- streamed dense weights with an independent staging ring.

The cache speeds up the gather but does not change router, ids or expert
weights. The lossy paths are declared explicitly in the configuration
(`DENSE_Q4`, `QKV_Q4`, `SHARED_Q4`, `COMP_Q8`, reduction of the active
experts).

## 6. Sampling, reasoning and tools

`Sampler` applies temperature, top-k, top-p, min-p and repetition penalty. The
resulting token is converted into bytes without assuming that every token is a
complete UTF-8 string.

The service events distinguish:

- visible text;
- reasoning content;
- streaming and completion of a tool call;
- progress and metrics;
- completion, stop and error.

When the DSML parser recognizes a call, `InferenceService` emits it as a
`.toolCall` event: it does not execute the tool. On the local path the loop is
orchestrated by `ChatStore+ToolLoop.swift`; on the distributed path by
`DistributedController`. These consumers invoke `ToolRegistry.executeAuto`,
collect the results and start the continuation. The local path passes them to
`InferenceService.provideToolResults`; the distributed one appends the
`toolResult` turns and re-invokes generation on the coordinator.

## 7. State ownership

| State | Owner |
|---|---|
| Conversations, agent selection and UI settings | `ChatStore` |
| Local/distributed tool loop | `ChatStore+ToolLoop` / `DistributedController` |
| Shared decoder, committed tokens and active generation | `InferenceService` |
| KV, scratch, expert cache and GPU profiling | `StreamingDecoder` |
| Persistent checkpoints | `DiskKVStore` |
| DSML rendering and conversation types | `DS4Core/Conversation` |
| Sampling | `DS4Core/Generation` |

Rule of thumb: the GUI must not mutate buffers or KV directly; the Metal
backend must not know about sessions, views or the HTTP protocol.

## 8. Code map

- `Sources/DS4Core/Conversation` — models, DSML rendering and parsing.
- `Sources/DS4Core/Tokenization` — BPE and byte-level detokenization.
- `Sources/DS4Core/Generation` — sampler.
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode` — prefill, decode, KV and
  caches of the currently operational backend.
- `Sources/DS4Metal/Backends/DeepSeekV4` — DeepSeek Metal shape, weights,
  experts and streaming; shared runtime and kernels stay outside the backend.
- `Sources/DS4Engine/Inference` — API and application actor.
- `Sources/DS4Engine/Persistence/KV` — on-disk checkpoints.
- `Sources/DwarfStar/Features/Chat` — chat state and presentation.

## 9. Verifying changes

A change to the pipeline requires, in proportion to the level touched:

1. pure tests for tokenizer, renderer, sampler or protocol;
2. parity tests for wrappers and the Metal graph;
3. a Release demo build;
4. comparison with identical prompt, seed and configuration;
5. separate measurement of prefill and decode, after warm-up.

The full procedure is in [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md).
