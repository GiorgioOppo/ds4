# Modules reference

Per-file index of `Sources/`. Each entry: purpose, key public API,
dependencies, and any cross-references. Use as a directory index when
deciding which file to open.

For Metal kernels see [`KERNELS.md`](KERNELS.md). For the
Python correspondence see [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md).
For contributor recipes (how to add a kernel/layer/test) see
[`DEVELOPING.md`](DEVELOPING.md). For the test inventory see
[`TESTING.md`](TESTING.md).

---

## `Sources/DeepSeekKit/` (top-level)

### `Device.swift`
Singleton wrapping `MTLDevice`, command queue, and the lazy-loaded
`default.metallib` (compiled by `MetalLibPlugin`). The converter only
touches `Device.shared.mtl` for `MTLBuffer` creation — the library is
loaded only on first `makePipeline` call.

Public: `Device.shared`, `Device.shared.mtl`, `Device.shared.queue`,
`Device.shared.library`, `Device.shared.makePipeline(_:)`.

### `Config.swift`
`ModelConfig`: Codable struct mirroring `ModelArgs` from
`Reference/inference/model.py:34`. Field names use Python snake_case via
`CodingKeys`. `ModelConfig.load(from:)` reads `config.json`.

Used by: every layer constructor, the assembly loader, the CLI.

### `Tensor.swift`
`Tensor` = shape + dtype + `MTLBuffer` + offset. Row-major. No autograd,
no broadcasting. `Tensor.empty(shape:dtype:)` allocates a fresh
`MTLBuffer`; `Tensor.from(bytes:shape:dtype:)` copies host bytes into a
new `MTLBuffer`; safetensors return tensors that share an mmapped
buffer.

`DType` enum: `f32`, `f16`, `bf16`, `i32`, `i8`, `fp8E4M3`, `fp4E2M1`,
`e8m0`. `Tensor.reshape(_:)` returns a same-buffer view.

`Tensor.toFloatArray()` is a host-side debug helper; supports
`f32`/`f16`/`bf16` only.

### `Quantization.swift`
Constants for the FP8/FP4/E8M0 block layouts plus scalar host-side
`dequantE4M3` / `dequantE2M1` / `dequantE8M0` helpers used by tests and
the converter. Documents the exact safetensors layout the loader
expects.

### `YaRN.swift`
`YaRN.precomputeFreqsCis(...)` returns a flat `[seqlen][rope_dim/2][2]`
(cos, sin) array. Pure Swift port of
`Reference/inference/model.py:199-229`. Called once per layer by `MLA`.

### `SafeTensors.swift`
`SafeTensorsFile`: mmap-backed reader. `init(url:)` opens the file with
`MAP_PRIVATE`, wraps the whole mapping as one `MTLBuffer` via
`makeBuffer(bytesNoCopy:)`. `load(_:)` returns a `Tensor` referencing
the shared buffer with the right offset.

The mmap is deallocated when the buffer is released (closure passed as
`makeBuffer` deallocator).

### `SafeTensorsWriter.swift`
Streaming writer used by the converter. Three source types:
- `.data(Data)` — already-in-memory bytes
- `.file(url:offset:byteCount:)` — copy directly from another file in
  64 MB chunks
- `.compute(byteCount:closure)` — lazy producer, called when the writer
  reaches that tensor

### `WeightLoader.swift`
Walks a directory of `.safetensors` shards, builds a flat name → shard
index, exposes `load(_:)` and `tryLoad(_:[fallbackNames])`. Skips LFS
pointer files (< 1 KB). Used by `Transformer.load(from:)`.

### `Tokenizer.swift`
Protocol `Tokenizer { encode/decode/bosId/eosId }` and
`TokenizerLoader.load(from:)` that constructs a `BPETokenizer` from a
`tokenizer.json` file.

### `BPETokenizer.swift`
Byte-level BPE compatible with HuggingFace `tokenizer.json`. Parses
`model.vocab`, `model.merges`, `added_tokens`, plus the pre-tokenizer
regex. GPT-2 byte-to-unicode map for UTF-8 round-trip. Greedy
lowest-rank merge BPE.

### `KVCache.swift`
Allocators for per-layer KV caches and a `CacheBank` aggregating them.
Currently unused at runtime — MLA holds its `kvCache` field directly,
populated in assembly.

### `Sampling.swift`
- `Sampler.argmax(_:)` — GPU-side reduction, returns a single Int.
- `Sampler.applyTemperature(_:)` — in-place GPU scaling.
- `Sampler.sample(_:history:options:)` — full pipeline (temperature →
  repetition penalty → top-K → top-P → Gumbel-max multinomial), done
  host-side after a single CPU-GPU sync.
- `SamplingOptions` struct carries the parameters + RNG state across
  decode steps.

### `Generation.swift`
`Generator` + `GenerationOptions` are the OO wrapper around the
generation loop. **Currently unused** — the CLI in
`Sources/deepseek/main.swift` open-codes the prefill+decode loop. Kept
as a future API surface.

### `Model.swift`
Three classes:

- `ParallelEmbedding` — embed table + `lookup([Int32])` returning
  `[N, dim]`.
- `ParallelHead` — final RMSNorm + HC head collapse + LM head matmul,
  producing logits on the LAST sequence position only.
- `Transformer` — assembled model; `forward(inputIds:startPos:)` runs
  embed → HC expand → all blocks → head, returning `[B, vocab]` logits.

### `Assembly.swift`
Two factory methods on `Transformer`:

- `Transformer.randomInit(config:)` — builds every module with small
  random F32 weights. End-to-end smoke testing without weights on
  disk.
- `Transformer.load(config:from:)` — walks the canonical V4 weight
  name tree, pulls each tensor via `WeightLoader`, builds the
  modules. Falls back to random init for missing names and prints a
  summary.

The canonical weight name list lives in the docstring.

---

## `Sources/DeepSeekKit/Layers/`

Each Layer file is a Swift wrapper around one or more Metal kernels in
`Kernels/`. Convention: every layer has a `callAsFunction(...)` or
similar public method, and (where it makes sense) a `referenceCPU(...)`
pure-Swift implementation used by tests.

### `Linear.swift`
`Linear(in:out:weight:scale:)`. Dispatches based on `weight.dtype`:
- `.bf16` → `gemm_bf16_to_f32` (or `gemm_f32_bf16_to_f32` if input is f32)
- `.f32`  → `gemm_f32_to_f32`
- `.fp8E4M3` → act_quant + `gemm_fp8_to_f32`
- `.fp4E2M1` → act_quant + `gemm_fp8_fp4_to_f32`

Output is always `f32`.

### `RMSNorm.swift`
`RMSNorm(weight:eps:)` + `callAsFunction(_:in:)`. One kernel
(`rmsnorm_f32`), one threadgroup per row.

### `RoPE.swift`
`RoPE(ropeHeadDim:freqs:)`. `apply(_:startPos:inverse:in:)` does
in-place rotary on the trailing `rope_head_dim` of each head. Freqs are
pre-baked by `YaRN.precomputeFreqsCis` at model construction. The
`inverse: true` path is used for the output projection's de-rotation.

### `Hadamard.swift`
`Hadamard.apply(_:in:)` — in-place Walsh-Hadamard transform on the last
axis. Requires power-of-2 dim. Used by Indexer and Compressor (when
`rotate == true`).

### `ActQuant.swift`
`ActQuant(format: .fp8 | .fp4)`. `quant(_:inplace:in:)`. Block-wise
activation quantization, with `inplace=true` performing a round-trip
(dequant immediately) so the buffer stays f32 — used for QAT noise
injection.

### `Elementwise.swift`
Bundle of small elementwise ops as static methods: `siluMul`, `axpy`,
`scale`, `addInPlace`. Used everywhere by Compressor / MoE / MLA /
HyperConnections.

### `SparseAttention.swift`
`SparseAttention.apply(q:kv:sink:topkIdxs:scale:in:)` — sparse multi-
head attention with FlashAttention-style online softmax and KV gather
by topk index. One thread per `(b, m, h)`.

### `HCSinkhorn.swift`
`HCSinkhorn(hcMult:sinkhornIters:hcEps:)`. `split(mixes:hcScale:hcBase:in:)`
produces `(pre, post, comb)` from a `[N, mix_hc]` mixing tensor.
Used inside `HyperConnections.pre`.

### `HyperConnections.swift`
`HyperConnections(config:dim:)`. Two methods:
- `pre(x:hcFn:hcScale:hcBase:in:)` → `(y[N, dim], post, comb)`. Collapses
  `[N, hc, dim]` to `[N, dim]` for the sublayer input.
- `post(x:residual:post:comb:in:)` → `[N, hc, dim]`. Re-expands.

### `SoftmaxAxis.swift`
`SoftmaxAxis.apply(_:axis:in:)` — softmax along any axis of an N-D
tensor. Used by Compressor's per-window softmax and indirectly by
Indexer.

### `TopK.swift`
`TopK.apply(_:k:in:)` → `(values, indices)`. Top-K along last axis.
In-register heap; max k = 32.

### `MoEDispatch.swift`
`MoEDispatch.prepare(...)` builds the host-side permutation tables from
gate output. `MoEDispatch.gather(...)` groups tokens by expert.
`MoEDispatch.scatter(y:outs:plan:in:)` does the weighted sum of
expert outputs back into the dense `[N, dim]` output.

### `OverlapTransform.swift`
`OverlapTransform.apply(_:padValue:in:)` — Compressor's overlap shuffle
(`[B, S, ratio, 2D] → [B, S, 2*ratio, D]`).

### `Einsum.swift`
Two specialized einsums used by V4 attention:
- `Einsum.bshdBtd(q:kv:in:)` → `[B, S, H, T]` (Indexer score)
- `Einsum.bsgdGrd(o:woA:in:)` → `[B, S, G, R]` (MLA grouped output)

### `AttentionIndices.swift`
Pure host-side helpers (no kernel):
- `slidingWindow(windowSize:batch:seqlen:startPos:)` → `[Int32]` window
  topk indices for both prefill (`startPos == 0`) and decode (wrap).
- `compressed(ratio:batch:seqlen:startPos:offset:)` → compressed-token
  topk indices for layers without an Indexer (`ratio == 128`).

### `Compressor.swift`
`Compressor(config:compressRatio:...)`. `callAsFunction(_:startPos:in:)`
returns the compressed KV when one is emitted, else nil.

Prefill (`startPos == 0`) processes the whole sequence in one shot.
Decode accumulates per-token into `kvState`/`scoreState` buffers and
emits one compressed token every `compressRatio` steps. Overlap
(`compressRatio == 4`) maintains a double-window state via
`compressor_overlap_concat_f32` + `compressor_state_shift_copy_f32`.

### `Indexer.swift`
`Indexer(config:compressRatio:wqB:weightsProj:compressor:kvCache:)`.
`callAsFunction(_:qr:startPos:offset:in:)` → `[B, S, K]` Int32 topk
indices into the compressed-KV cache. Used only by layers with
`compress_ratio == 4`.

### `Attention.swift`
`MLA` struct + `callAsFunction(_:startPos:in:)`. The full attention
forward (prefill or decode):

1. Low-rank Q (`wq_a → q_norm → wq_b → rsqrt re-norm`) + RoPE on rope tail
2. KV (`wkv → kv_norm`) + RoPE on rope tail
3. Window topk + optional compressed topk (Indexer or
   `AttentionIndices.compressed`)
4. KV cache write (full prefill, cutoff/wrap for `seqlen > window`, or
   ring-buffer single row for decode)
5. Compressor call (writes compressed tokens into the trailing KV cache slice)
6. `SparseAttention.apply`
7. Inverse RoPE on output
8. Grouped output via `Einsum.bsgdGrd` + `wo_b`

### `DecoderLayer.swift`
`Block(layerId:config:attn:ffn:...)`.
`callAsFunction(_:startPos:inputIds:in:)`:

```
residual = x
(y, post, comb) = HC.pre(x, hc_attn_fn, ...)
y = attn_norm(y)
out = attn(y, startPos)
x = HC.post(out, residual, post, comb)

residual = x
(y, post, comb) = HC.pre(x, hc_ffn_fn, ...)
y = ffn_norm(y)
out = ffn(y, inputIds)
x = HC.post(out, residual, post, comb)
return x
```

### `MoE.swift`
Three types in one file:

- `Gate(config:layerId:weight:bias:tid2eid:)` — top-k gating with
  `sqrtsoftplus` / `sigmoid` / `softmax` score func via function
  constants, plus the hash-routing branch for early layers.
- `Expert(w1:w2:w3:swigluLimit:)` — SwiGLU FFN expert.
- `MoEFFN(config:gate:experts:shared:)` — orchestrator. Builds the
  dispatch plan via `MoEDispatch.prepare`, runs each active expert on
  its assigned tokens, scatters back, adds the shared expert.

### `MTPBlock.swift`
`MTPBlock(block:eProj:hProj:...)`. Forward fuses the next-token
embedding with the previous block's hidden state, runs `Block.forward`,
then routes through `ParallelHead` for a speculative prediction.

Not yet wired into the CLI (Tier 2 building block, integration deferred).

---

## `Sources/DeepSeekKit/Encoding/`

### `Message.swift`
`Role`, `Message`, `ToolCall`, `ThinkingMode` data types.

### `EncodingDSV4.swift`
Port of `Reference/encoding/encoding_dsv4.py`. Covers:

- BOS + per-role markers (`<｜User｜>`, `<｜Assistant｜>`) + EOS.
- `<think>...</think>` reasoning blocks for `.high` / `.max` modes.
- Tool calls in DSML format (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke ...>`,
  `<｜DSML｜parameter ...>`).
- Optional `REASONING_EFFORT_MAX` system prompt and `TOOLS_TEMPLATE`
  block, auto-merged into the first system message.

`encodeMessages(_:mode:toolSchemasJSON:)` produces the prompt string.
`parseCompletion(_:mode:)` parses the model output back into a
`Message`, extracting any `<think>` and tool-call blocks.

Deferred: task tokens, response_format injection, latest_reminder
(simple string concatenation; do it caller-side until needed).

---

## `Sources/deepseek/main.swift`

CLI for token generation. Flags: `<model-dir> <prompt> [--mode raw|chat]
[--max-tokens N] [--temperature T]`.

Loads config, tokenizer, and model (via `Transformer.load`). Encodes
prompt with `EncodingDSV4` (chat mode) or as-is (raw). Runs prefill,
then decode loop using `Sampler.sample` with `SamplingOptions`. Streams
in raw mode; buffers + parses in chat mode.

## `Sources/converter/main.swift`

CLI that ports `Reference/inference/convert.py`. Reads HF safetensors,
applies the rename mapping (`self_attn → attn`, `mlp → ffn`,
`weight_scale_inv → scale`, etc.), fuses FP8/FP4 + scale into BF16/F16
where requested, packs the output into layer-aligned shards capped at
`--shard-size-gb`, writes `model.safetensors.index.json`, copies the
tokenizer files. Resume-safe: scans the output dir before writing and
skips shards already present at the right size.

Helpers used by the converter (defined in the same file): `renameKey`,
`shouldSkip`, `floatToBF16`, `floatToF16`, `fuseFP8ToNative`,
`fuseFP4ToNative`, `isFP8DType` / `isFP4DType` / `isE8M0DType`.

---

## `Tests/DeepSeekKitTests/`

One XCTest per kernel / module that has a GPU implementation paired
with a CPU reference. Convention: `*Tests.swift` files compare
Metal output vs `referenceCPU(...)` on small randomized inputs.

| Test file | What it covers |
|---|---|
| `HadamardTests.swift` | FWHT correctness + involution (H∘H = I) |
| `HCSinkhornTests.swift` | Sinkhorn-normalized comb output + doubly-stochastic |
| `ActQuantTests.swift` | FP8/FP4 round-trip + scale match |
| `SoftmaxAxisTests.swift` | Softmax along arbitrary axis |
| `TopKTests.swift` | Top-K values + indices, k = 1 special case |
| `MoEDispatchTests.swift` | gather + scatter round-trip identity |
| `OverlapTransformTests.swift` | Compressor shuffle, pad behavior on s=0 |
| `EinsumTests.swift` | `bshd,btd→bsht` + `bsgd,grd→bsgr` |
| `AttentionIndicesTests.swift` | sliding window + compressed indices |
| `LinearTests.swift` | f32 GEMM + FP8 GEMM relative bound |
| `SparseAttentionTests.swift` | full forward + all-padding zero case |
| `HyperConnectionsTests.swift` | pre + post round-trip |
| `CompressorTests.swift` | prefill no-overlap match vs CPU |
| `BPETokenizerTests.swift` | encode/decode round-trip on UTF-8 + special tokens |
| `MoEHashRoutingTests.swift` | hash routing emits correct expert ids from tid2eid |
