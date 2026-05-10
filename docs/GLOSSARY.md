# Glossary

Domain terms used throughout the codebase. Each entry has a one-line
definition, the file/line where it lives in code, and a link to the
doc that explains it in depth.

If you're an LLM reading the project for the first time, open this
first: every acronym below appears unexplained in source-file
docstrings.

## At a glance

| Term | One-liner | See |
|---|---|---|
| MLA | Multi-head Latent Attention: low-rank Q + shared KV + grouped O | [Architecture](ARCHITECTURE.md#what-the-model-is), `Sources/DeepSeekKit/Layers/Attention.swift` |
| MoE | Mixture-of-Experts FFN; top-K routed + 1 shared | [Architecture](ARCHITECTURE.md), `Sources/DeepSeekKit/Layers/MoE.swift` |
| HC | Hyper-Connections; replaces residual with Sinkhorn-mixed `hc_mult` copies | `Sources/DeepSeekKit/Layers/HyperConnections.swift` |
| Compressor | Gated softmax pooling over `compress_ratio` tokens; writes compressed KV | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| Indexer | Top-K learned KV-position selector for `compress_ratio == 4` layers | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| MTP | Multi-Token Prediction speculative head | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| FP4-E2M1 | 4-bit float: 1 sign + 2 exp + 1 mantissa | [DTYPES](DTYPES.md#fp4-e2m1) |
| FP8-E4M3 | 8-bit float: 1 sign + 4 exp + 3 mantissa | [DTYPES](DTYPES.md#fp8-e4m3fn) |
| E8M0 | 8-bit unbiased exponent block scale (MX spec) | [DTYPES](DTYPES.md#e8m0) |
| BF16 | bfloat16: F32 truncated to top 16 bits | [DTYPES](DTYPES.md#bf16) |
| RoPE | Rotary Position Embedding (interleaved cos/sin per pair) | `Sources/DeepSeekKit/Layers/RoPE.swift` |
| YaRN | RoPE frequency scaling for long context | `Sources/DeepSeekKit/YaRN.swift` |
| BPE | Byte-level Byte-Pair Encoding tokenizer | `Sources/DeepSeekKit/BPETokenizer.swift` |
| DSML | DeepSeek Markup Language; XML-like tool-call format | `Sources/DeepSeekKit/Encoding/EncodingDSV4.swift` |
| safetensors | HuggingFace tensor file format (header JSON + raw data) | [MEMORY](MEMORY.md), `Sources/DeepSeekKit/SafeTensors.swift` |

## 1. Model architecture

### MLA — Multi-head Latent Attention

DeepSeek-V4's attention variant. Instead of computing Q/K/V as separate
big matmuls, it factors them through low-rank intermediates and shares
the KV projection across all heads.

- Q path: `wq_a` (dim → q_lora_rank=1024) → RMSNorm → `wq_b`
  (q_lora_rank → n_heads*head_dim).
- KV path: single `wkv` (dim → head_dim=512) projection, shared across
  all heads. `kv_norm` on the result.
- O path: grouped low-rank, `wo_a` (n_heads*head_dim/n_groups →
  o_lora_rank per group) → `wo_b` (n_groups*o_lora_rank → dim).
- Each head has a learnable `attn_sink` scalar absorbed into the
  softmax denominator.

Implementation: `Sources/DeepSeekKit/Layers/Attention.swift:85`.

### MoE — Mixture-of-Experts FFN

Top-K routing over `n_routed_experts` experts (256 in V4-Flash, more in
V4-Pro), plus one always-active `shared_expert`. Each expert is a
standard SwiGLU FFN (`w1 = gate_proj`, `w3 = up_proj`,
`w2 = down_proj`).

Gating: scores from `x @ gate_weight^T`, normalized through one of
three score functions: `softmax`, `sigmoid`, or `sqrtsoftplus` (V4
default). The kernel `moe_gate` (`Sources/DeepSeekKit/Kernels/moe.metal`)
implements all three via a function constant.

Implementation: `Sources/DeepSeekKit/Layers/MoE.swift`.

### HC — Hyper-Connections

Replaces the standard residual `x = x + sublayer(norm(x))` with a
mixing scheme over `hc_mult = 4` parallel copies of the hidden state.
Each block has two HC passes: one wrapping attention, one wrapping FFN.

- `HC.pre(x [N, hc, d])` → `(y [N, d], post [N, hc], comb [N, hc, hc])`.
  The mixing matrix is normalized doubly-stochastic via Sinkhorn
  iterations.
- The sublayer consumes `y`.
- `HC.post(out [N, d], residual [N, hc, d], post, comb)` →
  `[N, hc, d]`.

Implementation: `Sources/DeepSeekKit/Layers/HyperConnections.swift`
and `Sources/DeepSeekKit/Layers/HCSinkhorn.swift`.

### Compressor

Gated softmax pooling that collapses `compress_ratio` consecutive
tokens into one "compressed token" in the KV cache. Maintains state
buffers (`kvState`, `scoreState`) across decode steps and emits one
compressed token every `ratio` steps.

When `ratio == 4`, the Compressor uses overlapping windows for smoother
boundaries (each emit pulls half from the previous window and half
from the current).

Implementation: `Sources/DeepSeekKit/Layers/Compressor.swift`.

### Indexer

Top-K learned position selector. For layers with `compress_ratio == 4`,
the Indexer scores every compressed-KV position against the current
query (via its own Compressor copy + Hadamard rotation + FP4 quant) and
picks the top `index_topk` (= 512) for the actual sparse attention to
consult.

Implementation: `Sources/DeepSeekKit/Layers/Indexer.swift`.

### Sliding window attention

The per-layer KV cache is a ring buffer of `window_size = 128` rows
plus a trailing slice for compressed tokens. During decode, each new
token overwrites the slot at `startPos % window_size`. The sparse
attention reads any subset of these positions selected by the
`topk_idxs` tensor.

Implementation: ring buffer write in `Sources/DeepSeekKit/Layers/Attention.swift`
at the "KV cache write" section. Topk index generation in
`Sources/DeepSeekKit/Layers/AttentionIndices.swift`.

### Sparse attention

FlashAttention-style attention where each query only attends to a
sparse subset of positions selected by `topk_idxs`. Uses online softmax
(running max + sum) so the answer is stable even when computed
incrementally. The `attn_sink` logit is folded into the denominator at
the end.

Implementation: `Sources/DeepSeekKit/Kernels/sparse_attn.metal` +
`Sources/DeepSeekKit/Layers/SparseAttention.swift`.

### attn_sink

A learnable per-head scalar `[n_heads]` whose `exp(sink - max_score)` is
added to the softmax denominator. Lets the model "do nothing" for a
position by routing weight onto the sink rather than any actual KV.
Helps stability when no KV is strongly relevant.

### Hash routing

For the first `n_hash_layers` layers, MoE routing bypasses the score
computation and picks experts directly from a `tid2eid` lookup table
keyed by the input token id. Stabilises early-layer routing during
training and is preserved at inference.

Verified by `Tests/DeepSeekKitTests/MoEHashRoutingTests.swift`.

### MTP — Multi-Token Prediction

Trailing speculative head. After the main block stack, the MTP block
fuses the embedding of the *known* next token with the last hidden
state and runs another forward pass to predict the token *after* that.
At inference time this is used for speculative decoding (~2× throughput
when speculative predictions are accepted).

Currently the MTPBlock forward is implemented but the CLI doesn't run
speculative decoding yet — see [ROADMAP.md](ROADMAP.md#deferred-not-on-critical-path).

Implementation: `Sources/DeepSeekKit/Layers/MTPBlock.swift`.

## 2. Quantization formats

### FP4-E2M1

4-bit float. 1 sign bit + 2 exponent bits + 1 mantissa bit. Bias 1.
Values: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` plus negatives. See
[DTYPES.md](DTYPES.md#fp4-e2m1) for the full grid.

Stored two-per-byte in safetensors with dtype `"F4_E2M1"` (or
`"FLOAT4_E2M1FN_X2"`).

### FP8-E4M3 (E4M3FN variant)

8-bit float. 1 sign + 4 exponent + 3 mantissa. Bias 7. Range ±448, with
NaN encoded at `exp = 0xF, mant = 0x7` (no infinities — "FN" = "Finite
or NaN"). Stored one byte per value with dtype `"F8_E4M3"`.

### E8M0

8-bit unbiased exponent format used as block scale (MX spec). A scale
byte `b` decodes to `2^(b - 127)` as a float. `b = 0xFF` is NaN. Dtype
`"F8_E8M0"` or `"FLOAT8_E8M0FNU"`.

### ue8m0

Marker indicating that scale tensors should be rounded to the nearest
power-of-2 (so they fit exactly in E8M0 representation). The reference
uses `ue8m0` rounding for FP8 GEMM scales.

### Block-wise scaling

Each quantized weight tensor pairs with a `.scale` tensor of E8M0
values, one per block. Layouts in this project:

- FP8 weight `[out, in]` with scale `[out/128, in/128]` (per 128×128
  block).
- FP4 weight `[out, in/2]` (packed) with scale `[out, in/32]` (per row,
  per 32-element K block).

### Dequant

Converting a quantized value back to F32 by reading its byte/nibble,
looking up the float meaning, and multiplying by the block scale.
Implemented in `Sources/DeepSeekKit/Quantization.swift`
(`dequantE4M3`, `dequantE2M1`, `dequantE8M0`).

### QAT — Quantization-Aware Training

Training procedure that simulates quantization noise (quant + dequant
round-trip) on activations during the forward pass. The reference V4
inference replays this noise via in-place `act_quant(..., inplace=true)`.
Our port currently skips the QAT noise injection on non-rope KV dims
(see [ROADMAP.md](ROADMAP.md#deferred-structural-prerequisite-needed)).

## 3. Position encoding

### RoPE — Rotary Position Embedding

Each rotary pair `(x_{2i}, x_{2i+1})` is rotated by an angle that depends
on position and dim index:
`(x', x'') = (x cos θ - x' sin θ, x sin θ + x' cos θ)`.
Applied only to the trailing `rope_head_dim = 64` of each head; the
leading `nope_head_dim` stays unrotated.

Implementation: `Sources/DeepSeekKit/Layers/RoPE.swift` + `Kernels/rope.metal`.

### YaRN — Yet another RoPE extensioN

Frequency scaling for long context. Instead of rotating with bare
`θ = base^(-2i/d)`, YaRN interpolates between the un-rescaled and a
factor-divided version using a smooth linear ramp on a correction
range `[low, high]` derived from `beta_fast` and `beta_slow`.

Implementation: `Sources/DeepSeekKit/YaRN.swift:precomputeFreqsCis`.

### rope_head_dim / nope_head_dim

The total head dim splits into:
- `rope_head_dim = 64`: rotated by RoPE
- `nope_head_dim = head_dim - rope_head_dim = 448`: not rotated,
  quantization noise applied here (in the full QAT path)

### beta_fast / beta_slow

Parameters of the YaRN correction range. `beta_fast = 32` and
`beta_slow = 1` define the rotation counts whose corresponding
frequencies bound the linear ramp. See model.py:208 for the exact
formulas.

## 4. Tooling

### safetensors

HuggingFace tensor file format. Layout: `[u64 LE header_len][JSON header][tensor data...]`.
The header maps tensor name → `{dtype, shape, data_offsets: [start, end]}`.
Multiple shards form a model release; an optional `model.safetensors.index.json`
maps tensor names to shards.

See [MEMORY.md](MEMORY.md) for the mmap-backed reader, and
[USAGE.md](USAGE.md) for the converter's output shard naming.

### mmap

`mmap(2)` system call that maps a file's bytes into the process's
virtual address space. `MAP_PRIVATE` flag means writes (if any) don't
go back to disk. The OS lazily pages bytes in on first access and may
evict them under memory pressure.

Used in `Sources/DeepSeekKit/SafeTensors.swift:init(url:)` to expose
the entire 140 GB checkpoint as one `MTLBuffer` without copying.

### MTLBuffer

Metal's GPU-readable buffer. On Apple Silicon (unified memory), the
GPU reads directly from the same physical pages the CPU sees. We
create `MTLBuffer`s either from scratch (`makeBuffer(length:options:)`)
or by wrapping existing memory (`makeBuffer(bytesNoCopy:length:options:deallocator:)`).
The latter requires page-aligned input.

### MetalLibPlugin

SwiftPM build-tool plugin in `Plugins/MetalLibPlugin/`. Compiles every
`.metal` file in `Sources/DeepSeekKit/Kernels/` into a single
`default.metallib` resource, bundled with the `DeepSeekKit` target so
`Device.shared.library` can find it at runtime.

Plain `swift build` doesn't run the Metal toolchain on `.process`-ed
files, hence the plugin.

### default.metallib

Compiled output of all the kernels. The runtime calls
`device.makeDefaultLibrary(bundle: Bundle.module)` to load it. Lives
under `.build/release/.../DeepSeekKit_DeepSeekKit.bundle/` after a
release build.

### simdgroup / simdgroup_matrix

Metal abstractions for a group of 32 threads executing in lockstep. On
M3+, `simdgroup_matrix` instructions accelerate BF16/F16 matrix
multiplies inside a single SIMD group. Our current GEMM kernels do
scalar tiled accumulation (no simdgroup_matrix yet) — see
[PERFORMANCE.md](PERFORMANCE.md) for the planned upgrade.

## 5. General acronyms

### BPE — Byte-Pair Encoding

Tokenizer algorithm that learns a vocabulary of subword pieces by
iteratively merging the most frequent adjacent symbol pair. The
"byte-level" variant first maps UTF-8 bytes through a fixed table to
printable unicode (so binary safety is preserved), then operates on
those characters.

Implementation: `Sources/DeepSeekKit/BPETokenizer.swift`.

### DSML — DeepSeek Markup Language

XML-like notation for tool calls inside model output:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">cats</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
```

Implementation: `Sources/DeepSeekKit/Encoding/EncodingDSV4.swift`.

### BOS / EOS

Begin- and end-of-sentence special tokens. In DeepSeek-V4:
- BOS: `<｜begin▁of▁sentence｜>` (id 0)
- EOS: `<｜end▁of▁sentence｜>` (id 1)

Plus the role markers `<｜User｜>` and `<｜Assistant｜>`.

## Index (alphabetical)

| Term | One-liner | Defined |
|---|---|---|
| `attn_sink` | per-head softmax-denominator bias | §1 |
| `beta_fast`/`beta_slow` | YaRN correction-range bounds | §3 |
| BF16 | F32 truncated to top 16 bits | [DTYPES](DTYPES.md) |
| BOS / EOS | sentence markers in tokenizer | §5 |
| BPE | byte-level Byte-Pair Encoding | §5 |
| Compressor | gated pool of `ratio` consecutive tokens | §1 |
| `default.metallib` | output of MetalLibPlugin | §4 |
| dequant | quant byte → float via LUT/formula | §2 |
| DSML | DeepSeek markup for tool calls | §5 |
| E8M0 | 8-bit unbiased exponent block scale | §2 / [DTYPES](DTYPES.md) |
| FP4-E2M1 | 4-bit float with 8-value grid | §2 / [DTYPES](DTYPES.md) |
| FP8-E4M3 | 8-bit float, max 448, NaN at 0x7F | §2 / [DTYPES](DTYPES.md) |
| Hash routing | first-layer MoE via `tid2eid` lookup | §1 |
| HC (Hyper-Connections) | Sinkhorn-mixed `hc_mult` residual scheme | §1 |
| Indexer | top-K learned KV position selector | §1 |
| MetalLibPlugin | SwiftPM build plugin for .metal files | §4 |
| MLA | Multi-head Latent Attention | §1 |
| mmap | OS-level lazy file mapping | §4 / [MEMORY](MEMORY.md) |
| MoE | Mixture-of-Experts FFN | §1 |
| MTLBuffer | Metal's GPU buffer abstraction | §4 |
| MTP | Multi-Token Prediction speculative head | §1 |
| nope_head_dim | non-rotary part of each head | §3 |
| QAT | Quantization-Aware Training noise | §2 |
| RoPE | Rotary Position Embedding | §3 |
| rope_head_dim | rotary part of each head (64) | §3 |
| safetensors | HF tensor file format | §4 / [MEMORY](MEMORY.md) |
| simdgroup_matrix | Metal SIMD matrix instructions | §4 / [PERFORMANCE](PERFORMANCE.md) |
| Sliding window | KV cache ring of size `window_size` | §1 |
| Sparse attention | FlashAttention over topk_idxs subset | §1 |
| `ue8m0` | round-to-power-of-2 scale format | §2 |
| YaRN | RoPE frequency scaling for long context | §3 |
