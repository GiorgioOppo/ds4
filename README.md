# DeepSeek-V4-Pro-MacOS

Swift + Metal port of [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
inference for Apple Silicon. Targets the same architecture as the upstream
Python implementation in `Reference/inference/`.

> **Status: scaffold aligned to the reference implementation.** Module shapes,
> Metal kernel signatures, and config fields all mirror the Python reference
> in `Reference/inference/{model.py, kernel.py}`. The five
> compute-heavy custom kernels (FP8 GEMM, FP4 GEMM, sparse attention, HC
> Sinkhorn, FP8/FP4 activation quant) are stubbed and trap on invocation —
> see "Why parts are intentionally unimplemented" below.

## Reality check

DeepSeek-V4-Pro is a 1.6T-parameter MoE (49B activated). Even at FP4 the
expert weights alone are ~800 GB. **No Mac can hold V4-Pro in unified
memory.** Realistic on-device target is **DeepSeek-V4-Flash** (284B / 13B
activated): at FP4 + FP8 mixed, ≈ 142 GB → fits a Mac Studio with 192 GB+
unified memory. Same code, different `config.json` and weights.

## Architecture map (Python → Swift)

The reference is in `Reference/inference/model.py` (827 lines).
Each module is mirrored 1:1 in Swift:

| Python (model.py) | Swift |
|---|---|
| `ModelArgs` | `Sources/DeepSeekKit/Config.swift` |
| `ParallelEmbedding` | `Sources/DeepSeekKit/Model.swift` |
| `Linear` (BF16/FP8/FP4 dispatch) | `Sources/DeepSeekKit/Layers/Linear.swift` |
| `RMSNorm` | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| `precompute_freqs_cis` (YaRN) | `Sources/DeepSeekKit/YaRN.swift` |
| `apply_rotary_emb` | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Kernels/rope.metal` |
| `Compressor` | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| `Indexer` | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| `Attention` (MLA + sliding window + sparse) | `Sources/DeepSeekKit/Layers/Attention.swift` |
| `Gate`, `Expert`, `MoE` | `Sources/DeepSeekKit/Layers/MoE.swift` |
| `Block` (with HC pre/post) | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| `HyperConnections.hc_pre / hc_post` | `Sources/DeepSeekKit/Layers/HyperConnections.swift` |
| `ParallelHead` | `Sources/DeepSeekKit/Model.swift` |
| `MTPBlock` | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| `Transformer` | `Sources/DeepSeekKit/Model.swift` |

The five tilelang kernels in `kernel.py` map to:

| Python (kernel.py) | Metal stub |
|---|---|
| `act_quant_kernel` (FP8), `fp4_quant_kernel` | `Kernels/act_quant.metal` |
| `fp8_gemm_kernel` | `Kernels/fp8_gemm.metal` |
| `fp4_gemm_kernel` | `Kernels/fp4_gemm.metal` |
| _(new — INT8 W8A16 weight-only)_ | `Kernels/int8_gemm.metal` |
| `sparse_attn_kernel` | `Kernels/sparse_attn.metal` |
| `hc_split_sinkhorn_kernel` | `Kernels/hc_sinkhorn.metal` |
| `rotate_activation` (Hadamard) | `Kernels/hadamard.metal` |

## Per-layer attention modes

The reference uses `compress_ratios = (0, 0, 4, 128, 4, 128, 4, 0)` (or
similar — confirm from the production `config.json`). Each ratio selects a
different forward path:

| compress_ratio | Mode | Modules used |
|---|---|---|
| 0 | pure sliding-window attention | MLA + window topk |
| 4 | sliding window + indexed sparse compression | MLA + Compressor (overlap) + Indexer + sparse_attn |
| 128 | sliding window + heavy compression | MLA + Compressor (no overlap) + sparse_attn |

## Build

```bash
swift build -c release
```

Requires macOS 14+, Xcode 15+, an Apple Silicon Mac. Today this builds, but
the CLI exits with a fatal error from any of the unimplemented stubs.

## Convert weights

```bash
swift run -c release converter \
    --hf-ckpt-path <upstream-hf-checkpoint> \
    --save-path <converted-output-dir> \
    --n-experts <N> \
    --target-dtype <bf16|f16|int8|keep>
```

`--target-dtype` controls how non-native dtypes in the upstream checkpoint
get rewritten:

| value | Linear weights | other tensors | disk footprint |
|---|---|---|---|
| `bf16` (default) | FP8/FP4 fused to BF16 | BF16 | ~3-4× input |
| `f16` | FP8/FP4 fused to F16 | F16 | ~3-4× input |
| `int8` | INT8 W8A16 (per-row × per-128 F16 scales) | BF16 | ~½ × BF16 |
| `keep` | preserved (FP4/FP8/etc.) | preserved | ≈ input |

INT8 quantization is symmetric round-to-nearest, range `[-127, 127]`, with
F16 group scales. Only the leaves consumed via the `Linear` module are
quantized (whitelist in `Int8Quant.shouldQuantizeToInt8`); embeddings,
LM head, norms, attention sinks, hyper-connection scalars, biases and
gates stay BF16. The Metal kernel `gemm_int8_w8a16_to_f32` accepts F32 or
BF16 activations and produces F32 output — no activation quantization.

## Run

```bash
.build/release/deepseek /path/to/DeepSeek-V4-Pro "Hello,"
```

## Roadmap (in dependency order)

The following work stands between this scaffold and end-to-end token
generation. Items are listed in the order they should be tackled.

### 1. Core math kernels (no dependency on weights)

- **`hadamard.metal`** — FWHT for power-of-2 dims (128, 512). Easiest target,
  good warm-up for the Metal toolchain. Validate against `scipy.linalg.hadamard`.
- **`hc_sinkhorn.metal`** — fixed-size hc=4, ~20 iters. Self-contained.
  Validate against the Python reference on random input.
- **`act_quant.metal`** — FP8/FP4 block quant. Needs `fast_log2_ceil` /
  `fast_pow2` bit hacks (port directly from `kernel.py:22–37`).

### 2. GEMM kernels (depend on quantized formats)

- **Dense BF16 GEMM** — needed by Compressor's wkv/wgate (FP32 in checkpoint),
  the gate weight matrix, and any non-quantized linear. Use simdgroup_matrix.
- **`fp8_gemm.metal`** — FP8 weight × FP8 act → BF16 out. Tile sizes per
  reference: 32×128×128. Cast FP8→FP16 on load (Metal has no native FP8).
- **`fp4_gemm.metal`** — FP8 act × FP4 weight, used only by experts.
  Per-32 weight scale, per-128 act scale. Unpack nibbles on load.

### 3. Attention kernel

- **`sparse_attn.metal`** — most complex. FlashAttention-style online softmax
  + KV gather by topk_idxs + per-head learnable sink. Validate numerically
  against a small forward pass run with the Python reference.

### 4. Module forwards (depend on §1–§3)

In rough dependency order:

- `Linear.callAsFunction` — wires act_quant → fp8/fp4_gemm
- `RoPE.apply` — already wired, validate end-to-end
- `Compressor.callAsFunction` — gated pooling + RoPE + fp4 quant
- `Indexer.callAsFunction` — top-k learned selection
- `MLA.callAsFunction` — full attention with sliding window + sparse
- `HyperConnections.pre` / `.post` — uses hc_split_sinkhorn
- `MoEFFN.callAsFunction` — top-k routing, expert dispatch.  
  Decode (M=1) first; prefill needs token-permutation scatter/gather.
- `Block.callAsFunction` — composes attn/ffn with HC wrapping
- `Transformer.forward` — embed → HC expand → blocks → head

### 5. Loader and tokenizer

- **`assembleModel`** in `Sources/deepseek/main.swift` — reads
  `model.safetensors.index.json`, walks every layer's weight names and builds
  the `Linear` / `RMSNorm` instances. Run `Reference/inference/convert.py`
  upstream to produce the single-rank `model0-mp1.safetensors` shard layout
  this loader expects.
- **`TokenizerLoader`** — port HuggingFace `tokenizer.json` parser
  (byte-level BPE + GPT-2 pre-tokenizer regex + DeepSeek special tokens
  `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`). Pure Swift, no FFI.

### 6. Chat encoding

`Reference/encoding/encoding_dsv4.py` (744 lines) builds the
chat string from OpenAI-style messages with `<think>` / `</think>` markers
for the three reasoning effort modes. Port it once the rest works; the CLI
in this scaffold takes a pre-formatted prompt string.

### 7. Numerical validation

For each module forward, dump activations from a Python reference run on a
fixed prompt and assert max-abs error below tolerance. Without this,
sparse_attn and hc_sinkhorn will silently produce wrong tokens.

### 8. Performance

After numerical parity:
- simdgroup matrix instructions for all GEMMs
- threadgroup memory swizzle for FP4 unpack
- persistent kernel for MoE dispatch
- pre-allocated KV cache pool, no per-step allocations
- pipeline state caching keyed by (kernel name, function-constant set)

## Why parts are intentionally unimplemented

Three reasons informed the decision to stub the five kernels in §1–§3:

1. **They need numerical validation that requires running the Python reference.**
   This sandbox can run Swift but not CUDA; I cannot generate ground-truth
   activations to validate against, so a guess that "compiles and looks right"
   could silently produce wrong tokens for a long time before being caught.
2. **They are non-trivial.** The reference `sparse_attn_kernel` is 80 lines
   of tilelang; porting to Metal Shading Language with simdgroup matrix
   instructions is ~300 lines plus careful tiling math.
3. **They build on shared machinery.** Most kernels need Hadamard FWHT,
   FP8/FP4 dequant LUTs, and act_quant; doing those first as standalone,
   testable units avoids debugging multiple unknowns at once.

Each stub names exactly what's missing and which line of the reference to
port. Module forwards are equally explicit about which Python line they
mirror (e.g. `model.py:484` for MLA forward).

## Layout

```
Package.swift
Sources/
  DeepSeekKit/                  Library
    Config.swift                ModelArgs (matches model.py:34)
    Device.swift                MTLDevice + library loader
    Tensor.swift                MTLBuffer-backed n-d tensor (f32/f16/bf16/fp8/fp4/e8m0)
    Quantization.swift          FP8-E4M3 / FP4-E2M1 / E8M0 layouts and dequant
    SafeTensors.swift           safetensors reader
    Tokenizer.swift             BPE protocol (loader unimplemented)
    KVCache.swift               sliding-window + compressed KV per layer
    Sampling.swift              argmax + temperature
    YaRN.swift                  precompute_freqs_cis (matches model.py:199)
    Generation.swift            generate loop (stub)
    Model.swift                 Embedding, Head, Transformer (stubs)
    Layers/
      Linear.swift              BF16/FP8/FP4 dispatch (stubs)
      RMSNorm.swift             working
      RoPE.swift                wired (validate when GEMM available)
      Elementwise.swift         silu_mul, axpy, scale, add (working)
      Attention.swift           MLA forward (stub)
      Compressor.swift          gated pooling (stub)
      Indexer.swift             top-k learned (stub)
      HyperConnections.swift    HC pre/post (stub)
      MoE.swift                 Gate, Expert, MoEFFN (stubs)
      DecoderLayer.swift        Block with HC wrapping (stub)
      MTPBlock.swift            multi-token prediction (stub)
    Kernels/
      common.metal              bf16 helpers
      rmsnorm.metal             working
      rope.metal                working (rope_apply_f32, last rope_head_dim only)
      softmax.metal             working
      elementwise.metal         silu_mul, axpy, scale, add
      sampling.metal            argmax, apply_temperature
      moe.metal                 gate scoring + top-k
      act_quant.metal           FP8/FP4 block-wise activation quant (working)
      hadamard.metal            FWHT for power-of-2 dims (working)
      hc_sinkhorn.metal         HC pre/post/comb splitter with Sinkhorn (working)
      fp8_gemm.metal            STUB — port fp8_gemm_kernel
      fp4_gemm.metal            STUB — port fp4_gemm_kernel
      int8_gemm.metal           INT8 W8A16 GEMM (per-row × per-128 F16 scales)
      sparse_attn.metal         STUB — port sparse_attn_kernel
  deepseek/
    main.swift                  CLI

Tests/
  DeepSeekKitTests/             XCTest target (Metal vs pure-Swift reference)
    HadamardTests.swift
    HCSinkhornTests.swift
    ActQuantTests.swift

Reference/                      Upstream Python source-of-truth (read-only)
  inference/                    model.py, kernel.py, generate.py, convert.py
  encoding/                     encoding_dsv4.py + tests
  generation_config.json
  tokenizer_config.json
  UPSTREAM_README.md            HF model card
  LICENSE                       upstream MIT
```

## License

MIT, mirroring the upstream model license.
