# DeepSeek-V4-Pro-MacOS

Native Swift + Metal port of [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
inference for Apple Silicon.

> **Status: scaffold.** The project structure, core data types, standard Metal
> kernels (RMSNorm, matmul, RoPE, softmax, SwiGLU, sampling) and the
> generation loop are in place. The DeepSeek-V4-specific pieces — Compressed
> Sparse Attention (CSA), Heavily Compressed Attention (HCA), Manifold-
> Constrained Hyper-Connections (mHC), and the safetensors weight wiring — are
> intentionally not implemented yet. They are stubbed with explicit fatal
> errors and per-file design notes pointing at the missing references. **The
> CLI does not produce text yet.**

## Reality check

DeepSeek-V4-Pro is a 1.6T-parameter MoE (49B activated). Even at 4-bit
quantization the weights are ~800 GB. **No Mac can hold that in unified
memory.** Running V4-Pro locally is not a goal this codebase can meet on
current hardware. Realistic targets for an on-device runtime are:

- **DeepSeek-V4-Flash** (284B / 13B activated) at 4-bit ≈ 142 GB → fits a
  Mac Studio with 192 GB+ unified memory.
- A future distilled variant.

The architecture is shared, so the same code works for both — only the
weight files and `config.json` change.

## Layout

```
Package.swift
Sources/
  DeepSeekKit/             Library
    Config.swift           Mirrors config.json
    Device.swift           MTLDevice + library loader
    Tensor.swift           MTLBuffer-backed n-d tensor
    Quantization.swift     int4 block layout (placeholder, confirm vs weights)
    SafeTensors.swift      .safetensors reader
    Tokenizer.swift        BPE interface (loader unimplemented)
    KVCache.swift
    Sampling.swift         argmax + temperature
    Generation.swift       generate loop
    Model.swift            DeepSeekV4 top-level
    Layers/
      Linear.swift           dense + q4 matvec
      RMSNorm.swift
      RoPE.swift
      Elementwise.swift      silu_mul, axpy, scale, add
      Attention.swift        hybrid CSA/HCA dispatcher (forward unimplemented)
      MoE.swift              top-k routing + per-expert SwiGLU
      DecoderLayer.swift
    Kernels/                 Metal source (built into the bundle)
      common.metal
      matmul.metal           f32 GEMM + int4 matvec
      rmsnorm.metal
      rope.metal
      softmax.metal
      elementwise.metal
      sampling.metal
      moe.metal              top-k gate
      attention_csa.metal    STUB — traps when invoked
      attention_hca.metal    STUB — traps when invoked
      mhc.metal              STUB — traps when invoked
  deepseek/
    main.swift             CLI: deepseek <model-dir> "<prompt>"
```

## Build

```bash
swift build -c release
```

Requires macOS 14+, Xcode 15+, and an Apple Silicon Mac for actual execution.

## Run

```bash
.build/release/deepseek /path/to/DeepSeek-V4-Pro "Hello,"
```

Today this exits with `Tokenizer is not implemented yet.` See Roadmap.

## Roadmap (in dependency order)

The work below is what stands between the current scaffold and a CLI that
actually generates tokens. Items are listed in the order they should be
tackled.

1. **Download the reference repo.**
   `git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro` (or
   `DeepSeek-V4-Flash` for an actually runnable target). Inspect:
   - `config.json` — confirm the field names assumed in `Config.swift` and
     read off `csa_compression_ratio`, `hca_compression_ratio`, MoE counts.
   - `modeling_deepseek_v4.py` (or whatever the inference reference is
     called) — this is the source of truth for CSA, HCA, mHC, and the weight
     naming convention used in the safetensors index.
   - `tokenizer.json` — tokenizer model type and special tokens.
   - The DeepSeek-V4 technical report PDF — equations for CSA/HCA/mHC.

2. **Tokenizer.** Parse `tokenizer.json` (HuggingFace `tokenizers` JSON
   format). Implement byte-level BPE with the merges array and byte-fallback.
   Pure Swift, no FFI.

3. **CSA kernel.** Port the Python reference. Required pieces:
   - the K/V compression operator (linear / strided / pooling)
   - the sparse attention pattern (block, window, top-k, sinks)
   - causal masking over compressed positions
   - flash-attention-style tiling for memory efficiency
   - numerical test against a Python reference on a small input

4. **HCA kernel.** Same approach. Likely shares plumbing with CSA.

5. **mHC residual.** Replace `Elementwise.addInPlace` calls in
   `DecoderLayer.swift` with the manifold-constrained mixing op once the
   reference is in.

6. **MoE prefill path.** Current `MoEFFN` only handles single-token decode.
   Prefill needs token-permutation scatter/gather + grouped GEMM across
   experts. This is its own substantial subproject.

7. **Quantization layout.** `Quantization.swift` assumes a generic block
   int4 layout. Verify against the actual quantized weights shipped (the
   dtype string in the safetensors header will tell you).

8. **Model assembly.** Implement `assembleModel` in `Sources/deepseek/main.swift`
   to read `model.safetensors.index.json`, walk every layer's weight names,
   build `Linear` / `RMSNorm` / `MoEFFN` instances and hand them to
   `DecoderLayer` and `DeepSeekV4`.

9. **Numerical validation.** For each layer, dump activations from a
   reference Python forward pass on a fixed prompt and assert max-abs error
   below tolerance. Without this CSA/HCA/mHC will silently produce wrong
   tokens.

10. **Performance.** The kernels in this scaffold are correct-first, not
    fast. After numerical parity:
    - flash-attention tiling for CSA/HCA
    - simdgroup matrix instructions for matmul
    - persistent kernel for MoE dispatch
    - pre-allocated KV cache pool, no per-step allocations

## Why parts are intentionally unimplemented

Three reasons informed the decision to stub rather than guess CSA/HCA/mHC
and tokenizer/assembly logic:

1. The HuggingFace repo is unreachable from this environment (host allowlist
   blocks `huggingface.co`), so the reference Python and config could not
   be inspected.
2. CSA, HCA, and mHC are new in V4 with no public reference implementation
   outside DeepSeek's repo. The architecture summary in the technical report
   underdetermines the exact operators (compression op shape, sparsity
   pattern, manifold constraint factorization).
3. Wrong inference math is worse than missing math: a guessed implementation
   would compile, run, and silently produce garbage, masking the work
   actually required.

Each stub names exactly what is missing and where to look.

## License

MIT, mirroring the upstream model license.
