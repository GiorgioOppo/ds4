# Roadmap & known limitations

What's implemented, what's stubbed, and what's deferred. When opening
a file expecting a specific feature, check here first.

## ✅ Tier 1 — Multi-token chat (done)

The CLI generates token-by-token end-to-end. All the blockers from the
"single token per invocation" prototype are removed.

- `MLA.forward` — prefill (any seqlen, with cutoff/wrap when > window)
  and decode (single-token ring-buffer write).
- `Compressor.forward` — prefill (with and without overlap), decode
  (state machine that emits one compressed token every `ratio` steps,
  including overlap with state-shift).
- `Transformer.forward` — full layer chain, embed → HC expand → blocks
  → head.
- Multi-token CLI loop with streaming output (raw mode) and `<think>`
  buffering (chat mode).

## ✅ Tier 2 — UX completeness (done)

- `Sampler.sample(_:history:options:)` — temperature, repetition
  penalty, top-K, top-P, Gumbel-max multinomial. Done host-side.
- `EncodingDSV4` — port of the practical chat-with-tools surface:
  BOS/role markers, EOS, `<think>...</think>` reasoning blocks,
  REASONING_EFFORT_MAX system prompt, tool_calls DSML emit + parse.
- `MTPBlock.callAsFunction` — forward pass implemented; speculative
  decoding integration in the CLI is the next step.

## ⏳ Tier 3 — Parity with Python reference (partial)

Item by item:

### Done

- `MoEHashRoutingTests` — verifies hash-routing layers route via the
  `tid2eid` lookup, exercising the `model.py:577` branch that the
  score-based tests don't cover.

### Deferred — structural prerequisite needed

- **act_quant noise on non-rope KV dims** (`MLA`, `Compressor`).
  The reference applies `act_quant(kv[..., :-rope_dim], ...)` to
  inject QAT noise. Our port skips it because the `Tensor` type has
  no strided view — `kv[..., :-rope_dim]` along the last axis is not
  a contiguous slice. Adding strided views to `Tensor` is a
  structural refactor (carrying stride per dim, updating every kernel
  that consumes a tensor) and was scoped out.
  Impact: forward results differ from the reference by < 1 % typical
  (QAT noise is small).

### Deferred — not on critical path

- **`cast_e2m1fn_to_e4m3fn` in converter** (`--expert-dtype fp8`
  lossless re-encode). Unused with the default `--target-dtype bf16`
  path since FP4 experts are fully dequantized to BF16. Re-enable
  this only if you want FP8 experts on disk to save space vs BF16
  while keeping inference fast.

### Deferred — needs external environment

- **End-to-end numerical validation vs Python reference**. Requires
  PyTorch + CUDA to dump activations from `Reference/inference/generate.py`
  on a toy config, then compare with the Swift forward. Plan:
  1. `Reference/inference/dump_activations.py` (write) — runs the
     Python forward on a fixed prompt with a tiny config (n_layers=2,
     dim=64, etc.) and dumps every layer's activations to JSON.
  2. `Tests/DeepSeekKitTests/EndToEndForwardTests.swift` (write) —
     loads the JSON, runs the Swift forward on the same toy config,
     asserts relative error < 1e-2 per layer.
  Useful for catching subtle ordering or sign-flip bugs that
  per-kernel tests miss.

## Performance — correctness-first, not yet optimized

See [PERFORMANCE.md](PERFORMANCE.md) for the full bottleneck
breakdown, hardware sizing estimates, profiling instructions, and
per-optimization spec sheets.

Headline opportunities, none pursued yet:

- simdgroup_matrix BF16 GEMM → ~5-10× on every Linear
- FlashAttention tiling for sparse_attn → ~3-5× on attention
- Persistent MoE dispatch kernel → ~2× per layer
- Pipeline state caching → ~10-50 ms saved per inference call
- KV cache pool → matters for multi-session serving

## Known limitations

### Currently the CLI is single-batch
`Sources/deepseek/main.swift` takes one prompt and produces one
completion. The model code (`Transformer.forward`) is shaped for
`[B, S]` input, but the CLI never exercises B > 1. Batched serving
is a CLI-level rework, not a model-code change.

### MTPBlock not used at inference
`MTPBlock.callAsFunction` is implemented but the CLI doesn't invoke
it for speculative decoding. To wire it in:
1. After the standard block stack, run each `MTPBlock` to get
   speculative logits.
2. Sample N candidate tokens.
3. Verify by running them through the next forward step. Accept
   matching prefix, retry from the first mismatch.
This is a CLI-level change and is left for a future iteration.

### Encoding stubs
- Task tokens (`<｜action｜>`, `<｜query｜>`, etc., from `DS_TASK_SP_TOKENS`)
  not emitted. Caller can prepend manually.
- `response_format` schema injection (encoding_dsv4.py:49) not ported.
  Same — prepend in the system message.
- `latest_reminder` token (encoding_dsv4.py:25) not emitted. Same.
- Three thinking effort modes: `.chat` and `.max` produce different
  prompts; `.high` currently behaves like `.chat`. The Python
  reference's `.high` mode adds a less-extreme reasoning prompt that
  hasn't been ported.

### Single-rank only
The converter and loader assume `model_parallel == 1`. The reference
supports multi-rank via `model{i}-mp{N}.safetensors` sharding. Not on
the critical path for Mac inference but blocks distributed use.

### `bf16` ParallelEmbedding
`ParallelEmbedding.init` precondition: `weight.dtype == .f32`. After
the converter's `--target-dtype bf16` pass, `embed.weight` lands as
BF16 and `Transformer.load` will skip it (falls back to random init).
Workaround: relax the precondition and add a BF16 lookup branch (~20
LOC), or run the converter with `--target-dtype keep`.

### `wo_a` always fused
The converter fuses `wo_a.weight + scale → BF16` regardless of
`--target-dtype`. This is necessary because `MLA.forward` uses
`Einsum.bsgdGrd` (which expects FP32-compatible) instead of `Linear`
for the grouped output projection. Leaving wo_a in FP8 would require
either an FP8-aware einsum or restoring the Linear path.

## How to extend

When adding new features, follow the conventions:

1. **New kernel**: see [`KERNELS.md`](KERNELS.md) "How to add a new
   kernel" section. Always pair with a Swift wrapper + CPU reference
   + XCTest.
2. **New layer composition**: put it in `Sources/DeepSeekKit/Layers/`.
   Document the corresponding Python source line range in the file
   header docstring.
3. **New CLI flag**: in `Sources/deepseek/main.swift` (inference) or
   `Sources/converter/main.swift` (converter). Update `USAGE.md`.
4. **New weight name convention**: update the canonical map in
   `Assembly.swift`'s `Transformer.load`. Add a fallback name list
   via `loader.tryLoad([...])` so old and new naming both work.
