# Metal kernels reference

Per-kernel index of `Sources/DeepSeekKit/Kernels/*.metal`. Each entry
gives the input/output buffer layout, the dispatch shape, and any
function constants. The Swift wrapper in `Sources/DeepSeekKit/Layers/`
is the only thing that should bind buffers — never call these
directly from CLI code.

For dispatch convention details and how the wrappers create
`MTLComputePipelineState`, see `Device.swift` and any wrapper as an
example (e.g. `Layers/Hadamard.swift`).

## Conventions

- Buffer 0 is the primary input. Subsequent indices follow the
  declaration order in the kernel.
- Function-constant indices are GLOBAL across the library (MSL spec).
  Reserved indices:
  ```
  0   BLOCK_SIZE_FP8    (act_quant.metal)
  1   BLOCK_SIZE_FP4    (act_quant.metal)
  2   SCORE             (moe.metal)
  3   ROUTE_SCALE       (moe.metal)
  4   HC                (hc_sinkhorn.metal)
  5   SINKHORN_ITERS    (hc_sinkhorn.metal)
  6   HC_EPS            (hc_sinkhorn.metal)
  ```
  Add new function constants at index 7+.
- `[[threadgroup_position_in_grid]]`, `[[thread_position_in_threadgroup]]`,
  and `[[threads_per_threadgroup]]` in the same kernel must share
  dimensionality (all `uint`, all `uint2`, or all `uint3`).

## File index

### `rmsnorm.metal` — `rmsnorm_f32`
`y[r] = x[r] * rsqrt(mean(x[r]²) + eps) * weight[d]`. One threadgroup
per row, warp-reduce sum-sq.

### `rope.metal` — `rope_apply_f32`
Applies rotary embeddings to the trailing `rope_head_dim` of each head,
reading precomputed cos/sin from a `freqs` tensor. Buffer 3 packs
`(startPos, inverse)` as a `uint2`. Inverse mode negates the sine
component.

### `softmax.metal` — `softmax_f32`
Numerically stable softmax over the last dim. One threadgroup per row.

### `softmax_axis.metal` — `softmax_axis_f32`
Softmax along an arbitrary axis. Tensor is viewed as
`[outer, axis, inner]`. Dispatch: `(outer, inner)` threadgroups, each
with up to 256 cooperating threads.

### `sampling.metal` — `argmax_f32`, `apply_temperature`
Argmax over the last dim (single uint32 output) and in-place
temperature scaling. Both used by `Sampler`.

### `elementwise.metal` — `silu_mul_f32`, `axpy_f32`, `scale_f32`, `add_inplace_f32`
Standard pointwise ops. SwiGLU uses `silu_mul`; HC and MoE both use
the others.

### `common.metal`
BF16 ↔ float helpers (`bf16_to_float`, `float_to_bf16`). Sourced by
multiple kernels.

### `gemm_bf16.metal`
Three naive tiled GEMMs (16×16 threadgroups, scalar inner loop):
- `gemm_bf16_to_f32`  — bf16 × bf16 → f32
- `gemm_f32_bf16_to_f32` — f32 × bf16 → f32 (used when input is
  activation-side f32)
- `gemm_f32_to_f32` — f32 × f32 → f32

Naive, correctness-first. simdgroup matrix promotion is a future perf
pass.

### `fp8_gemm.metal` — `gemm_fp8_to_f32`
FP8-E4M3 weight × FP8-E4M3 activation → f32. Per-128 block scale
applied via accumulator. Includes the `deq_e4m3` inline (duplicated
from `act_quant.metal` because Metal compilation units don't share
inlines).

### `fp4_gemm.metal` — `gemm_fp8_fp4_to_f32`
FP8 activation × FP4-E2M1 weight (packed two-nibbles-per-byte) → f32.
Per-128 activation scale + per-32 weight scale. Used only by MoE
experts when the converter keeps FP4.

### `act_quant.metal` — `act_quant_fp8`, `act_quant_fp4`
Block-wise activation quantization. FP8 uses block size 128, FP4 uses
32. Two output modes: `inplace=true` round-trips (writes back BF16
after quant+dequant for QAT noise), `inplace=false` writes the raw
fp8 bytes / packed fp4 nibbles.

Function constants: `BLOCK_SIZE_FP8` (0), `BLOCK_SIZE_FP4` (1).
Specialised at pipeline creation time by `Layers/ActQuant.swift`.

### `hadamard.metal` — `hadamard_f32`
Walsh-Hadamard transform in-place. One threadgroup per row,
threadgroup memory tile of `dim` floats, log₂(dim) butterfly passes
with barriers between strides. dim must be a power of 2.

### `hc_sinkhorn.metal` — `hc_split_sinkhorn_f32`
Sinkhorn-normalized comb matrix for Hyper-Connections. One threadgroup
per token, `hc*hc` threads cooperating on the tile (hc=4 so 16 cells).
Iterates `SINKHORN_ITERS` times.

Function constants: `HC` (4), `SINKHORN_ITERS` (5), `HC_EPS` (6).

### `hc_kernels.metal`
Four small composition kernels used by `Layers/HyperConnections`:
- `rsqrt_mean_square_f32` — `out[n] = rsqrt(mean(x[n]²) + eps)`
- `broadcast_row_mul_f32` — `y[r, c] *= scale[r]` (in place)
- `hc_collapse_f32` — `y[n, d] = Σ_h pre[n, h] · x[n, h, d]`
- `hc_post_compose_f32` — `y[n, j, d] = post[n, j] · x[n, d]
                          + Σ_k comb[n, k, j] · residual[n, k, d]`

### `moe.metal` — `moe_gate`
MoE top-K gating with score function selected by function constant
`SCORE`: 0 = softmax, 1 = sigmoid, 2 = sqrtsoftplus. Multiplies by
`ROUTE_SCALE` at the end. One thread per token.

Function constants: `SCORE` (2), `ROUTE_SCALE` (3). Specialised by
`Layers/MoE.swift`.

### `moe_dispatch.metal` — `moe_gather`, `moe_scatter`
- `moe_gather` — `gathered[t, d] = x[assignTok[t], d]`. Pads to 0 when
  `assignTok[t] == -1`.
- `moe_scatter` — `y[n, d] += Σ_{t ∈ slots[n]} weights[t] · outs[t, d]`.

Host builds the assignment tables in `MoEDispatch.prepare(...)`.

### `compressor_kernels.metal`
Three kernels for `Layers/Compressor`:
- `broadcast_add_4d_2d_f32` — `y[b, ns, r, c] += w[r, c]` for adding
  `ape` to score.
- `weighted_sum_axis2_f32` — `y[b, ns, c] = Σ_r kv[b, ns, r, c]
                              * score[b, ns, r, c]` (post-softmax pool).
- `compressor_overlap_concat_f32` — gathers a `[B, 2R, D]` view from
  the `[B, 2R, 2D]` state buffer, using the first-half-low /
  second-half-high slice trick.
- `compressor_state_shift_copy_f32` — copies `state[:, R:, :]` into a
  fresh `tmp`; the host then blit-copies `tmp` back to `state[:, :R, :]`
  to avoid aliased copies during state shift after an overlap emit.

### `indexer_kernels.metal`
Two kernels for `Layers/Indexer`:
- `indexer_score_reduce_f32` — `y[b, s, t] = Σ_h max(0, score[b, s, h, t])
                                · weights[b, s, h]`, with optional prefill
  causal mask.
- `indexer_topk_postprocess_i32` — in-place mask invalid + offset add on
  topk indices.

### `sparse_attn.metal` — `sparse_attn_f32`
FlashAttention-style sparse multi-head attention. One thread per
`(b, m, h)`. Loops over the `K` topk indices, gathers KV rows, runs
online softmax with running max + sum, accumulates `o`. Sink logit
folded into the denominator after the loop.

### `einsum.metal`
Two specialized contractions:
- `einsum_bshd_btd_to_bsht_f32` — Indexer score `(q, kv) → score`
- `einsum_bsgd_grd_to_bsgr_f32` — MLA grouped output `(o, woA) → oR`

### `overlap_transform.metal` — `overlap_transform_f32`
Compressor's overlap shuffle: `[B, S, R, 2D] → [B, S, 2R, D]`, with
pad value coming from a buffer (used to inject `-inf` for the score
side).

### `embedding.metal` — `embed_lookup_f32`, `hc_expand_f32`
- Embedding table lookup: one thread per `(n, d)` cell.
- HC expand: tile a `[N, D]` tensor into `[N, HC, D]` by copying.

### `topk_f32.metal` — `topk_f32`
Top-K values + Int32 indices along the last axis. In-register heap
(`MAX_K = 32`). Output is descending-sorted. Used by Indexer's
post-score topk and reusable for `topK` sampling.

## How to add a new kernel

1. Write the `.metal` in `Sources/DeepSeekKit/Kernels/foo.metal`.
   Follow the dispatch / function-constant conventions above.
2. Add a Swift wrapper in `Sources/DeepSeekKit/Layers/Foo.swift` that:
   - Constructs the `MTLComputePipelineState` once (lazy or static).
   - Exposes a public method that builds the encoder, binds buffers,
     dispatches, ends encoding. Caller manages `commit/waitUntilCompleted`.
   - Includes a `referenceCPU(...)` pure-Swift implementation when
     practical.
3. Add a test in `Tests/DeepSeekKitTests/FooTests.swift` comparing
   Metal output to the reference on a small randomized input. Build
   inputs with `Tensor.from(bytes:shape:dtype:)`, read back with
   `Tensor.toFloatArray()`.
4. The `MetalLibPlugin` (declared in `Package.swift`) picks up
   `.metal` files automatically on the next `swift build`.
