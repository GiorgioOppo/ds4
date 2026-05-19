# The model

A complete reference for the DeepSeek-V4 transformer as implemented by
this repo. Read it when you want to understand what each component
does, how they fit together, what dtypes flow between them, and where
each part lives in `Sources/DeepSeekKit/`.

The companion docs cover related concerns:

- [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) — Swift ↔ Python line-by-line
  cross-walk.
- [`MODULES.md`](MODULES.md) — per-file index of `Sources/`.
- [`KERNELS.md`](KERNELS.md) — per-`.metal` kernel reference.
- [`DTYPES.md`](DTYPES.md) — FP8 / FP4 / E8M0 / BF16 bit layouts.
- [`MEMORY.md`](MEMORY.md) — mmap, KV cache lifecycle, working-set
  estimates.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — engine data flow + the
  desktop-app architecture around the engine.
- [`GLOSSARY.md`](GLOSSARY.md) — one-liner definitions for every acronym
  below.

> 🇮🇹 La versione italiana di questo documento è
> [`MODEL.it.md`](MODEL.it.md).

---

## 1. What the model is

DeepSeek-V4 is an autoregressive decoder-only transformer with a
**Mixture-of-Experts** feed-forward and a stack of non-standard
attention and residual choices. The Swift port mirrors the reference
PyTorch implementation in `Reference/inference/model.py` 1:1; that
file is the source of truth for any disagreement.

The signature ingredients, top-to-bottom:

| Component | Notes |
|---|---|
| **MLA** (Multi-head Latent Attention) | Low-rank Q via `wq_a → q_norm → wq_b`, shared single-head KV via `wkv → kv_norm`, grouped low-rank O via `wo_a → wo_b`. Per-head learnable `attn_sink` scalar. |
| **Sliding window + sparse attention** | Per-token KV is kept in a 128-entry ring buffer; optional per-layer compressed-KV tail. FlashAttention-style online softmax over a top-k subset. |
| **Compressor** | Gated softmax-pooling over `compress_ratio` consecutive tokens, emitting one compressed KV every `ratio` steps. Overlap mode at `ratio == 4`. |
| **Indexer** | Top-k learned selector that picks which compressed-KV positions the sparse attention should attend to. Used only on `ratio == 4` layers. |
| **Hyper-Connections (HC)** | The hidden state is held as `hc_mult = 4` parallel copies; each block uses Sinkhorn-normalised mixing instead of a plain residual. |
| **MoE FFN** | `sqrtsoftplus` top-2 routed experts + 1 shared expert. Optional hash routing for the first `n_hash_layers`. |
| **RoPE + YaRN** | RoPE applied only to the last `rope_head_dim = 64` of each head. YaRN frequency correction for long context. |
| **MTP** (Multi-Token Prediction) | A trailing speculative block that predicts the *next* token given the current hidden state + the new embedding, sharing the same `ParallelEmbedding` and `ParallelHead`. |

### Two release sizes

The released checkpoints come in two flavours. The runtime is identical
— what changes is `ModelConfig` and therefore the per-tensor shapes.

| | V4-Pro | V4-Flash |
|---|---|---|
| Total parameters | ≈ 1.6 T | 284 B |
| Activated per token | ≈ 50 B | 13 B |
| `n_layers` | larger | typically 7-ish |
| Disk @ FP8/FP4 native | ≈ 800 GB | ≈ 142 GB |
| Disk @ BF16 fused | ≈ 1.6 TB | ≈ 600 GB |
| Realistic on a 192 GB Mac? | **no** | yes (mmap) |
| Realistic on a 16 GB Mac? | no | yes (streaming) |

The realistic on-device target is **V4-Flash**. V4-Pro doesn't fit any
Mac's unified memory. See [`MEMORY.md`](MEMORY.md) for the streaming
loader.

---

## 2. The hyperparameters: `ModelConfig`

`ModelConfig` (`Sources/DeepSeekKit/Config.swift:5`) mirrors the
`ModelArgs` dataclass at `Reference/inference/model.py:34`. Field names
use Python `snake_case` via `CodingKeys`; the `init(fromDict:)` path
also accepts the HuggingFace `transformers` aliases (`hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `rms_norm_eps`,
`rope_scaling.factor`, …) so the same code reads both the repo's
post-converter `config.json` and the upstream HF release card.

### Shape parameters

| Field | Default (V4-Flash) | Meaning |
|---|---|---|
| `vocabSize` | 129 280 | Token vocabulary size. |
| `dim` | 4096 | Residual stream / hidden size. |
| `nLayers` | 7 | Number of main transformer blocks. |
| `nMtpLayers` | 1 | Trailing MTP blocks for speculative prediction. |
| `nHashLayers` | 0 | First N layers route via a precomputed hash table instead of a scored gate. |
| `nHeads` | 64 | Attention heads. |
| `headDim` | 512 | Per-head dimension (Q, K, V share this size). |
| `ropeHeadDim` | 64 | Tail of each head receiving RoPE. The leading `headDim - ropeHeadDim = 448` dims are *nope* (no positional encoding). |
| `maxBatchSize` | 4 | Used to size the KV cache, not the runtime batch. |
| `maxSeqLen` | 4096 | KV ring + compressed slice size. Long-context inference needs this raised. |

### Low-rank attention

| Field | Default | Meaning |
|---|---|---|
| `qLoraRank` | 1024 | Bottleneck size on the Q path (`wq_a: dim → qLoraRank`, `wq_b: qLoraRank → nHeads·headDim`). |
| `oLoraRank` | 1024 | Per-group bottleneck on the O path. |
| `oGroups` | 8 | Number of low-rank groups for the output projection. Each group projects `nHeads·headDim / oGroups` columns through its own `oLoraRank`-sized rank. |
| `windowSize` | 128 | Sliding-window attention size (the ring buffer's row count). |

### Per-layer compression policy

| Field | Default | Meaning |
|---|---|---|
| `compressRatios` | `[0, 0, 4, 128, 4, 128, 4, 0]` | One entry per layer (main + MTP, in this order). `0` = pure sliding window; `4` = window + Compressor (with overlap) + Indexer; `128` = window + Compressor (no overlap, no Indexer). |

The array length must equal `nLayers + nMtpLayers`. `ModelConfig.inferred(from: loader)`
patches mismatched configs by inferring `nLayers` from
`compressRatios.count - nMtpLayers`.

### MoE feed-forward

| Field | Default | Meaning |
|---|---|---|
| `nRoutedExperts` | 8 (V4-Flash production: 256) | Pool of experts the gate routes over. |
| `nActivatedExperts` | 2 | Top-K experts active per token. |
| `nSharedExperts` | 1 | Always-active "shared" expert added on top of the routed contribution. |
| `moeInterDim` | 4096 | SwiGLU FFN inner dim per expert. |
| `scoreFunc` | `"sqrtsoftplus"` | One of `softmax / sigmoid / sqrtsoftplus`. V4 uses `sqrtsoftplus`. |
| `routeScale` | 1.0 | Scalar applied to per-expert weights after normalisation. |
| `swigluLimit` | 0.0 | If non-zero, clamps the SwiGLU gate. Zero disables the clamp. |

### RoPE / YaRN

| Field | Default | Meaning |
|---|---|---|
| `ropeTheta` | 10 000 | Base θ for non-compressor RoPE (used on layers with `ratio == 0`). |
| `compressRopeTheta` | 40 000 | Base θ for compressor-touching layers (`ratio > 0`); higher base = slower frequencies → longer effective context. |
| `originalSeqLen` | 0 | YaRN context "training" length. `0` disables YaRN frequency correction. |
| `ropeFactor` | 40 | YaRN extrapolation factor. |
| `betaFast`, `betaSlow` | 32, 1 | YaRN ramp boundaries. |
| `mscale` | 1.0 | Optional V3-style softmax-scale correction. V4 leaves it at 1 (see Attention.softmaxScale below). |

### Indexer

| Field | Default | Meaning |
|---|---|---|
| `indexNHeads` | 64 | Indexer attention heads. |
| `indexHeadDim` | 128 | Indexer per-head dim (separate from the main `headDim`). |
| `indexTopk` | 512 | Top-K compressed positions selected per query. |

### Hyper-Connections

| Field | Default | Meaning |
|---|---|---|
| `hcMult` | 4 | Number of parallel copies of the hidden state. |
| `hcSinkhornIters` | 20 | Sinkhorn iteration count when normalising the comb mixing matrix. |
| `hcEps` | 1e-6 | Numerical-stability epsilon for the gated `sigmoid + eps` clamp. |

### Quant dtype + scale

| Field | Default | Meaning |
|---|---|---|
| `dtype` | `"fp8"` | Native checkpoint dtype for non-expert linears. `"fp8"` (E4M3) or `"bf16"`. |
| `expertDtype` | `nil` (= same as `dtype`) | Override for routed experts. V4-Flash uses `"fp4"` (E2M1, packed two-per-byte). |
| `scaleFmt` | `"ue8m0"` | Block-scale dtype companion to FP8/FP4 weights. |
| `scaleDtype` | `"fp8"` | Activation-quant scale dtype, used by `act_quant`. |

### Diagnostic helpers

- `ModelConfig.summary` — human-readable dump of every field, used by
  `--print-config`.
- `ModelConfig.inferred(from:)` — patches `n_layers`, `vocab_size`,
  `dim`, `q_lora_rank`, `n_heads`, `o_lora_rank`, `moe_inter_dim`, and
  `index_n_heads` from actual tensor shapes when the on-disk
  `config.json` is incomplete or stale.
- `ModelConfig.projectedKVCacheBytes` — coarse upper bound on the KV
  cache memory footprint at the chosen `(max_seq_len, max_batch_size)`.
  The loader refuses early if this would blow the budget.
- `ModelConfig.compressRatioLCM` — LCM of the non-zero compress ratios.
  Used by KV-rewind to enforce a window-aligned `pos` (see §10.3).
- `ModelConfig.nopeHeadDim` — derived as `headDim - ropeHeadDim`.

---

## 3. Forward pass at a glance

```
input_ids: [[Int]]  (outer = batch, inner = seqlen, all same length)
        │
        ▼
ParallelEmbedding.lookup     →  h: [B·S, dim] f32
        │
        ▼
hc_expand_f32 kernel         →  h: [B, S, hc_mult, dim]
        │
        ▼
for layer i in 0 ..< n_layers:                     ┐
    block_i(h, start_pos, input_ids)               │  Block:
        │                                          │   HC.pre  → attn_norm → MLA → HC.post
        ▼                                          │   HC.pre  → ffn_norm  → MoE → HC.post
   h: [B, S, hc_mult, dim]                         ┘
        │
        ▼
ParallelHead(h, hc_head_*, norm)
        │  collapse hc → norm → take last sequence position → lm_head matmul
        ▼
logits: [B, vocab_size] f32   →  Sampler.sample(...)
```

Each main block runs its own command buffer (`cmd.commit + waitUntilCompleted`)
so the streaming-pool loader can rotate one layer's shard at a time
and so per-layer numerical traces (under `--trace-norms`) have a clean
boundary.

The single-token decode path collapses `S = 1` and adds a ring-buffer
write into the sliding-window KV cache; the prefill path with `S > 1`
runs the whole sequence at once. Both paths share the same code in
`MLA.callAsFunction` and `Compressor.callAsFunction`.

For a per-component, per-line view of the decode path see
`Sources/DeepSeekKit/Model.swift:214` (`Transformer.forward`).

---

## 4. Components

### 4.1 Tokenizer and embedding

The tokenizer is a separate concern (it lives in
`Sources/DeepSeekKit/{BPETokenizer,SentencePieceTokenizer,WordPieceTokenizer}.swift`
and is documented in [`MODULES.md`](MODULES.md)). The engine only sees
the integer ids it emits.

The embedding table is a single `[vocab, dim]` matrix on F32 or BF16.
`ParallelEmbedding.lookup` (`Sources/DeepSeekKit/Model.swift:6`) reads
one row per id with the `embed_lookup_f32` or
`embed_lookup_bf16_to_f32` kernel and returns a `[N, dim]` F32 tensor,
where `N = batch · seqlen`.

The class is named `ParallelEmbedding` to keep the Python name; this
port is single-rank, so there is no actual sharding.

### 4.2 Hyper-Connections (HC)

Standard residual is `x = x + sublayer(norm(x))`. HC replaces this with
mixing over `hc_mult = 4` parallel copies of the hidden state, with
the mixing matrix doubly-stochastic via Sinkhorn iterations. The
result: each block has *two* HC passes — one wrapping attention, one
wrapping FFN — and the residual stream lives as `[B, S, hc_mult, dim]`
instead of `[B, S, dim]`.

`HyperConnections` (`Sources/DeepSeekKit/Layers/HyperConnections.swift:7`)
has two phases:

**`pre(x, hcFn, hcScale, hcBase)` →  `(y, post, comb)`**

1. `rsqrt(mean(x²) + normEps)` per row over the flattened `[N, hc·dim]`
   axis (kernel `rsqrt_mean_square_f32`).
2. `mixes = x_flat @ hcFnᵀ` (F32 linear, `hcFn: [(2+hc)·hc, hc·dim]`).
3. `mixes *= rsqrt` (broadcast over the row dim, `broadcast_row_mul_f32`).
4. Sinkhorn split (`HCSinkhorn.split`, kernel `hc_split_sinkhorn_f32`):
   reshapes the `mixes` into a `[N, (2+hc), hc]` tile, applies
   `sigmoid(... · hcScale + hcBase) + hcEps`, then iterates
   `hc_sinkhorn_iters` rounds of column/row normalisation to produce
   a doubly-stochastic `comb` matrix; the leading `(2)` slabs are split
   off into `pre` and `post`.
5. `y[n, d] = Σ_h pre[n, h] · x[n, h, d]` — kernel `hc_collapse_f32`.
   This is the "input the sublayer sees", a `[N, dim]` view of the `hc`
   copies after weighted collapse.

**`post(out, residual, post, comb)` → `[N, hc, dim]`**

`hc_post_compose_f32` kernel:

```
y[n, j, d] = post[n, j] · out[n, d]
           + Σ_k comb[n, k, j] · residual[n, k, d]
```

So the sublayer's `out` is broadcast back to the `hc` copies with
weighting `post`, and each copy receives a Sinkhorn-mixed contribution
from the pre-sublayer residual via `comb`.

The HC parameters loaded per-block are six tensors:
`hc_attn_fn / hc_attn_base / hc_attn_scale` (for the attention sublayer)
and `hc_ffn_fn / hc_ffn_base / hc_ffn_scale` (for the FFN sublayer).

The final `ParallelHead` uses a simpler **sigmoid-only collapse** (no
Sinkhorn): the gating `hcFn / hcBase / hcScale` are read from
`hc_head_fn / hc_head_base / hc_head_scale` and used directly to
collapse `[B, S, hc, dim]` → `[B, S, dim]` before the LM head matmul.

### 4.3 RMSNorm

`RMSNorm(weight, eps)` (`Sources/DeepSeekKit/Layers/RMSNorm.swift:4`):

```
y[r, d] = x[r, d] · rsqrt(mean(x[r]²) + eps) · weight[d]
```

The kernel comes in two flavours selected by the gain dtype:
`rmsnorm_f32` (F32 gain) and `rmsnorm_bf16w_f32` (BF16 gain, dispatched
when the loader returned a BF16 weight from a HF-native checkpoint).

`eps = norm_eps = 1e-6` everywhere it appears.

The model uses RMSNorm in seven places per block: `attn_norm`,
`q_norm` (on the Q low-rank intermediate), `kv_norm` (on the KV
projection), `ffn_norm`, the Compressor's `norm` (on the pooled
output), plus the final `norm` before the LM head. There is also one
hot rsqrt-by-row inside both the MLA Q path and the HC pre — those use
the `rsqrt_mean_square_f32` kernel directly (no gain multiplied),
producing only the inverse-magnitude factor that's then broadcast.

### 4.4 Multi-head Latent Attention (MLA)

`MLA` (`Sources/DeepSeekKit/Layers/Attention.swift:16`) is V4's
attention variant. The signature changes from textbook multi-head
attention:

- **Q is low-rank**: `wq_a: dim → q_lora_rank` → `q_norm` → `wq_b:
  q_lora_rank → n_heads·head_dim`. Plus a per-head rsqrt re-norm (one
  inverse-sqrt scaling, no learned gain), so each head's vector ends up
  with bounded magnitude.
- **KV is shared, single-head**: a single `wkv: dim → head_dim` matrix.
  All heads read the same K and V. This is the "latent" in MLA — KV
  capacity does not scale with `n_heads`.
- **O is grouped low-rank**: `nHeads · headDim` reshaped as
  `[nGroups, nHeads·headDim / nGroups]`, then projected to
  `[nGroups, oLoraRank]` via `wo_a`, then `nGroups · oLoraRank → dim`
  via `wo_b`. Implemented as an explicit einsum
  (`Einsum.bsgdGrd`) plus a final `wo_b` matmul.
- **Per-head attention sink**: a learnable scalar `attn_sink[h]` is
  folded into the softmax denominator (see §4.5).
- **No softmax mscale**: `softmax_scale = head_dim^(-0.5)`, plain
  inverse-sqrt. V3's `mscale * mscale` correction is *not* applied —
  trying to import it actually hurt V4 outputs.

Step-by-step (decode-style; prefill follows the same path but with
multiple sequence rows):

1. `xFlat = x.reshape([B·S, dim])`.
2. **Q path**:
   - `qrFlat = q_norm(wq_a(xFlat))`  → `[B·S, q_lora_rank]`.
   - `q = wq_b(qrFlat).reshape([B·S, nHeads, headDim])`.
   - `q *= rsqrt(mean(q²) + eps)` over the headDim axis (per-head
     re-norm).
   - **RoPE** on the trailing `ropeHeadDim` of each head (see §4.6).
3. **KV path**:
   - `kvFlat = kv_norm(wkv(xFlat))` → `[B·S, headDim]` (single-head).
   - RoPE on the trailing `ropeHeadDim`.
   - **FP8 QAT noise** on the leading `nopeHeadDim` (the non-RoPE
     dims): `ActQuant.partialInplaceQuant(..., blockSize=64)` mirrors
     `act_quant(kv[..., :-rd], 64, ..., True)` at `model.py:506`.
     Without this step the KV nope dims exceed FP8 range in deep
     layers, attention scores grow, and the residual stream amplifies
     uncontrollably (observed L2 climbing layer 0 → layer 42 from 75 to
     615 000 before the QAT noise was added back).
4. **Build top-k indices**: the sparse attention only reads `kWin +
   kComp` KV positions per query. `AttnIndicesGPU.window` fills the
   window slice (always present); when `compressRatio > 0` either the
   Indexer (for `ratio == 4`) or `AttnIndicesGPU.compressedDeterministic`
   (for `ratio == 128`) fills the compressed slice.
5. **KV cache write**:
   - Decode (`startPos > 0`, `seqlen == 1`): single-row ring-buffer
     write `kvCache[:B, startPos % windowSize] = kv[:, 0]`.
   - Prefill, `seqlen ≤ windowSize`: contiguous fill from row 0.
   - Prefill, `seqlen > windowSize`: keep only the last `windowSize`
     rows of the prompt, with cutoff/wrap so the ring ends at
     `(S - 1) % windowSize`.
6. **Compressor** runs for side effects: it updates its rolling
   `kvState` / `scoreState` and may emit a fresh compressed token into
   the trailing slice of the KV cache (`§4.7`). The Compressor's
   `kvCache` Tensor *aliases* the trailing slice of MLA's `kvCache`
   buffer (different `offset`, same `MTLBuffer`) so the write is
   visible to MLA without a copy.
7. **Sparse attention** — `SparseAttention.apply(q, kvFull, sink,
   topkIdxs, scale)` — see §4.5.
8. **Inverse RoPE** on the attention output `o`, so the rotated frame
   is undone before the output projection (which is trained on
   un-rotated outputs).
9. **Grouped output**:
   - `o.reshape([B, S, nGroups, nHeads·headDim / nGroups])`.
   - `oR = Einsum.bsgdGrd(o, woA: woA.weight.reshape([nGroups, oLoraRank,
     perGroupD]))` → `[B, S, nGroups, oLoraRank]`. When `woA` is FP8 on
     disk the einsum kernel dequantises inline with `woA.scale`; for
     INT-quantized or BF16-fused models the path is automatic.
   - `result = wo_b(oR.reshape([B·S, nGroups·oLoraRank])).reshape([B, S, dim])`.

`MLA.callAsFunction` takes the command buffer as `inout`: when the
indexer is active it needs to commit-and-wait at one point to read GPU
output to host, and on return it hands back a fresh command buffer.

### 4.5 Sliding-window sparse attention

`SparseAttention.apply` (`Sources/DeepSeekKit/Layers/SparseAttention.swift:13`)
runs FlashAttention-style online softmax. One thread per `(b, m, h)`:

- Iterates over the `K = kWin + kComp` top-k indices for this query.
- Gathers the corresponding KV row from `kvFull[:, idx, :]`.
- Computes `score = q · kv * scale` (with `scale = head_dim^(-0.5)`).
- Online-softmax accumulates `acc = Σᵢ e^(scoreᵢ - max) · kvᵢ`,
  rescaling on `max` updates.
- After the loop, folds the per-head `attn_sink[h]` into the
  denominator: `sumExp += exp(sink[h] - sMax)`.
- Writes `o[b, m, h, :] = acc / sumExp`.

The sink trick comes from the V4 paper: it gives the softmax a learned
"null position" that absorbs probability mass when no real token is a
good match. The model can effectively choose to attend to nothing.

The "top-k" mechanism is what makes this sparse:

- The window slice is just `topkIdxs[b, s, 0..kWin-1] = [windowStart
  + 0, ..., windowStart + kWin - 1]`. During decode the start is
  `startPos % windowSize` (ring wrap); during prefill it's slot 0.
  Indices that would point past the actual prompt are padded with
  `-1`, which the kernel skips.
- The compressed slice is either:
  - **From the Indexer** (`ratio == 4`): the Indexer scores every
    compressed-KV position against the current query and returns the
    top `indexTopk = 512`. Heavily learned.
  - **Deterministic** (`ratio == 128`): the kernel
    `attn_compressed_indices_i32` produces `[compOffset, compOffset+1,
    ...]` covering the available compressed tokens (capped by
    `endPos / ratio`).
- For layers with `ratio == 0` the compressed slice has length zero —
  pure sliding-window attention.

### 4.6 RoPE (with YaRN frequency correction)

`RoPE(ropeHeadDim, freqs)` (`Sources/DeepSeekKit/Layers/RoPE.swift:7`)
applies in-place rotation to the trailing `ropeHeadDim` columns of
each head of a `[tokens, heads, head_dim]` tensor. The leading `headDim
- ropeHeadDim` columns are untouched (the "no-position-encoding" /
nope split).

The freqs table is precomputed by `YaRN.precomputeFreqsCis`
(`Sources/DeepSeekKit/YaRN.swift:11`):

1. Base frequencies `f_i = 1 / base^(2i / dim)` for `i = 0 ..
   ropeHeadDim/2 - 1`.
2. **YaRN correction** when `originalSeqLen > 0`:
   - Compute the correction range `[lo, hi]` from `betaFast` /
     `betaSlow`.
   - For each `i`, ramp factor `s_i ∈ [0, 1]`, blend
     `f_i := f_i / factor · (1 - s_i) + f_i · s_i`.
   - This extrapolates the rotation frequencies so longer-than-training
     positions still produce un-collided rotations.
3. Bake out `[seqlen][rope_dim/2][2]` (cos, sin) pairs as F32.

The `freqs` Tensor is per-layer because each layer picks one of two
RoPE bases:

- `ratio > 0` layers use `compress_rope_theta = 40 000` and `useYarn =
  true`.
- `ratio == 0` layers use `rope_theta = 10 000` and `useYarn = false`.

The Indexer also uses the same RoPE instance (`Indexer.rope` is wired
to the parent MLA's `RoPE` on first forward).

`RoPE.apply(_, startPos, inverse)`:
- `inverse = false` rotates by angle `+freq[t]`.
- `inverse = true` rotates by `-freq[t]` (sin component negated). Used
  by MLA on the attention output to undo the rotation before `wo_a`.

### 4.7 Compressor

`Compressor` (`Sources/DeepSeekKit/Layers/Compressor.swift:15`) pools
`compressRatio` consecutive tokens into one compressed KV row. Two
modes:

- **`ratio == 128`**: no overlap. Each window of 128 tokens produces
  one compressed token.
- **`ratio == 4`**: overlap on. Each emit blends the previous and
  current 4-token window (`coff = 2`), giving smoother boundaries.

Internal parameters per layer (loaded under `layers.<i>.attn.compressor.*`):

- `ape: [ratio, coff·head_dim]` — additive positional encoding for the
  pooling weights.
- `wkv: dim → coff·head_dim` — Linear that produces the per-token KV
  contribution.
- `wgate: dim → coff·head_dim` — Linear that produces the pooling
  score.
- `norm: RMSNorm(head_dim)` — post-pool normalisation.

The Compressor's `kvCache` is *not* its own buffer — it is a
zero-copy view onto the trailing slice of the parent MLA's `kvCache`,
wired in on first forward:

```swift
comp.kvCache = Tensor(shape: [B, compRows, headDim], dtype: .f32,
                      buffer: kvCache.buffer,
                      offset: kvCache.offset + win * bytesPerRow)
```

(`Sources/DeepSeekKit/Layers/Attention.swift:195`.) Symmetrically the
Indexer's own compressor aliases the Indexer's `kvCache`.

#### Prefill (`startPos == 0`)

Walking the prompt all at once. The reference is `model.py:325`.

1. Project the whole prompt through `wkv` and `wgate` → `[B·S,
   coff·head_dim]` each.
2. **State stashing** so a later decode that crosses a compression
   boundary has the prompt tokens available:
   - For overlap with `cutoff >= ratio`: copy `kv[cutoff-ratio:cutoff]`
     and `score[cutoff-ratio:cutoff] + ape` into `kvState / scoreState
     [:, :ratio]`.
   - For any `remainder = S % ratio > 0`: copy the tail rows
     `kv[cutoff:]` and `score[cutoff:] + ape[:remainder]` into the
     appropriate state slots (`[:, ratio:ratio+remainder]` for overlap;
     `[:, 0:remainder]` for non-overlap).
3. If `numBlocks = S / ratio > 0`, reshape `kv[:cutoff]` and
   `score[:cutoff]` into `[B, numBlocks, ratio, coff·head_dim]`.
4. Broadcast-add `ape` into the score tensor.
5. **Overlap transform** if `ratio == 4`: shuffle into `[B, numBlocks,
   2·ratio, head_dim]` so each pooled block sees the previous-window
   halves too. Pad value is `-inf` for the score side, `0` for the KV
   side.
6. `softmax` along the ratio axis.
7. Weighted sum: `pooled[b, n, d] = Σ_r kv[b, n, r, d] · score[b, n,
   r, d]`. Yields `[B, numBlocks, head_dim]`.
8. **Post-process** (see below).
9. Blit `result` into `self.kvCache[:B, :numBlocks]`.

#### Decode (`startPos > 0`)

Single-token incremental. The reference is `model.py:343`.

1. Project the single new token through `wkv` and `wgate` → `[B,
   coff·head_dim]` rows.
2. Add `ape[startPos % ratio]` to the score row.
3. Write the new row into the rolling state:
   - Overlap: slot `ratio + startPos % ratio`.
   - Non-overlap: slot `startPos % ratio`.
4. If `(startPos + 1) % ratio != 0`: the window isn't full yet, return
   `nil`. The caller treats this as "no new compressed token this
   step".
5. Otherwise:
   - **Overlap path**: build `pooledKV` and `pooledScore` as
     `[B, 2·ratio, head_dim]` via `overlapConcat` — first-half-low
     (taking the leading `head_dim` of the first half of the state)
     concatenated with second-half-high (the trailing `head_dim` of
     the second half).
   - **Non-overlap path**: alias `kvState` / `scoreState` directly as
     `[B, ratio, head_dim]`.
6. `softmax` along the ratio axis, weighted sum, get the single
   `[B, 1, head_dim]` emitted token.
7. **State shift-down** (overlap only): copy `state[:, ratio:]` into
   `state[:, :ratio]` so the second-half slot becomes the first half
   for the next window.
8. Post-process.
9. Blit `result` into `self.kvCache[:B, startPos / ratio]`.

#### Post-process (shared)

```
result := norm(result)
RoPE.apply(result, startPos: <ratio-adjusted>, inverse: false)
if rotate:
    Hadamard.apply(result)           # in-place
    ActQuant(.fp4).quant(result, inplace: true)   # FP4 QAT noise (Indexer)
else:
    ActQuant.partialInplaceQuant(    # FP8 QAT noise on nope dims (MLA)
        result, colStart: 0, colEnd: nopeHeadDim,
        blockSize: actBlockSizeFP8KVNope=64)
```

`rotate = true` is the Indexer-owned Compressor path (FP4 with Hadamard
rotation); `rotate = false` is the MLA-owned path (FP8 nope QAT).

#### State rewind

`Compressor.rewindStateTo(pos:)` resets the rolling state to a clean
window boundary. Required for any KV-cache rewind across the model:

- Returns `false` if `pos % compressRatio != 0` — mid-window state
  can't be reconstructed without re-running the prompt.
- On success, zeroes `kvState` and sets `scoreState` to `-Float.infinity`
  (so unused slots contribute 0 mass through softmax). The main
  `kvCache` is left untouched.

This is one half of the cross-restart resume + delegation KV-snapshot
machinery — see §10.

### 4.8 Indexer

`Indexer` (`Sources/DeepSeekKit/Layers/Indexer.swift:10`) is a learned
top-k selector used only on `ratio == 4` layers. Where the Compressor
collapses tokens, the Indexer **picks which compressed positions** the
sparse attention attends to.

Parameters:

- `wqB: q_lora_rank → index_n_heads · index_head_dim` — projection from
  the MLA's low-rank Q intermediate to the indexer's own
  multi-head query.
- `weightsProj: dim → index_n_heads` — per-head scoring weights.
- `compressor: Compressor` — the indexer keeps its **own** Compressor
  (`rotate = true`, `head_dim = index_head_dim`), with its own state
  buffers and its own kvCache. The same prefill/decode logic as MLA's
  compressor but with FP4 quant + Hadamard at the post-process step.
- `kvCache: [maxBatch, maxSeqLen/ratio, index_head_dim]` — distinct
  from MLA's kvCache.

Forward (called from MLA with the shared `qr` low-rank Q intermediate):

1. `q = wq_b(qr) → [B, S, index_n_heads, index_head_dim]`.
2. **RoPE** on the rope tail of `q` (using the shared MLA RoPE instance).
3. **Hadamard rotation** on the head dim of `q` (per head, in-place).
4. **FP4 QAT noise** on `q` to mirror the training-time
   `fp4_act_quant` round-trip.
5. Run the internal Compressor with the layer's `x` → it stashes state
   and writes a new compressed token into `Indexer.kvCache` when a
   window closes.
6. `weights = weightsProj(x) * (softmaxScale * n_heads^(-0.5))`.
7. **Score**: `score = einsum("bshd,btd→bsht", q, kvCache[:T])` →
   `[B, S, index_n_heads, T]`, where `T = endPos / ratio`.
8. **Reduce**: `y[b, s, t] = Σ_h max(0, score[b, s, h, t]) · weights[b, s,
   h]`. Per-head `relu(score)` weighted by the per-head `weights`.
   With prefill causal mask when `startPos == 0`.
9. **Top-K**: `topkIdxs = TopK(y, k=min(index_topk, T))` → `[B, S, K]`.
10. **Post-process**: mask invalid slots (`-1`) and add the offset
    (`compOffset = isDecode ? windowSize : S`) so the returned indices
    are absolute positions inside the merged `[window | compressed]`
    KV table the sparse attention reads.

The Indexer's own `releaseCache()` / `restoreKVCacheBytes(...)` mirror
MLA's: ARC frees the buffer; restore writes the cache back from a
snapshot and re-wires the internal Compressor's alias.

### 4.9 MoE feed-forward

The FFN sublayer is a top-K Mixture-of-Experts plus one always-active
shared expert. Three classes in `Sources/DeepSeekKit/Layers/MoE.swift`:

**`Gate(config, layerId, weight, bias, tid2eid)`** — top-K routing.

- `weight: dim → n_routed_experts` — Linear (kept in F32, see below).
- `bias: [n_routed_experts]?` — optional additive bias on the logits
  before scoring.
- `tid2eid: [vocab, top_k]?` — token-id → expert-id lookup for hash
  routing.

Two paths:

- **Score-based routing** (`layerId >= n_hash_layers`): the kernel
  `moe_gate` computes `logits = x @ weight^T + bias`, applies the
  scoring function (specialised at pipeline creation time via the
  `SCORE` function constant: 0 = softmax, 1 = sigmoid, 2 =
  sqrtsoftplus), and picks the top `topK` experts per token. Final
  weights are renormalised to sum to 1 and scaled by `route_scale`.
- **Hash routing** (`layerId < n_hash_layers`): per-token expert ids
  come from `tid2eid[input_id, :]` (a precomputed table); the per-expert
  weight is `sqrt(softplus(logits[expert])) / Σ`, multiplied by
  `route_scale`. Earlier this branch used a uniform `1/topK` weight,
  which silently degraded the first three layers of V4-Flash by
  replacing learned gating with a flat average — fixed in
  `Sources/DeepSeekKit/Layers/MoE.swift:75`.

**Critical numeric note**: the gate's `Linear` is built with
`castOutputToBF16: false`. The reference `model.py:566` explicitly
runs the gate in F32:

```python
scores = linear(x.float(), self.weight.float())
```

Quantising the logits to BF16 (7 mantissa bits) before `sqrt(softplus)
+ topk` perturbs which experts get selected; on V4-Flash that
perturbation produces an 8.4× residual-stream amplification at the
first score-routed layer (`= the first layer past n_hash_layers`).
Hash-routed layers are spared because their indices come from a
precomputed table.

**`Expert(w1, w2, w3, swigluLimit)`** — single SwiGLU FFN.

```
g = w1(x)
u = w3(x)
h = silu(g) · u                      # optional clamp via swigluLimit
y = w2(h)
```

For V4-Flash the expert weights live in **FP4** (E2M1) on disk, packed
two-per-byte, with E8M0 group scales (one scale per `[1, 32]` K-block).
The FP4 GEMM kernel dequantises inline; with `--target-dtype bf16` the
converter fuses FP4 + E8M0 → BF16 ahead of time.

`MoEFFN(gate, experts, shared)`:

1. Gate: `(weights, indices) = gate(x, inputIds)`.
2. Read indices and weights back to host to build the dispatch plan
   (`MoEDispatch.prepare`): for each routed expert, the list of token
   rows assigned to it.
3. Gather: pack tokens by expert into `gathered: [T_total, dim]`.
4. Forward each active expert on its slice, writing into the right
   offset of `outs: [T_total, dim]`.
5. Scatter back into `y: [N, dim]` with a weighted sum: `y[n, d] =
   Σ_assignments weight · outs[t, d]`.
6. Add the shared expert's contribution: `y += shared(x)`.

The shared expert exists once per layer and runs on *all* tokens; its
output is summed into `y` regardless of routing. V4-Flash has
`n_shared_experts = 1`.

### 4.10 The transformer block (DecoderLayer)

`Block` (`Sources/DeepSeekKit/Layers/DecoderLayer.swift:6`) glues
attention and FFN with their HC pre/post:

```
x : [B, S, hc, dim]

# Attention sublayer
(yA, postA, combA) = HC.pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
yA = attn_norm(yA)
oA = attn(yA, start_pos)
x  = HC.post(oA, residual=x, post=postA, comb=combA)

# FFN sublayer
(yF, postF, combF) = HC.pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
yF = ffn_norm(yF)
oF = ffn(yF, input_ids)
x  = HC.post(oF, residual=x, post=postF, comb=combF)

return x
```

Each block takes the command buffer as `inout`: MLA (when the indexer
is on) and MoE both need to commit-and-wait mid-flight to read GPU
output into host memory, and they hand back a fresh command buffer.
The block continues encoding into the swapped buffer.

### 4.11 Multi-Token Prediction (MTP)

`MTPBlock` (`Sources/DeepSeekKit/Layers/MTPBlock.swift:21`) trails the
main stack. Its purpose is speculative decoding: given the current
hidden state and the *next* input token's embedding, predict that
token's logits. If the prediction matches the sampler's choice at
decode time, the next forward can skip one full pass.

The MTP block holds:

- An inner `Block` (the same Block class as the main stack, with its
  own attention + MoE).
- `e_proj`, `h_proj` — two linear projections that fuse the new-token
  embedding with the previous block's hidden state.
- `enorm`, `hnorm` — pre-projection RMSNorms.
- `norm` — final RMSNorm before the LM head.
- `hc_head_fn`, `hc_head_base`, `hc_head_scale` — its own HC head
  collapse parameters.
- Non-owning references to the shared `ParallelEmbedding` and
  `ParallelHead`.

Forward (mirrors `model.py:756`):

```
e = enorm(embed(input_ids))           # [N, dim]
xN = hnorm(x.flatten(2)).reshape([N, hc, dim])
combined[N, hc, dim] = e_proj(e)[N, 1, dim] + h_proj(xN)[N, hc, dim]   # broadcast across hc
after = inner_block(combined.reshape([B, S, hc, dim]), start_pos, input_ids)
logits = head(after, hc_head_fn, hc_head_scale, hc_head_base, norm)
```

The MTP layers contribute to `Transformer.layers` indirectly: their
own `compress_ratio` entries live at the tail of `compress_ratios`
(`compressRatios[nLayers .. nLayers+nMtpLayers-1]`). The MTP block is
**not** yet integrated into the CLI's decode loop today — the
infrastructure is in place but `Sources/deepseek/main.swift` runs the
plain head only. See `docs/ROADMAP.md` for the integration plan.

### 4.12 LM head (ParallelHead)

`ParallelHead` (`Sources/DeepSeekKit/Model.swift:45`) finalises the
forward pass. The simpler-than-block HC collapse is a *sigmoid-only*
gating (no Sinkhorn), used once at the very end:

1. `rsqrt(mean(x²) + eps)` per row over the flattened `hc·dim`.
2. `mixes = x_flat @ hc_head_fn^T` (F32 linear).
3. `mixes *= rsqrt`.
4. `pre[n, h] = sigmoid(mixes[n, h] · hc_head_scale + hc_head_base[h]) +
   hc_eps`. Done host-side via a scalar broadcast (the operation is
   cheap and the values are needed immediately for the next step).
5. `y[n, d] = Σ_h pre[n, h] · x[n, h, d]` — `hc_collapse_f32` kernel.
6. `y = norm(y)` — final RMSNorm.
7. **Slice the last sequence row per batch**: `last[b, :] = y[b, S-1,
   :]`. The LM head only produces logits for the last position.
8. `logits = last @ lm_head_weight^T` → `[B, vocab_size]` F32.

The LM head Linear has `castOutputToBF16 = false` — losing the bottom
16 mantissa bits would collapse near-ties and warp the temperature
scaling downstream.

---

## 5. The full data flow at decode

For each generated token, the CLI (`Sources/deepseek/main.swift`)
calls `Transformer.forward(inputIds: [[id]], startPos: pos)` once.
Inside:

1. **Embed lookup**: `flatIds = inputIds.flatMap { $0.map(Int32.init) }`
   → `embed.lookup(flatIds, in: cmd)` → `[1, dim]` F32.
2. **HC expand**: `hc_expand_f32` tiles to `[1, hc_mult, dim]`. The
   residual stream now lives in `[B, S, hc_mult, dim]` shape.
3. **Streaming hint**: for each layer K, `loader.ensureLayer(K)`
   (no-op when not in `.streaming` mode) — page in the layer's shard
   before referencing its Tensors.
4. **Block K** runs on its own command buffer:
   - `HC.pre` → `attn_norm` → `MLA(start_pos)` → `HC.post`.
   - `HC.pre` → `ffn_norm` → `MoEFFN(input_ids)` → `HC.post`.
   - Inside MLA: the **Compressor's kvCache alias** into the trailing
     slice of MLA's kvCache is wired the first time. After that, the
     pure-window or window+compressed top-k path runs depending on
     `compressRatios[K]`.
   - During decode, `MLA` writes one row into `kvCache[:, startPos %
     windowSize]` and the Compressor (if any) accumulates into its
     rolling state, possibly emitting one compressed row into
     `kvCache[:, windowSize + startPos / ratio]`.
5. `cmd.commit()` + `cmd.waitUntilCompleted()` between blocks.
6. `loader.releaseLayer(K)` — in `.streaming` mode this marks the
   layer's pages for `MADV_DONTNEED` so the next layer's shard has
   room.
7. **Head**: `head(x, hc_head_fn, hc_head_scale, hc_head_base, norm)`
   on a final command buffer; commit + wait.
8. Sampler picks the next id from the `[1, vocab]` logits.
9. Repeat with `startPos += 1`, feeding the freshly sampled id.

The first forward in a session is "prefill" — `startPos = 0`, `S =
prompt length`. After the prompt is digested the loop switches to
single-token decode.

---

## 6. Numeric data types per component

The model is a fluent mix of dtypes. Apple Silicon natively supports
F32, F16, BF16 (Metal 3+), I32, I8; FP8 / FP4 / E8M0 are not native
and get unpacked in shader. See [`DTYPES.md`](DTYPES.md) for full bit
layouts and the converter's fusion math.

### On disk

| Tensor family | Native HF release | Post-`--target-dtype bf16` (default) | Post-INT8 |
|---|---|---|---|
| Embeddings | BF16 | BF16 | BF16 |
| Attention linears (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`) | FP8-E4M3 + E8M0 scale per 128×128 | BF16 | INT8 + per-row F16 scale |
| Routed expert linears (`w1`, `w2`, `w3`) | FP4-E2M1 (packed) + E8M0 scale per row × 32-K-block | BF16 (4× disk!) | INT8 |
| Shared expert linears | FP8 | BF16 | INT8 |
| MoE gate (`ffn.gate`) | F32 in code, BF16 on disk | F32 | F32 |
| HC parameters (`hc_*_fn`, `hc_*_base`, `hc_*_scale`) | F32 (small) | F32 | F32 |
| RMSNorm gains (`*_norm.weight`) | BF16 | BF16 | BF16 |
| Compressor `wkv`, `wgate`, `ape` | BF16 / FP8 | BF16 | BF16 |
| Indexer `wqB`, `weightsProj` | FP8 | BF16 | INT8 |
| LM head (`lm_head.weight`) | BF16 | BF16 | BF16 |

### In flight (during a forward pass)

| Tensor | Dtype |
|---|---|
| Embedding output | F32 |
| Residual stream | F32 |
| All RMSNorm outputs | F32 |
| Linear outputs | F32 (BF16 round-trip via `castOutputToBF16: true` for everything *except* the gate and LM head) |
| RoPE freqs | F32 |
| KV cache | F32 |
| Compressor `kvState` / `scoreState` | F32 (scoreState initialised to `-Float.infinity`) |
| FP8 / FP4 GEMM inputs (activations) | Quantised on the fly via `ActQuant` — FP8 block 128, FP4 block 32 |
| Indexer-internal Q (post-Hadamard) | F32 round-tripped through FP4 |

### Activation quantization-aware noise (QAT)

DeepSeek-V4 was trained with FP8/FP4 activations on a couple of
specific paths, and the inference engine must reproduce the same
round-trip noise on inference:

- **MLA's KV nope dims**: `ActQuant.partialInplaceQuant(kv, 0,
  nopeHeadDim, blockSize=64)` — FP8 round-trip, block size 64. Mirrors
  `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` at
  `model.py:506`.
- **Compressor's MLA-side post-process**: same call on the emitted
  compressed token.
- **Indexer's Q**: `ActQuant(.fp4).quant(q, inplace: true)` mirrors
  `fp4_act_quant(q, fp4_block_size, True)`.
- **Indexer's compressor post-process**: FP4 quant + Hadamard.

Without these noise injections the residual stream amplifies into the
1e5 range in deep layers and outputs become garbage.

---

## 7. Weight naming convention

The weights tree the loader walks (`Sources/DeepSeekKit/Assembly.swift:152`).
Names are written exactly as `WeightLoader.tryLoad(_:)` looks for them,
with `try` fallbacks listed where the loader accepts multiple names.

```
embed.weight                                 # or model.embed.weight
norm.weight
head.weight                                  # or lm_head.weight
hc_head_fn
hc_head_base
hc_head_scale

# Per main layer i ∈ [0, n_layers)
layers.<i>.attn_norm.weight
layers.<i>.ffn_norm.weight

# MLA
layers.<i>.attn.wq_a.weight  + .scale | .weight_scale_inv      # quantized
layers.<i>.attn.q_norm.weight                                  # bf16/f32
layers.<i>.attn.wq_b.weight  + .scale | .weight_scale_inv
layers.<i>.attn.wkv.weight   + .scale | .weight_scale_inv
layers.<i>.attn.kv_norm.weight
layers.<i>.attn.wo_a.weight  + .scale | .weight_scale_inv      # bf16 after converter
layers.<i>.attn.wo_b.weight  + .scale | .weight_scale_inv
layers.<i>.attn.attn_sink                                      # [n_heads] f32

# Compressor (when compress_ratios[i] > 0)
layers.<i>.attn.compressor.ape                                 # [ratio, coff·head_dim]
layers.<i>.attn.compressor.wkv.weight     + .scale
layers.<i>.attn.compressor.wgate.weight   + .scale
layers.<i>.attn.compressor.norm.weight

# Indexer (when compress_ratios[i] == 4)
layers.<i>.attn.indexer.wq_b.weight       + .scale
layers.<i>.attn.indexer.weights_proj.weight + .scale
layers.<i>.attn.indexer.compressor.ape
layers.<i>.attn.indexer.compressor.wkv.weight     + .scale
layers.<i>.attn.indexer.compressor.wgate.weight   + .scale
layers.<i>.attn.indexer.compressor.norm.weight

# MoE
layers.<i>.ffn.gate.weight                  # f32 (no scale even when others quantized)
layers.<i>.ffn.gate.bias                    # optional, when i >= n_hash_layers
layers.<i>.ffn.gate.tid2eid                 # [vocab, top_k] i32, when i < n_hash_layers

layers.<i>.ffn.experts.<j>.w1.weight  + .scale     # j ∈ [0, n_routed_experts)
layers.<i>.ffn.experts.<j>.w2.weight  + .scale
layers.<i>.ffn.experts.<j>.w3.weight  + .scale

layers.<i>.ffn.shared_experts.w1.weight  + .scale
layers.<i>.ffn.shared_experts.w2.weight  + .scale
layers.<i>.ffn.shared_experts.w3.weight  + .scale

# Hyper-Connections (one set per sublayer)
layers.<i>.hc_attn_fn        # [(2+hc)·hc, hc·dim] f32
layers.<i>.hc_attn_base      # [(2+hc)·hc]         f32
layers.<i>.hc_attn_scale     # [3]                 f32
layers.<i>.hc_ffn_fn
layers.<i>.hc_ffn_base
layers.<i>.hc_ffn_scale
```

**MTP blocks** (when `nMtpLayers > 0`) live under the same `layers.*`
tree at indices `[n_layers, n_layers + n_mtp_layers)`. They have the
same MLA + MoE substructure plus the four MTP-specific tensors:
`mtp.<k>.e_proj.weight`, `mtp.<k>.h_proj.weight`,
`mtp.<k>.enorm.weight`, `mtp.<k>.hnorm.weight`, `mtp.<k>.norm.weight`,
and the MTP-block's own `hc_head_fn`, `hc_head_base`, `hc_head_scale`.

The loader is **forgiving**: any tensor it can't find is filled with
random init via `MiniRNG` (`Sources/DeepSeekKit/Assembly.swift:502`)
and reported on stderr at the end. This lets a partially-converted or
pruned checkpoint still produce a forward pass — useful for
incremental porting work.

### FP4 / FP8 storage quirks

- **FP4 routed experts** are stored as **raw `i8` bytes** in the
  HuggingFace release (safetensors has no FP4 dtype). The loader
  detects the pattern (`.experts.` in the name + `.i8` dtype) and
  reinterprets as `.fp4E2M1` with the last dim doubled
  (`Sources/DeepSeekKit/Assembly.swift:472`).
- **Scale companion names**: `weight_scale_inv` on the HF-native release
  vs `scale` post-converter. The loader tries both
  (`Sources/DeepSeekKit/Assembly.swift:492`).
- **Hash routing table (`tid2eid`)** is typically stored as `i64` on
  disk; the loader casts to `i32` via `AssemblyHelpers.castIntToI32`
  for downstream consumption.

---

## 8. KV cache: layout, lifecycle, snapshot

The KV cache is the only large dynamic allocation the model owns. It
must stay resident in unified memory (it's read and written every
token, so memory-mapping it from disk is a no-go), so its size
multiplied by `max_seq_len` × `max_batch_size` is the most important
"will this fit?" number after the weights themselves.

### 8.1 Shape per layer

For a layer with `compress_ratios[i] = r`:

```
kvCache shape = [maxBatchSize, kvCacheRows, headDim]   dtype f32
kvCacheRows   = windowSize + (r > 0 ? maxSeqLen / r : 0)
```

For V4-Flash defaults (`maxBatchSize=4`, `windowSize=128`,
`maxSeqLen=4096`, `headDim=512`):

| Layer ratio | Rows | Bytes per layer |
|---|---|---|
| 0 (pure window) | 128 | 4 × 128 × 512 × 4 B = ~1 MB |
| 4 (window + compressed) | 128 + 1024 = 1152 | ~9 MB |
| 128 (window + heavy compressed) | 128 + 32 = 160 | ~1.3 MB |

Plus the Indexer's own KV cache (only on `ratio == 4` layers) of shape
`[maxBatchSize, maxSeqLen/ratio, indexHeadDim]`, plus the Compressor
state buffers `[maxBatchSize, coff·ratio, coff·headDim]`. Total across
the V4-Flash 7-layer stack stays under a few GB.

For long-context inference (`maxSeqLen = 1 M`), this scales linearly
in the `compressed` slice — `maxSeqLen / 4 = 256 K` rows per `ratio=4`
layer, ~512 MB per such layer. That's why
`ModelConfig.projectedKVCacheBytes` refuses early when the request
exceeds the system memory budget.

### 8.2 Layout inside each batch slot

```
kvCache[b]:  [ window_size rows         ][ compress_rows                  ]
             [ sliding-window ring buf  ][ compressor.kvCache (alias slice) ]
              ↑                          ↑
              MLA writes here on every   Compressor writes here when a
              decode step (`pos%win`)    new compressed token is emitted
                                         (`pos // ratio`)
```

The Compressor and MLA write to non-overlapping regions of the same
buffer (different offsets). MLA reads the entire buffer at attention
time, with the top-k indices telling it which slots to actually load.

### 8.3 Allocation site

`Sources/DeepSeekKit/Assembly.swift:336` is the only place the KV
cache is allocated:

```swift
let kvCacheRows = config.windowSize +
    (ratio > 0 ? config.maxSeqLen / ratio : 0)
let kvCacheShape = [config.maxBatchSize, max(kvCacheRows, 1),
                    config.headDim]
let kvCache = kvCacheFile != nil
    ? kvFile.tensor(at: off.attnKVCache, shape: kvCacheShape, dtype: .f32)
    : Tensor.empty(shape: kvCacheShape, dtype: .f32)
```

The `kvCacheFile` path is an optional cross-restart persistence layer
(`Sources/DeepSeekKit/KVCacheFile.swift`): when provided, the
allocation is a zero-copy slice of an mmapped backing file instead of
a fresh MTLBuffer. The chat surface uses this to keep a chat's KV
state through quit-and-relaunch.

### 8.4 Lifecycle

- **Lazy alloc on first forward**: `MLA.ensureKVCache()` /
  `Indexer.ensureKVCache()` / `Compressor.ensureKVState()` each
  re-allocate from the saved shape if the previous call released the
  buffer. That makes `releaseCache()` cheap and non-destructive.
- **Per-layer release**: `releaseCache()` drops the cache + the
  Compressor's rolling state + the Indexer's cache. ARC frees the
  `MTLBuffer`s and unified-memory pages return to the system.
- **Transformer-wide release**: `Transformer.releaseCache()`
  (`Sources/DeepSeekKit/Model.swift:314`) walks every Block and MTP
  block and calls each one's `releaseCache()`. Useful between
  unrelated prompts or under memory pressure.
- **Restore from snapshot**: `MLA.restoreKVCacheBytes(shape:, dtype:,
  bytes:)` re-allocates from a Data blob; `Indexer.restoreKVCacheBytes`
  is symmetric and additionally re-wires the internal Compressor's
  alias to the freshly restored buffer (otherwise the next forward
  would read stale data).

### 8.5 Rewind

For incremental prefill — when the user edits the last message and
the new prompt shares a long prefix with the previous one — the
engine can rewind the KV state to a position `P` and prefill only
the delta, instead of cold-prefilling the whole new prompt.

`Transformer.rewindKVTo(pos:)` (`Sources/DeepSeekKit/Model.swift:342`):

- Returns `true` iff every layer's `MLA.rewindKVTo` and every
  Indexer's compressor rewind succeeded.
- The caller must round `pos` down to a multiple of
  `compressRatioLCM` (= 128 for the default V4 ratios), so the
  rewind is window-aligned for every compressor at once.
- On failure (or when alignment can't be honoured), the safe fallback
  is `releaseCache()` + cold prefill from `startPos = 0`.

Why this works: ratio-0 layers have no rolling state to reset, the
ring buffer self-overwrites on the next forward. Compressor layers
need their `kvState` / `scoreState` zeroed back to a clean
window-boundary; entries at positions `[0, pos)` in the main `kvCache`
are still valid (they're the preserved prefix). Entries past `pos`
will be overwritten by the next forward.

### 8.6 Snapshot/restore (sub-agent delegation)

When the desktop app delegates a sub-task to another agent, it
snapshots the current model's KV state, runs the sub-agent (which
mutates the cache), then restores the original cache. This avoids
paying a cold re-prefill on return.

Implementation lives in `Sources/DeepSeekKit/Model+KVSnapshot.swift`
(MLA + Compressor + Indexer cache + state bytes serialised to a
`KVCacheSnapshot`), and the InferenceService side wires
`beginDelegation()` / `endDelegation()` (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#agents-delegation-kv-snapshots)).

The snapshot is a "frozen" copy; the running model continues to
mutate its own cache after the call returns the snapshot id. The
restore call writes the saved bytes back into freshly allocated
buffers and re-wires the Compressor / Indexer aliases.

---

## 9. Putting some numbers on V4-Flash

For the default V4-Flash config (`n_layers = 7`, `dim = 4096`,
`n_routed_experts = 256` in production, `n_activated_experts = 2`,
`moe_inter_dim = 4096`, `n_heads = 64`, `head_dim = 512`):

### Per-layer parameter counts

| Tensor | Shape | FP4/FP8 bytes | BF16 bytes |
|---|---|---|---|
| `attn.wq_a` | [4096, 1024] | 4 MB | 8 MB |
| `attn.wq_b` | [1024, 64·512=32768] | 32 MB | 64 MB |
| `attn.wkv` | [4096, 512] | 2 MB | 4 MB |
| `attn.wo_a` | [4096·512/8, 8·1024=8192] | always BF16 | 64 MB |
| `attn.wo_b` | [8·1024, 4096] | 8 MB | 16 MB |
| `attn.attn_sink` | [64] | 256 B | 256 B |
| All compressor / indexer per-layer | varies | ~10–20 MB | ~20–40 MB |
| One expert (`w1+w2+w3`) | 3 × [4096, 4096] | 12 MB (FP4) | 96 MB (BF16) |
| 256 experts | — | ~3 GB (FP4) | ~24 GB (BF16) |
| Shared expert | — | 48 MB | 96 MB |
| Gate (`ffn.gate.weight`) | [256, 4096] | 1 MB | (always F32 in code) |
| `attn_norm` + `ffn_norm` + …norms | — | tens of KB | — |
| HC params per block | 6 small | a few KB | a few KB |

A V4-Flash block with all 256 experts in FP4 + attention in FP8 lands
around **3–4 GB on disk**. With 7 blocks plus MTP and the embed/head,
that totals ~28 GB — but the released checkpoint is **~142 GB**
because the full V4-Flash has more layers (the field default `n_layers
= 7` is a toy default; the actual released config carries the real
value).

### Active-parameters-per-token

With `n_activated_experts = 2 + n_shared_experts = 1` of 256 experts
active per token, the per-token MoE workload is `3 / 256 ≈ 1.2%` of
the total expert weight. That's where the "284 B params, 13 B
activated" headline comes from.

### KV cache projection (default config)

`ModelConfig.projectedKVCacheBytes`:

```
for each layer i with ratio r:
    cacheRows = (r > 0) ? windowSize + maxSeqLen / r : windowSize
    + attention kvCache:  maxBatchSize · cacheRows · headDim · 4 bytes
    + indexer kvCache (only if r == 4): same shape
    + compressor kvState (only if r > 0):
        hcMult · ratio · maxBatchSize · headDim · 4 bytes
```

For 7 layers at default settings: ~30–60 MB total. For long-context
(`max_seq_len = 1 M`): ~tens of GB. The loader refuses early when this
exceeds `SystemProbe.effectiveProcessBudget()` so the process doesn't
crash mid-allocation.

---

## 10. Operational concerns

### 10.1 Loader strategies

The loader picks one of three strategies based on available RAM
(`Sources/DeepSeekKit/LoadStrategy.swift`):

- **`preload`** (≥ 192 GB RAM): every shard slurped into memory upfront.
  Fastest steady-state, highest cold-start cost.
- **`mmap`** (32–192 GB): every shard mmapped, the OS pages weights in
  on demand. The first forward triggers ~13 GB of sequential SSD
  reads (~2 s on Apple SSDs at 7 GB/s).
- **`streaming`** (16–32 GB): one layer's shard at a time. The
  Transformer's `forward(...)` calls `loader.ensureLayer(K)` before
  block K runs and `loader.releaseLayer(K)` after, bounding the
  working set to "the layer the GPU is currently reading from".

See [`MEMORY.md`](MEMORY.md) for full mmap details and per-phase
footprint estimates.

### 10.2 Streaming and per-layer commits

The decision to commit each block's command buffer separately (rather
than batching the whole forward into one big buffer) is a
deliberate trade-off:

- ✅ The streaming-pool loader can rotate shards layer-by-layer
  (working set stays bounded).
- ✅ Per-layer numerical traces under `--trace-norms` have a clean
  sync point.
- ❌ Some GPU concurrency opportunities are lost.

For the V4-Flash size on Apple Silicon this is the right balance:
the model is mostly memory-bound, the per-layer commit overhead is
small compared to the per-layer GEMM cost, and the alternative
(one big buffer) would either OOM on streaming or block until the
whole forward completes before any progress is visible to the
caller.

### 10.3 Rewind alignment and the LCM

`compressRatioLCM` is the lowest common multiple of every non-zero
entry in `compressRatios`. For the default V4 `[0, 0, 4, 128, 4, 128,
4, 0]`, that's 128.

For `Transformer.rewindKVTo(pos:)` to succeed, `pos` must be a
multiple of `compressRatioLCM`. Concretely:

- Position 0 ✓ (full reset).
- Position 128 ✓ (one window).
- Position 256 ✓.
- Position 100 ✗ — mid-window for `ratio == 128`.
- Position 132 ✗ — mid-window for `ratio == 4`.

The caller is expected to round down to the nearest LCM multiple
before calling. If the rounded value is too far back to be useful,
fall back to a full `releaseCache()` + cold prefill.

### 10.4 Random init for missing tensors

The loader logs a warning and fills in random F32 weights when a
named tensor is missing. This is on purpose: a partially-pruned
checkpoint can still run a forward, which is useful when porting,
when debugging the loader path, or when iterating on the layer
shapes against a not-yet-finalised release.

In the absence of all weights, `Transformer.randomInit(config:)`
builds the same structure with random F32 everywhere — used by the
smoke tests in `Tests/DeepSeekKitTests/` to verify the full forward
chain end-to-end without any checkpoint on disk.

---

## 11. Python ↔ Swift cross-walk

The full line-by-line table lives in [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md).
The model-relevant entries (every class and major function in
`Reference/inference/model.py`):

| Python | Lines | Swift |
|---|---|---|
| `ModelArgs` dataclass | 34–81 | `Sources/DeepSeekKit/Config.swift` (`ModelConfig`) |
| `ParallelEmbedding` | 83–105 | `Sources/DeepSeekKit/Model.swift` (`ParallelEmbedding`) |
| `linear` dispatch fn | 108–120 | `Sources/DeepSeekKit/Layers/Linear.swift` |
| `Linear` class | 123–152 | same file |
| `RMSNorm` | 183–196 | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| `precompute_freqs_cis` (YaRN) | 199–229 | `Sources/DeepSeekKit/YaRN.swift` |
| `apply_rotary_emb` | 232–244 | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Kernels/rope.metal` |
| `rotate_activation` (Hadamard) | 247–251 | `Sources/DeepSeekKit/Layers/Hadamard.swift` + `Kernels/hadamard.metal` |
| `get_window_topk_idxs` | 254–265 | `Sources/DeepSeekKit/Layers/AttentionIndices.swift` |
| `get_compress_topk_idxs` | 268–276 | same file |
| `Compressor` | 279–377 | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| `Indexer` | 380–433 | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| `Attention` (MLA) | 436–543 | `Sources/DeepSeekKit/Layers/Attention.swift` |
| `Gate` | 546–584 | `Sources/DeepSeekKit/Layers/MoE.swift` (`Gate`) |
| `Expert` | 587–606 | same file (`Expert`) |
| `MoE` | 609–644 | same file (`MoEFFN`) |
| `Block` | 647–700 | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| `Block.hc_pre / hc_post` | 673–686 | `Sources/DeepSeekKit/Layers/HyperConnections.swift` |
| `ParallelHead` | 703–735 | `Sources/DeepSeekKit/Model.swift` (`ParallelHead`) |
| `MTPBlock` | 738–766 | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| `Transformer` | 769–809 | `Sources/DeepSeekKit/Model.swift` (`Transformer`) |

Open the Python first when answering an architectural question; the
Swift mirrors it. When the Swift seems to diverge — for example the
explicit FP8/FP4 QAT noise calls, the `inout cmd` pattern, or the per-
batch alignment guards inside the Compressor — comments in the Swift
source explain why.

---

## 12. Source map

By the topic you care about, here's where to open first.

| Topic | File |
|---|---|
| Config + field aliases + KV budget projection | `Sources/DeepSeekKit/Config.swift` |
| Tensor + DType enum | `Sources/DeepSeekKit/Tensor.swift` |
| Embed + Head + Transformer | `Sources/DeepSeekKit/Model.swift` |
| Assembly (weight name tree, random init, load path) | `Sources/DeepSeekKit/Assembly.swift` |
| MLA forward | `Sources/DeepSeekKit/Layers/Attention.swift` |
| MoE gate + expert + scatter/gather | `Sources/DeepSeekKit/Layers/MoE.swift` + `MoEDispatch.swift` |
| Hyper-Connections (pre/post + Sinkhorn) | `Sources/DeepSeekKit/Layers/HyperConnections.swift` + `HCSinkhorn.swift` |
| Compressor (prefill, decode, overlap) | `Sources/DeepSeekKit/Layers/Compressor.swift` |
| Indexer (top-k selector) | `Sources/DeepSeekKit/Layers/Indexer.swift` |
| Block (HC + sublayer composition) | `Sources/DeepSeekKit/Layers/DecoderLayer.swift` |
| Multi-Token Prediction | `Sources/DeepSeekKit/Layers/MTPBlock.swift` |
| RoPE + YaRN | `Sources/DeepSeekKit/Layers/RoPE.swift` + `Sources/DeepSeekKit/YaRN.swift` |
| RMSNorm | `Sources/DeepSeekKit/Layers/RMSNorm.swift` |
| Sparse attention (FlashAttention-style) | `Sources/DeepSeekKit/Layers/SparseAttention.swift` + `Kernels/sparse_attn.metal` |
| Linear (BF16 / FP8 / FP4 / INT8/4/2 dispatch) | `Sources/DeepSeekKit/Layers/Linear.swift` |
| FP8 / FP4 act quant (QAT noise) | `Sources/DeepSeekKit/Layers/ActQuant.swift` + `Kernels/act_quant.metal` |
| KV cache snapshot/restore + delegation | `Sources/DeepSeekKit/Model+KVSnapshot.swift` |
| KV cache persistence (cross-restart) | `Sources/DeepSeekKit/KVCacheFile.swift` + `KVCacheLayout.swift` |
| Loader + streaming strategy | `Sources/DeepSeekKit/WeightLoader.swift` + `LoadStrategy.swift` + `StreamingPool.swift` |
| Per-Metal-kernel reference | [`KERNELS.md`](KERNELS.md) |
| Per-file index for everything | [`MODULES.md`](MODULES.md) |

---

## 13. Known limitations and deferred work

Tracked in detail in `TODO.md` (project root) and
[`ROADMAP.md`](ROADMAP.md). Model-specific items at a glance:

- **MTP is not wired into the decode loop.** The block class exists,
  the loader builds it, but `Sources/deepseek/main.swift` runs only
  the plain LM head. Speculative decoding would need: take MTP's
  `logits`, check if the sampler's choice matches, conditionally skip
  the next prefill.
- **`act_quant` QAT noise on non-rope KV dims uses a contiguous
  partial-block kernel.** A strided Tensor view would let the same
  kernel touch arbitrary slices without a copy. Tier 3 — deferred
  because the contiguous path covers the actual call sites.
- **Numerical validation against Python is hand-tested, not
  automated.** A harness exists in `Tests/` but a full PyTorch +
  CUDA-equivalent reference run isn't part of CI (needs CUDA).
- **`cast_e2m1fn_to_e4m3fn` in the converter is not ported.** The
  `--expert-dtype fp8` path falls back to relabel-only; the BF16
  fusion path (`--target-dtype bf16`, default) doesn't need it.
- **Multi-batch prefill of Compressor** with `S % ratio != 0` is
  guarded: B > 1 prompts must have `S` divisible by `ratio` for now.
  The single-batch path supports any remainder.

For perf optimisations (simdgroup matrix, fp8 layout tweaks) see
[`PERFORMANCE.md`](PERFORMANCE.md).
