# DeepSeek-V4-Flash — Compressed-KV + Sparse-Indexer Attention Port

Reference spec + plan for finishing the pure-Swift `StreamingDecoder` so it runs
DeepSeek-V4-Flash **faithfully**, including the per-layer KV compression and the
sparse indexer that the C engine (`antirez/ds4`, `ds4.c`) implements.

> Why this exists: the first pure-Swift decode path implemented only the
> **uncompressed dense** attention (ratio==0). DeepSeek-V4-Flash uses compressed
> attention for **41 of its 43 layers** (every layer `il >= 2`). The model file
> `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-...gguf` literally
> names the `F16` compressor/indexer tensors that the dense path ignores, so it
> produces incoherent output. This is architectural, not a quant bug: the q2 and
> q4 GGUFs share this architecture. (The q4 quant dispatch itself is correct.)

All line numbers below refer to `ds4.c` in `antirez/ds4`. The authoritative
numeric reference is the CPU path (`compressor_decode_one` @8478,
`indexer_allowed_decode_one` @8907, helpers @2405-2568); the Metal graph
(`metal_graph_encode_decode_layer` @13919) issues kernels reproducing it.

## 0. Shape constants (Flash)

```
n_embd=4096  n_head=64  n_head_kv=1  n_head_dim=512  n_rot=64
n_lora_q=1024  n_lora_o=1024(rank)  n_out_group=8
n_indexer_head=64  n_indexer_head_dim=128  n_indexer_top_k=512
n_swa=128                       # raw cache cap on compressed layers (sliding window)
rope_freq_base=10000  compress_rope_freq_base=160000  rope_scale_factor=16  rope_orig_ctx=65536
comp_cap (per layer) = ctx_size/4 + 2          # max emitted compressed rows
```

Per-layer ratio (`ds4_expected_layer_compress_ratio` @608, Flash):
`il<2 -> 0`; even `il -> 4`; odd `il -> 128`.  `compressed = ratio != 0`.

`coff = (ratio==4) ? 2 : 1`; `comp_width = coff*head_dim` (attn: 1024/512),
`index_width = coff*128` (= 256 for ratio 4).

## 1. Attention cases by ratio

- **ratio==0** (layers 0,1): dense causal attention over the raw per-token KV
  cache only. No compressor/indexer. Dense RoPE (base 10000, scale 1, ext 0).
  *(This is what the Swift port already does.)*
- **ratio==128**: compressor emits one pooled row every 128 tokens. **No indexer.**
  Attention = raw SWA window + **all** compressed rows (dense). Long-context RoPE.
- **ratio==4**: attn compressor (row every 4 tokens) **plus** a parallel indexer
  compressor lane (row every 4 tokens). Once `n_index_comp > 512`, the indexer
  scores all indexer rows and selects **top-512**; attention runs over raw SWA
  window + those 512 selected compressed rows. Below that threshold it behaves
  like the dense raw+all-compressed path.

## 2. Compressor (per layer, recurrent across the sequence) — `compressor_decode_one` @8478

Input `x` = `attn_norm` (the RMS-normed hidden, n_embd wide). Same `attn_norm`
feeds q_a, kv, attn-compressor, indexer-compressor, and indexer_proj.

Rolling state: `state_kv`, `state_score`, each `rows × width` floats.
`rows = ratio` (ratio!=4) or `2*ratio = 8` (ratio==4, two lanes). The compressed
**cache** is a separate growing array `comp_cache[n_comp][head_dim]`.

Per token (every token):
1. **Paired F16 matvec** of `x`: `kv_cur = Wkv^T·x`, `sc_cur = Wgate^T·x`
   (`attn_compressor_kv`, `attn_compressor_gate`, both F16 `[n_embd, comp_width]`).
2. **APE on score only**: `sc_cur[j] += ape[j, pos%ratio]`
   (`attn_compressor_ape` F16 `[comp_width, ratio]`).
3. **Store** into state row `row = (ratio==4 ? ratio + pos%ratio : pos%ratio)`.
   (matches `kernel_dsv4_compressor_store_one`, dsv4_kv.metal:288.)
4. If `(pos+1) % ratio != 0` → return (no emit).

On emit (`(pos+1) % ratio == 0`), produce one row of `head_dim`:

5. **Per-dimension softmax pool** over scores (`compressor_pool_decode_state` @8423):
   - ratio!=4: `w_r = exp(score[r][j]-max)`, `out[j] = Σ_r w_r·kv[r][j] / Σ_r w_r`.
   - ratio==4 (two lanes): for `r in 0..ratio`, combine a "prev" term `(r, j)` and a
     "cur" term `(ratio+r, head_dim+j)`:
     `wp=exp(score[r][j]-max)`, `wc=exp(score[ratio+r][head_dim+j]-max)`,
     `out[j] = (Σ wp·kv[r][j] + wc·kv[ratio+r][head_dim+j]) / Σ(wp+wc)`.
     `max` is joint over all wp/wc; if `max<=NEG_INF*0.5` → `out[j]=0`.
6. **RMS-norm with weight** `attn_compressor_norm` F32 `[head_dim]`:
   `rms = 1/sqrt(mean(out^2)+eps)`, `out[i] *= rms * norm[i]`.
7. **RoPE tail** at position `comp_pos = pos+1-ratio` (window start), long-context
   params (§5), n_rot=64.
8. **Quantize round-trip** (part of the forward math, not just storage):
   - attn (head_dim==512): `dsv4_fp8_kv_quantize_row` on the **NoPE** part only
     (first `head_dim-n_rot = 448` dims, 64-wide blocks, E4M3FN). RoPE tail left as-is.
   - indexer (head_dim==128): `dsv4_indexer_qat_row` (§3b).
9. **Append** to `comp_cache[n_comp]`, `n_comp++`.
10. ratio==4 **lane rotation** (@8552): rows 4..7 → 0..3, then 0..3 → 4..7.

## 3. Sparse indexer (ratio==4 only) — `indexer_allowed_decode_one` @8907

### 3a. Indexer compressor lane
Same algorithm as §2 with `head_dim=128`, `index_width=256`, weights
`indexer_compressor_{kv,gate,ape,norm}`. Emits into `index_comp_cache[n_index_comp]`,
quantized at emit by **QAT** (§3b). `n_index_comp++` on emit.

### 3b. QAT (`dsv4_indexer_qat_row_inplace_cpu` @2531) — applied AFTER RoPE
1. **128-pt Hadamard** (fast Walsh-Hadamard butterfly, strides 1..64) then ×`1/√128`
   (`0.08838834764831845`).
2. **FP4/E2M1 activation quant** in 32-wide blocks: `amax`(floor 7.05e-38),
   `scale=2^ceil(log2(amax/6))`, clamp ±6, E2M1FN dequant {0,.5,1,1.5,2,3,4,6}×scale.

### 3c. Query, scoring, top-k (only when `n_comp>threshold(0)` AND `n_index_comp>512`)
1. `indexer_q[64×128] = indexer_attn_q_b^T · qr_norm` (input is **qr_norm**, the
   q-LoRA norm — not attn_norm). Then RoPE tail (pos), then QAT per head (§3b).
2. `weights[64] = indexer_proj^T · attn_norm` (F16 `[n_embd,64]`).
   `index_scale = 1/sqrt(128*64)`.
3. **Score** each indexer row `c`: `s_c = Σ_h ReLU(dot(idx_kv[c], q[h])) · weights[h] · scale`.
4. **Top-512** selection (`ds4_gpu_indexer_topk_tensor` @14353) → `comp_selected`,
   `n_selected = min(512, n_index_comp)`. **Keep the full 512** (config contract
   @14376). Indexer row `c` maps 1:1 to attention compressed row `c`.

## 4. GPU op order — `metal_graph_encode_decode_layer` @13919

Common prefix (all ratios): rms_norm(flat_hc) → matmul(hc_attn_fn) →
hc split+weighted-sum+norm → **attn_norm**; matmul_q8(qr, attn_q_a); matmul_q8(kv_raw,
attn_kv) → fused q/kv rms-norm → qr_norm, kv; matmul_q8(q, attn_q_b); head_rms_norm(q);
rope_tail(q, pos); rope_tail(kv, pos); kv_store(kv → raw_cache[raw_row]).

`if compressed`: paired matvec(comp_kv/gate) → compressor_update (APE+store+pool+norm+
rope) → on emit fp8_quantize + n_comp++.  `if ratio==4`: indexer compressor lane →
on emit qat + n_index_comp++ → if gated: indexer_q (matmul+rope+qat), weights
(matmul indexer_proj), indexer_score_one, topk(512) → comp_selected.

Attention: `raw_start = raw_start_for_span(pos, n_raw)`.
- sparse (ratio4 + selection): `attention_indexed_mixed` over SWA raw rows + selected
  compressed rows (`kernel_dsv4_indexed_mixed_attention_heads8_rb16`).
- else: `attention_decode_heads` over raw + all `n_comp` compressed rows (or raw-only
  if n_comp==0).

Post-attention (all ratios): rope_tail(heads, inverse=true) → attn_output_a (low/q8)
→ attn_output_b + HC-expand → after_attn_hc.  Then HC pre-FFN, router, MoE (unchanged).

## 5. RoPE — @13944, @6655

```
freq_base  = compressed ? 160000 : 10000
freq_scale = compressed ? 1/16   : 1
ext_factor = (compressed && scale_factor>1) ? 1 : 0
attn_factor= 1; if ext_factor!=0 && freq_scale>0: attn_factor /= 1 + 0.1*ln(1/freq_scale)
n_ctx_orig = compressed ? 65536 : 0
```
Raw q/kv and the compressed stream use the **same** base/scale on a compressed
layer; only the position differs (raw uses `pos`, compressed row uses `pos+1-ratio`,
indexer query uses `pos`). RoPE acts on the tail n_rot=64 lanes; the `attn_factor`
pre-divide (@13948) and `mscale` re-multiply (@6689) cancel to net magnitude 1.0 —
replicate both halves to stay bit-faithful. (Swift `ropeTail` already matches this;
`DSV4Shape.ropeParams` already returns these per-layer params.)

QAT never replaces RoPE — they compose (RoPE on the 64-lane tail first, then
Hadamard128+FP4 over all 128 lanes for indexer rows/query).

## Gotchas
- `attn_norm` feeds q_a/kv/both compressors/indexer_proj; `qr_norm` feeds q_b and
  indexer_attn_q_b.
- APE adds to the **score/gate** stream only, indexed `[j, pos%ratio]`.
- ratio==4 doubled width = two-lane (low/high half) strided window + lane rotation.
- Compressed-row RoPE position is the **window start** `pos+1-ratio`.
- Quant round-trips (E4M3 attn NoPE; Hadamard+FP4 indexer) change cache values — part
  of the forward math.
- Top-512 is a hard pre-softmax mask; keep the full 512.
- Compressed layers cap the raw cache at `n_swa=128`; older tokens live only as
  compressed rows. This is a different cache layout than the ratio==0 full raw cache.

## GGUF tensor names (per layer `blk.<il>.`)
```
attn_compressor_ape.weight   F16 [comp_width, ratio]      (ratio != 0)
attn_compressor_kv.weight    F16 [n_embd, comp_width]
attn_compressor_gate.weight  F16 [n_embd, comp_width]
attn_compressor_norm.weight  F32 [head_dim]
indexer.attn_q_b.weight      F16/Q8_0 [n_lora_q, 64*128]  (ratio == 4)
indexer.proj.weight          F16 [n_embd, 64]
indexer_compressor_ape.weight   F16 [index_width, ratio]
indexer_compressor_kv.weight    F16 [n_embd, index_width]
indexer_compressor_gate.weight  F16 [n_embd, index_width]
indexer_compressor_norm.weight  F32 [128]
```

## Available Metal kernels (already embedded; need Swift dispatch wrappers)
| capability | kernel | Swift wrapper |
|---|---|---|
| compressor frontier store | `kernel_dsv4_compressor_store_one` (dsv4_kv.metal:288) | `compressorStoreOne` ✓ |
| softmax pool | `kernel_dsv4_softmax_pool` | ✓ |
| ratio4 lane shift | `kernel_dsv4_ratio4_shift_f32` | `ratio4Shift` ✓ |
| fp8 KV quant (E4M3) | `kernel_dsv4_fp8_kv_quantize_f32` | `fp8KVQuantize` ✓ |
| indexer Hadamard+FP4 (QAT) | `kernel_dsv4_indexer_hadamard_fp4_f32` | `indexerHadamardFP4` ✓ |
| indexer score (decode) | `kernel_dsv4_indexer_score_one_direct` (dsv4_misc.metal:142) | **MISSING** |
| indexer weighted-sum | `kernel_dsv4_indexer_weighted_sum` | `indexerWeightedSum` ✓ |
| top-k argsort | `kernel_argsort_f32_i32_desc` (≤256) + `kernel_argsort_merge_f32_i32_desc` | `argsortTopKDesc` ✓ / merge **MISSING** |
| topk mask + scatter | `kernel_dsv4_topk_mask*` | `topkMaskAndScatter` ✓ |
| paired F16 matvec | `ds4_gpu_matmul_f16_pair` | **MISSING** (or two `matmulF16`) |
| indexed mixed attention | `kernel_dsv4_indexed_mixed_attention_heads8_rb16` (dsv4_misc.metal:685) | **MISSING** |

## Staged implementation plan
1. **Foundation** (this commit): shape constants (`nSWA`, indexer dims, `compCap`,
   `coff`/`compWidth`/`indexWidth`/`emit` helpers), `LayerWeights` optional
   compressor/indexer tensors, `GGUFWeights.layer` loads them for ratio≠0 layers.
2. **Cache architecture**: per-layer compressed caches + recurrent state buffers in
   `StreamingDecoder`; compressed layers use an `n_swa`-capped circular raw cache.
3. **Kernel wrappers**: add the MISSING Swift dispatchers (indexer score-one, argsort
   merge for top-512, paired F16 matvec, indexed mixed attention) by matching the
   `.metal` arg structs + the `ds4_metal.m` fillers.
4. **Graph wiring**: branch `decodeRoute` on `compressRatio`; run §2/§3 then the right
   attention case in the exact §4 order.
5. **Verify on macOS** against the real model (golden vectors in `tests/test-vectors`
   of the C repo). None of stages 1–4 can be built/verified on Linux (no Swift
   toolchain, no Metal GPU, no full model in this environment).
