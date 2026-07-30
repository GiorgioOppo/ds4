**English** | [Italiano](README.it.md)

# Laguna engine

`LagunaResidentModel` is the first-cut resident engine: every validated
tensor of the official Q8_0-signal / Q4_K-routed recipe is uploaded once into
shared `MTLBuffer`s (Laguna requires full residency upstream — no SSD
streaming, sidecars or expert cache), each layer owns an F16 ring KV cache
(512 rows on sliding-window blocks, the configured context on full-attention
blocks), and the per-token graph mirrors `laguna_graph_forward_token`:
RMSNorm → paired Q8_0 matvecs (Q/K, V/gate) → per-head norm/RoPE → ring
store → gated GQA attention → output projection and residual → dense or
routed FFN. It dispatches the shared GLM primitives where upstream shares
them (`kernel_glm52_rms_norm_f32`, `kernel_glm52_matvec_pair_sg`,
`kernel_glm52_router_select` with configurable top-N, the `glm52_moe` K-quant
matvecs) plus the Laguna kernels beside them. The router selection is read
back on the host to address expert slabs, like the GLM chained decode.

Routed experts may be Q2_K, Q3_K or Q4_K per layer (coherent, as the schema
guarantees), so both the official Q4_K_M file and the mixed
RoutedQ2_K-Last27Q3_K file run; the Q3_K dot helpers live beside the other
K-quants in `metal/glm5.2/glm52_quant.metal`. Deliberate scope limits of
this cut, refused with distinct errors at load: the legacy F16/Q4_K recipe
(its matvec paths are not wired).
`LagunaResidentModelOptions.layerCount` truncates the stack from the front
for bring-up runs.

Multi-token prompts use a layer-major prefill. Q/K norm+RoPE runs over the
whole chunk, new K/V rows are staged as F16, gated causal GQA attends all
rows in parallel (sharing each K/V load across the six production query
heads), and the staged rows are committed afterward. Staging preserves the
sequential ring semantics when a sliding-window chunk wraps. Activation
planes are retained and grown only when a later prompt needs a wider chunk.
Set `DS4_LAGUNA_PREFILL_BATCH=0` to restore the per-token attention dispatches
for logits or performance A/B diagnostics.
`DS4_PREFILL_DENSE_MM` (alias `DS4_LAGUNA_PREFILL_DENSE_MM`) defaults on and
also batches RMSNorm, Q/K/V/gate, the attention output projection, router,
and dense/shared FFNs. It uses the Q8_0 multi-token matmuls shared with
DeepSeek; F16 activation staging makes it numerically close but not
bit-identical. Set it to `0` to restore the exact matvec path. On the mixed
GGUF, M1 Pro, top-6 and chunk 256, a 414-token A/B reduced prefill from
73.8 to 32.6 seconds (5.61 to 12.72 tokens/s).

During chat decode, sampling reads the shared Metal logits buffer directly
instead of copying all 100,352 floats into a new Swift array per token.
Explicit repetition penalty remains correct and takes a private copy because
it must mutate selected logits.

Three lossless decode variants remain available for A/B runs, but default
off because they regressed wall clock on the target M1 Pro: baseline
1.94–1.97 tok/s, `DS4_DECODE_CHAINED=1` 1.17 tok/s,
`DS4_DECODE_SPLIT_K=1` 0.59 tok/s, and
`DS4_DISCARD_UPLOAD_PAGES=1` 1.74 tok/s (roughly 1k prompt, top-6, 2 GiB
cache). Chained overlaps expert tail N with attention/router trunk N+1;
split-K applies grouped GQA to the 12 global layers beyond
`DS4_DECODE_SPLIT_K_MIN` (384); discard drops copied mmap interior pages.
They remain explicit experiments for different GPUs and memory pressure.

`DS4_SHARED_EXPERT_OVERLAP` (alias
`DS4_LAGUNA_SHARED_EXPERT_OVERLAP`) starts the resident shared
expert before the first routed-slab wait. It writes a separate accumulator
while `pread` advances, and the closing pair of additions preserves the exact
`(after_attn + routed) + shared` association. This applies to decode and
layer-major prefill. It remains off in the M1 Pro preset: top-10 A/B results
were unstable and the stabilized measurements favored the non-overlapped path.

On the 12 global layers, `DS4_INDEXED_ATTN` (alias
`DS4_LAGUNA_INDEXED_ATTN`, on by default) builds an F16 block-centroid index
during prefill. Beyond 4,096 tokens, each query head scores and selects the
top 32 blocks entirely on GPU, then attention reads the original F16 K/V rows
of those blocks plus the latest 512 dense tokens. Global decode attention is
therefore bounded to roughly `topBlocks × blockSize + recent`, rather than
growing with the whole context. Laguna has none of DeepSeek's learned
compressor/indexer weights, so centroid lookup is an approximate sparse
selection while K/V for selected tokens remains exact. Controls are
`DS4_LONG_ATTN_BLOCK=16`, `DS4_LONG_ATTN_TOP_BLOCKS=32`,
`DS4_LONG_ATTN_RECENT=512`, and `DS4_LONG_ATTN_THRESHOLD=4096`, with matching
`DS4_LAGUNA_INDEXED_ATTN_*` aliases. Set `DS4_INDEXED_ATTN=0` to restore
original dense attention.

The active KV grows lazily: the 12 global layers start at
`DS4_KV_INITIAL`/`DS4_LAGUNA_KV_INITIAL=512`, while the 36 sliding layers
keep their fixed 512-row rings. A configured 32k maximum therefore no
longer reserves roughly 1.5 GiB up front: on a 999-token test, initial KV
fell from 1,608 to 96 MiB and decode rose from 0.36 to 1.67 tok/s. Excess
global capacity is released when switching conversations.

The GUI service also supports prefix-indexed native `LKV1` disk
checkpoints, like DeepSeek/GLM. Live KV stays in Metal during decode; SSD
storage suspends/restores sessions to avoid repeated prefill. Writes stream
through F_NOCACHE without building a second cache-sized `Data`. Below 512
tokens a file costs about 192 KiB/token; beyond the window it costs about
48 KiB/token plus a fixed 72 MiB for sliding rings.

`LagunaResidentModelOptions.expertStreaming` is an opt-in, declared
divergence from upstream (which mandates full residency for Laguna): the
Q8_0 signal path stays resident and the selected routed-expert slabs are
read with `pread`/`F_NOCACHE` directly into an LRU cache of shared Metal
buffers. Hits skip both I/O and copying. Demo and GUI default to 2,048 MiB
while streaming (529 slots on the tested mixed GGUF);
`DS4_LAGUNA_EXPERT_CACHE_MB=3072` remains available for A/B runs and `=0`
disables the cache. On the 16 GB M1 Pro, 3,072 MiB raises hits from 46% to
53% and cuts gather traffic, but its memory pressure hurts both prefill and
decode. The best observed wall clock remains the 2,048 MiB profile.
`DecodeProfile`
(`profileReport()`) reports the per-phase cost.

`DS4_ACTIVE_EXPERTS` (alias `DS4_LAGUNA_ACTIVE_EXPERTS`) reduces the
actually-executed top-k from 10 to `1...10`. The router selects top-N
directly and renormalizes the remaining weights, matching DeepSeek;
buffers, staging, I/O and compute shrink together. The engine default 10
preserves upstream numerics; the M1 Pro/10 GiB GUI preset uses the measured
fast top-6 profile. `DS4_EXPERT_CACHE_SLOTS` can set the global slot count
and takes precedence over the MiB budget; `DS4_RESIDENT_LAYERS` keeps the
first N routed layers resident while streaming the rest.

The other lossless controls shared with DeepSeek are `DS4_PREFILL_CHUNK`
(default 256, further limited by cache capacity), `DS4_EXPERT_PREAD` (on),
`DS4_PREAD_SPLIT` (`1...8`), `DS4_WILLNEED_EXPERTS` (mmap fallback),
`DS4_MTLIO` (opt-in with automatic fallback), `DS4_MLOCK` (opt-in for the
head and expert pool), and `DS4_NSG` (`1...8`, default 4). Each also accepts
its `DS4_LAGUNA_*` override.

`DS4_LAGUNA_PREFILL_MOE_BATCH=1` enables an experimental expert-major MoE
prefill variant. It remains off by default: on the same GGUF it increased
prefill time from 12.8 to 13.2 seconds.

`LagunaRuntimeGate.enabled` stays `false` until this engine passes
end-to-end logits parity against the reference C engine on real weights;
selection, catalog availability and the demo dispatch all key off that one
constant.
