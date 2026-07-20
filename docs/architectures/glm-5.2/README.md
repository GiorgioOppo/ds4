**English** | [Italiano](README.it.md)

# GLM 5.2

This folder describes the native port of GLM 5.2 (`glm-dsa`) in DwarfStar.
The port is **operational end to end**: the streaming engine
(`GLM52ResidentModel`) runs prefill and decode from the real GGUF on a
16 GB machine, and GUI, demo, local server, benchmark, auto-tune and disk-KV
checkpoints all select GLM natively. Only distribution remains DeepSeek-only.

## Current state

| Capability | State |
|---|---|
| Catalog and GUI download | yes, three monolithic GGUFs with pinned revision/size/SHA |
| Recognition of `general.architecture = glm-dsa` | yes |
| Configuration, GGUF schema, tokenizer, chat/tool protocol | yes |
| End-to-end prefill, decode and logits output | **yes** — streaming engine, layer-major prefill, greedy/sampled decode |
| GUI and `DS4Demo` selection | **yes** (chat, settings per-backend, tuning) |
| Local server (stateless OpenAI/Anthropic semantics) | **yes**, with incremental-KV prefix reuse |
| Benchmark and auto-tune | **yes** (GUI button and `DS4_GLM_AUTOTUNE=1`) |
| Disk-KV checkpoint | yes, single `state.glmkv` per model (longest-prefix restore) |
| Per-phase profiling | yes — DeepSeek-style report plus GPU/host split |
| Distribution | no (DeepSeek-only) |

## Engine and measured optimizations (M1 Pro 16 GB, IQ2_XXS)

The engine streams: 3 leading dense layers resident (adaptive floor under
RAM pressure — more residents get paged by the OS and cost ~750 ms/token of
driver residency, measured), the 75 sparse layers stream their big tensors
per token with double-buffered prefetch, and the routed experts arrive as
contiguous records through the staging arena. Optimizations are engine
defaults, each with a measured verdict:

- **commit fusion** (`DS4_GLM_FUSE`): layer N's FFN and layer N+1's trunk in
  one command buffer — half the synchronous waits (~154 → ~77/token);
- **vectorized kernels**: IQ2_XXS/Q8_0/Q4_K dots with float4 FMA and wide
  loads (−19% GPU);
- **batched MoE** (`DS4_GLM_MOE_BATCH`): every routed expert in two
  dispatches; the shared expert (always active) stays separate;
- **fused GPU router** (`DS4_GLM_GPU_ROUTER`): matvec + sigmoid + top-8 in
  the trunk commit, 64-byte readback (−18% prefill);
- **device argmax** for greedy decode (4-byte readback), **paired qA+kvA
  matvec**, **device indexer top-k** (contexts beyond 2,048);
- **mlock of resident weights** (`DS4_GLM_MLOCK`): head 433 → 39 ms/token;
- **parallel prefill reads** (`DS4_GLM_READ_SPLIT`, prefill only — measured
  counterproductive in decode, where the serial fill pace is what leaves
  SSD bandwidth to the demand expert reads);
- **Q4 layer sidecar** as a single pack (`<gguf>.q4dense`, sections per
  layer, resumable, any subset useful) plus the expert bundle
  (`<gguf>.expbundle`) in the same container format.

Measured on the published IQ2_XXS with a 58/75-layer sidecar: **prefill
~1.05 s/token, decode ~3.0 s/token (0.33 tok/s)** sustained over 64 tokens —
from a 6.0 s/token baseline at the start of the optimization campaign. The
remaining decode profile is ~69% SSD I/O: the next levers are completing the
sidecar (disk-bound) and self-speculative decode.

Rejected by measurement (kept as opt-in knobs, verdicts in the code
comments): expert speculation on a saturated SSD, F_NOCACHE reads, MetalIO
for decode layer streaming, usage-driven expert cache (GLM routing is almost
uniform: top experts cover ~15% of routes at a 2 GB budget, against
DeepSeek's 69% hit rate).

## Contract verified against the real GGUF

On July 17, 2026 the real header of the IQ2_XXS variant of the Antirez
snapshot `2638b3b878f5c6cc3ae7334b8dbea1275025f52e` was read:

- 66 metadata KVs and 1,809 tensor descriptors;
- `glm-dsa` architecture, 79 stored blocks and 78 autoregressive blocks;
- hidden 6,144, vocabulary 154,880, 64 heads, KV-LoRA 512 and RoPE tail 64;
- 256 routed experts, top-8, one shared expert and three dense layers;
- indexer 32×128, top-k 2,048 and full indexer on 21 layers;
- declared context of 1,048,576 tokens;
- `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = glm4`;
- BOS `154822 = [gMASK]`, `<sop> = 154824`, EOS
  `154820 = <|endoftext|>`.

The optional test `GLM52RealHeaderIntegrationTests` uses a sparse copy of the
original size: it validates configuration, vocabulary and all descriptors
without reading the weight payload. The ordinary tests use synthetic fixtures
and do not require the GGUF.

## DSA cache and memory

The compact F16 layout keeps per token:

- 78 × 512 KV-LoRA values;
- 78 × 64 RoPE values;
- 21 × 128 indexer keys.

The total is 95,232 bytes/token, roughly 372 MiB at 4,096 tokens and 8.87 GiB
at 100,000 tokens. The DwarfStar planner grows in append-only slabs and can
enforce a resident budget: a large logical window therefore does not, by
itself, trigger the immediate allocation of the entire cache.

## Download manifest

| Variant | Filename | Exact size | SHA-256 |
|---|---|---:|---|
| IQ2_XXS RoutedIQ | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` | 211,075,856,448 bytes | `a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0` |
| Q2_K RoutedQ2K | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | 262,036,650,048 bytes | `b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5` |
| Q4_K RoutedQ4K | `GLM-5.2-UD-Q4_K_RoutedQ4K.gguf` | 434,170,886,208 bytes | `7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5` |

These are monolithic alternatives, not shards. The downloader writes to
`.part`, uses bounded buffers, supports resume and verifies the SHA in blocks
without loading the GGUF into RAM.

## What remains

1. self-speculative decode (cheap draft + `forwardBatch` verify — the only
   way past the per-token layer-stream I/O ceiling) and, later, MTP on the
   unexecuted nextn block (blk78);
2. prefill progress/cancellation and mid-prefill checkpointing (DeepSeek
   parity for long prompts);
3. full sidecar coverage (disk-bound: ~2.4 GB per remaining layer);
4. distribution.

The detailed map against the upstream branch is in
[`PORTING-ANTIREZ.md`](PORTING-ANTIREZ.md); GGUF names, shapes and types are
in [`CONTRATTO-GGUF.md`](CONTRATTO-GGUF.md).
