**English** | [Italiano](README.it.md)

# GLM 5.2

This folder describes the native port of GLM 5.2 (`glm-dsa`) in DwarfStar.
The current state is no longer just "download": detector, GGUF contract,
tokenizer and chat protocol are implemented and verified. The decoder however
remains intentionally non-runnable; GUI, demo, server and benchmark cannot yet
select GLM.

## Current state

| Capability | State |
|---|---|
| Catalog and GUI download | yes, three monolithic GGUFs with pinned revision/size/SHA |
| Recognition of `general.architecture = glm-dsa` | yes |
| Frontend/tokenizer selection | yes |
| Configuration and GGUF schema | yes, strict geometry and 1,809 tensors |
| Weight map and top-8 read plan | yes, payload-free and quant-block-aware |
| Payload reads from the GGUF | yes, bounded `pread` over descriptors and top-8 plans (gate\|up\|down records) |
| GPT-2 tokenizer + `glm4` pretokenizer | yes |
| Native chat template, reasoning and XML tools | yes |
| CPU oracle for router, DSA/IndexShare and compact cache | yes |
| GLM Metal kernels | all phases as validated primitives; GPU composition of the first-token layer and of the decode step against the oracles |
| End-to-end prefill, decode and logits output | no |
| GUI or `DS4Demo` selection | no |
| Server, benchmark, KV checkpoints and distribution | no |

The catalog entries remain `downloadOnly`. A completed file can be inspected
and tokenized, but `BackendSelector` rejects it with a "backend not yet
implemented" error; there is no fallback to the DeepSeek V4 decoder.

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
itself, trigger the immediate allocation of the entire cache. The runtime will
still have to decide how to handle contexts that exceed RAM and SSD budget
before declaring the one-million-token context supported.

## Download manifest

| Variant | Filename | Exact size | SHA-256 |
|---|---|---:|---|
| IQ2_XXS RoutedIQ | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` | 211,075,856,448 bytes | `a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0` |
| Q2_K RoutedQ2K | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | 262,036,650,048 bytes | `b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5` |
| Q4_K RoutedQ4K | `GLM-5.2-UD-Q4_K_RoutedQ4K.gguf` | 434,170,886,208 bytes | `7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5` |

These are monolithic alternatives, not shards. The downloader writes to
`.part`, uses bounded buffers, supports resume and verifies the SHA in blocks
without loading the GGUF into RAM.

## Runtime progress

The enablement sequence is binding:

1. wire the validated weight map to the top-8 SSD/MetalIO reads and to the
   caches — started: `GLM52PayloadReader` executes descriptors and top-8 plans
   with bounded `pread` (double bounds check, rejection of truncated GGUFs at
   open) and `GLM52ExpertSlotCache` provides the per-expert LRU cache with
   byte-identical hits and batch pinning; MetalIO and residency remain;
2. complete Q/KV-LoRA, RoPE, indexer, DSA attention and IndexShare
   — done at the validation level: the decode step is composed on GPU
   (`glm52DecodeLayer`) with the exact upstream wiring — cache stores BEFORE
   selection/attention (normed KV-LoRA prefix + raw K-RoPE tail), indexer key
   with centered LayerNorm and RoPE on the prefix, `visible = pos+1` with
   fill-range or score+top-k, verbatim IndexShare, and per-row rotation of the
   K tail at attention time — judged by the `GLM52DecodeCPUReference` oracle.
   The persistent graph is composed: resident quantized weights loaded once
   (attention, dense/shared FFN, output head), compact cache and resident
   indexer keys updated in place, activations chained on buffers (a single
   command buffer on the fill-range path), expert accumulation on GPU with
   per-token streaming of the selected records, and the multi-layer forward
   `glm52ResidentDecodeForward` with the real IndexShare policy on absolute
   indices. The real engine `GLM52ResidentModel` loads the validated weight
   map from the GGUF into resident buffers (embedding row per token, routed
   experts via slot cache), runs token-by-token prefill and greedy decode;
   the IQ2_XXS kernel (routed format of the published file) is validated
   against the reference dequant, so the entire 78-layer stack is loadable
   from the real GGUF. The parity gate is ready in opt-in form:
   `GLM52LogitsParityIntegrationTests` compares engine and CPU oracle on the
   real dequantized weights (truncated 4-layer stack: dense + first sparse
   with IndexShare and IQ2_XXS experts), exact selections and router and all
   154,880 logits within |delta| <= 0.05 + 1% with identical argmax; the
   extension to the full 78-layer stack remains;
3. complete dense layers, routed/shared MoE, residual RMS and output head;
4. compare embedding, every layer and logits against an independent oracle;
5. verify prefill and decode on real prompts, including chat and tool calls;
6. measure RAM, pressure, SSD and session deletion;
7. only then flip the catalog from `downloadOnly` and enable GUI/demo;
8. server, benchmark, KV checkpoints and distribution come after the local
   runtime.

The detailed map against the upstream branch is in
[`PORTING-ANTIREZ.md`](PORTING-ANTIREZ.md); GGUF names, shapes and types are
in [`CONTRATTO-GGUF.md`](CONTRATTO-GGUF.md).
