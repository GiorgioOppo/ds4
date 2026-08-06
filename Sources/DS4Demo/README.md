**English** | [Italiano](README.it.md)

# DS4Demo

`DS4Demo` is a CLI demo and diagnostic executable for the pure-Swift engine
(`DS4Core` + `DS4Metal`). It runs without the SwiftUI app and without external
engine links: no C engine, no static library, no subprocess inference path.

It is useful for:

- bringing up the Metal runtime and compiling embedded kernels;
- running a GPU self-test;
- auditing a GGUF's tensor dtypes and tokenizer special tokens;
- running a one-token forward smoke test;
- generating real tokens while measuring prefill, decode, and expert I/O.

[`Command/main.swift`](Command/main.swift) opens a GGUF with no-copy mmap,
detects MoE quantization, runs one forward pass, and when requested performs
layer-major prompt prefill followed by streaming decode.

The runnable DeepSeek profiles are Flash and Pro Q2 in one GGUF. The demo
derives layer count, dimensions, compression, RoPE and router geometry from the
validated profile. It does not assemble the two-file Pro Q4 package.

The demo uses `maxKeys = 4096` by default; `DS4_DEMO_CONTEXT` can select a
different configured capacity for controlled empty-context A/B measurements.
Sampling is greedy by default (`temperature = 0`), while the `DS4_DEMO_*`
variables below enable the same sampler filters used by the engine. It is
intended for engine measurement, not for chat UX; use the DwarfStar app for
normal conversation.

## Positional Arguments

```sh
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| # | Argument | Default | Meaning |
|---|---|---|---|
| none | *(none)* | none | Metal bring-up + GPU self-test only. No model is required. |
| 1 | `gguf-path` | none | Path to a `.gguf` file. Opens the model, prints detected MoE quantization, and runs one forward smoke test. |
| 2 | `maxNew` | `4` | Number of tokens to generate. `0` means smoke-test forward only, no streaming generation. |
| 3 | `prompt` | `"ciao come stai? rispondi in 1 parola"` | User text rendered through the model chat template. `@/path/file` uses the file's CONTENT as the prompt (long texts / prefill benchmarks, no shell quoting) — truncated to `DS4_PROMPT_MAX_CHARS` chars (default `12000`, ≈3k tokens); prompt plus output must fit `DS4_DEMO_CONTEXT`. |

Arguments are positional. To pass a prompt, you must also pass `maxNew`:

```sh
swift run DS4Demo model.gguf 32 "Explain RoPE"
```

The prompt is read from `CommandLine.arguments[3]`; quote it if it contains
spaces. There are no `--prompt` or sampling flags.

## Environment Variables

Advanced options are environment variables because the engine reads the same
knobs at runtime in the demo, the app, and tests.

How to read the knobs:

- **Diagnostics** add measurement and logging without changing the model.
- **Memory / I/O knobs** change how weights are read or kept resident; they
  matter most when the model does not fit in RAM and SSD I/O dominates decode.
- **Expert-cache knobs** keep hot MoE experts resident on GPU. They cost wired
  RAM but can reduce SSD reads when routing is concentrated.
- **Numerical / quality knobs** intentionally change computation or active expert
  count; use them for A/B tests, not parity runs.

Rule of thumb: first run with `DS4_DIAG=1`, then change one knob at a time and
compare token/s, `gather`, `route/attn`, expert-cache hit-rate, and steady-state
throughput.

### Demo-Specific Variables

| Variable | Values | Default | Effect |
|---|---|---|---|
| `DS4_DEMO_CONTEXT` | `1024...1000000` | `4096` | Configured `maxKeys` capacity used to construct the decoder. It changes capacity only: a short identical prompt keeps the live KV equally empty, making this the controlled knob for context-capacity A/B tests. |
| `DS4_DEMO_LIVE_CONTEXT` | `8...(DS4_DEMO_CONTEXT-maxNew-1)` | unset | Replaces the normal short prompt token stream with an exactly N-token synthetic chat context, tiled from the supplied prompt's ordinary text while preserving BOS/User/Assistant framing. Intended only for live-context performance frontiers; generated-text quality is not a semantic benchmark. |
| `DS4_PROMPT_FILE` | path | unset | Reads the prompt from a UTF-8 file. This is equivalent to the positional `@/path/file` form and avoids shell argument splitting for long prompts. |
| `DS4_PROMPT_MAX_CHARS` | positive integer | `12000` | Maximum number of characters read from a prompt file before truncation. The final tokenized prompt plus requested output must still fit `DS4_DEMO_CONTEXT`. |
| `DS4_TYPES_ONLY` | present, usually `=1` | off | GGUF audit mode. Prints critical tensor dtypes, tokenizer special ids, and prompt tokenization, then exits before constructing the decoder. Use it first when a model produces nonsense or when validating a new quantization. |
| `DS4_DIAG` | `=1` | off | Full streaming diagnostic run. Before generation it prints active knobs, measures disk bandwidth with `F_NOCACHE`, and checks whether MTP weights exist in the GGUF. After generation it prints per-layer routing, expert concentration, cache-slot allocation, and effective gather bandwidth versus measured SSD ceiling. |
| `DS4_DSPARK_GGUF` | path or `=1` | unset | Opens the checkpoint-matched DSpark support GGUF and enables the complete greedy speculative path: three no-copy transformers, private KV, Markov/confidence proposal and exact target verification. Requires temperature 0 and repetition penalty 1. |
| `DS4_DSPARK_EXACT_REPLAY` | `=1` | off | Replays even fully accepted proposals through ordinary one-token decode for bit-level A/B diagnostics. This intentionally removes the fast full-accept commit. |

`DS4Demo inspect-dspark <support.gguf>` prints metadata, shapes and quantization
without starting Metal; add `--bind` to validate the three no-copy bindings.
| `DS4_ACTIVE_EXPERTS` | `1...6` | `6` | Reduces how many routed MoE experts are actually used per token. This lowers I/O and gather time but changes quality because both supported DeepSeek profiles use top-6. Useful as a degraded low-RAM mode or to estimate expert-I/O cost. |
| `DS4_USAGE_FILE` | path or `off` | `<gguf-path>.usage.json` | JSON file for the usage imatrix, i.e. the historical expert choices made by the router. Keeping it enabled lets the next run pre-warm the cache with historically hot experts. Use a dedicated path for repeatable benchmarks; use `off` for cold runs. |
| `DS4_WARMUP` | integer `>=0` | `0`; with `DS4_DIAG`, `min(4, maxNew-1)` | Excludes the first N generated tokens from the decode profile. Early tokens often pay one-time costs such as cold cache, buffer wiring, and memory settling. |
| `DS4_AB_TRACE` | output prefix | unset | Writes `<prefix>.json` metadata and `<prefix>.f32` full-vocabulary logits for the A/B harness. Logits are retained copy-on-write and serialized only after the timed prefill/decode region. Intended for `scripts/metal_ab.sh`, not normal generation. |
| `DS4_AB_TRACE_FRAMES` | `1...64` | `9` | Maximum retained logit frames: one prefill result followed by decode-forward results. The hard cap bounds extra RAM to roughly 32 MiB on the Flash vocabulary even if `maxNew` is accidentally very large. |

### Engine Knobs

`DS4Demo` builds a `StreamingDecoder`, so it inherits the same runtime knobs used
by the app. Every row states whether the option only changes storage/scheduling,
can alter accumulation order, or is deliberately lossy. Do not infer numerical
equivalence merely from the fact that a knob is performance-oriented. The same
knobs are documented from the app's point of view in the root README's
[Configuration Reference](../../README.md#configuration-reference).

| Variable | Values | Default | Effect |
|---|---|---|---|
| `DS4_RAW_RING` | `=1` | off | Stores raw KV in an `nSWA` ring instead of the whole context. This is a shared-memory Metal buffer, not SSD-backed Disk KV. Sliding-window attention reads only the latest 128 rows, so raw-KV memory is constant; a wrapped window is reordered and converted F32→F16 by one GPU dispatch. It does not eliminate the compressed KV cache. |
| `DS4_KV_INITIAL` | token count | `512` | Initial physical frontier for the NSA attention/indexer caches. Their logical ceiling remains `DS4_DEMO_CONTEXT`, but the contiguous Metal buffers grow geometrically only as live KV crosses this frontier. This keeps an empty 1M-token session from binding ~13 GB of virtual compressor resources. Growth preserves emitted rows and does not change numerics. |
| `DS4_PREFILL_UNION` | integer | `192` | Maximum number of experts grouped together in layer-major prefill I/O. Each group re-reads its whole union from disk (with `DS4_EXPERT_PREAD` the page cache is bypassed), so prefill bytes/token scale with union ÷ tokens-per-group: at the old default `64` the gather read ~1.7 GB/token (~257 experts!) and saturated the SSD; `192` covers ~3× the tokens per read — measured best on M1 Pro. Costs ~1.3 GB packed × 2 (pipeline) of transient memory during prefill; lower it on tight-RAM machines. Never below `k` (6). |
| `DS4_PREFILL_FFN_BATCH` | `=0` disables | on | Batched prefill FFN: all of a group's token-FFNs are encoded into ONE Metal command buffer (one commit+wait per group) instead of one per token. On Flash, the old per-token path produced 43 layers × 512 tokens ≈ 22k GPU syncs per chunk; Pro has 61 layers. Serial encoder ⇒ identical dispatch order and numerics. Use `=0` only for A/B parity checks against the old path. |
| `DS4_PREFILL_CHUNK` | integer | `512` | Tokens per prefill chunk. Every chunk reloads ALL layers' dense weights (~6 GB with `DENSE_STREAM`), so a larger chunk amortizes that fixed cost over more tokens (`1024` halves it) at ~160 KB/token of extra transient activations. Long prompts only — a chunk never exceeds the prompt length. |
| `DS4_PREFILL_MM` | `=1` | off (OPT-IN) | Routed AND shared FFN of the batched prefill through matrix-matrix kernels: expert weights are read once per 64×32 tile for ALL the group's tokens instead of once per token (matvec), and the shared expert runs 3 matmuls per GROUP instead of 3 matvecs per token (Q8_0 shared weights; `DS4_SHARED_Q4` residents keep the per-token path). Same math, different accumulation order (simdgroup MMA, f16 mid) — outputs are close but NOT bit-identical to the matvec path, hence opt-in until A/B-validated. This optimized quant path is Flash-shape-only; keep it off for Pro Q2, which uses the supported matvec path. Groups with `DS4_ACTIVE_EXPERTS` < 6 or fewer than 8 tokens also fall back to matvec. Costs ~80 MB of extra transient staging per chunk on Flash. |
| `DS4_PREFILL_ROUTE_BATCH` | integer (`0`/`1` = off) | `32` | Batched prefill route phase: up to N consecutive tokens' route/attention are encoded into ONE command buffer — each token's scratch snapshot (FFN inputs + router selection) is blit-copied GPU-side before the next token overwrites it, and the CPU reads all the selections after a single wait. Cuts phase-A syncs N×. Serial encoder ⇒ identical numerics. The app's full benchmark tests 16/32/64/128 live. |
| `DS4_GPU_INDEXER_TOPK` | `0` disables | on | Long-context decode keeps indexer scores, exact heap-equivalent top-K mask, and attention in one command buffer. `0` restores the historical CPU selection for parity checks. |
| `DS4_ADAPTIVE_SPLITK` | `0` disables | on | Chooses exactly `min(32, max(1, ceil((rawRows + compressedRows)/32)))` FlashAttention workgroups. There is no power-of-two rounding: 128 rows use 4 workgroups and 129 use 5, avoiding the old 4→8 cliff. `0` restores the historical fixed depth of 32 for A/B. |
| `DS4_VECTOR_COPY` | `=1` enables | off | Experimental packed-four F32/F16 contiguous copies. Exact on M1 Pro, including scalar tails and F16 bit transport, but the order-balanced full-model A/B was decode-neutral/slightly negative; the generic copy remains the default. |
| `DS4_FLASH_KV_STAGE` | `=1` enables | off | Fuses raw-ring gather, compressed-cache F32→F16 staging and partial K/V/mask padding into one dispatch. Exact across wrapped/non-wrapped tests and 2,197,760 full-model logits. On M1 Pro it measured roughly neutral in decode and about +2% prefill, so it remains opt-in for other chips/context frontiers. |
| `DS4_ROPE_PAIR` | `=1` enables | off | In-place pair-only RoPE specialization; decode also reconstructs affine positions on the GPU unless `DS4_ROPE_AFFINE=0`. Baseline/pair/affine are bit-identical on tested normal/YaRN/inverse shapes. The M1 Pro end-to-end A/B showed no decode gain, therefore the generic kernel remains the measured default. |
| `DS4_ROPE_AFFINE` | `=0` disables affine positions | on when `DS4_ROPE_PAIR=1` | Keeps the pair kernel but restores the host-provided position array. It has no effect while `DS4_ROPE_PAIR` is off. |
| `DS4_DENSE_Q4_KERNEL` | `0` disables | on | Dedicated resident dense-Q4 matvec. It reuses the exact Q4_K row implementation but removes the synthetic single-expert ID wrapper. Set `0` for bit-parity/performance A/B. |
| `DS4_FUSED_ROUTER_PROBS` | `0` disables | on | Computes the router's `sqrt(softplus(logit))` in one vector dispatch instead of two scalar passes. Set `0` for parity A/B. |
| `DS4_FUSED_ROUTER_FINALIZE` | `0` disables | on | Combines top-6 selection and bit-identical route-weight normalization, saving one dispatch on every routed layer. |
| `DS4_FUSED_COMP_PROJ` | `0` disables | on | Computes compressor KV+gate projections together for F16 and Q8, sharing activation traffic and one dispatch while preserving each reduction order. |
| `DS4_DEMO_TEMPERATURE` | float | `0` | Demo sampling temperature. `0` keeps historical greedy decoding; `0.3` is a focused choice for the 2-bit model. |
| `DS4_DEMO_TOP_K` | integer | `0` | Candidate cap; use `40` with nonzero temperature to avoid the noisy 129k-vocabulary tail. |
| `DS4_DEMO_TOP_P` / `DS4_DEMO_MIN_P` | float | `1` / `0` | Nucleus and minimum-probability filters for the demo sampler. |
| `DS4_DEMO_REPEAT_PENALTY` | float | `1` | Values above 1 discourage loops; `1.1` matches the GUI default. |
| `DS4_DEMO_REPEAT_LAST_N` | integer | `64` | Recent-token window used by repetition penalty. |
| `DS4_EXPERT_CACHE_SLOTS` | integer | `0` (off) | Enables a per-layer LRU GPU cache for MoE experts. Each slot costs about 6.9 MB wired per layer on the 2-bit model. `8` is the effective minimum when enabled. If hit-rate rises, SSD gather drops; if RAM pressure rises, performance may worsen. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1` | off | Disables usage-driven slot redistribution. By default, layers with concentrated routing receive more slots at the same total budget. Use this for A/B comparisons. |
| `DS4_EXPERT_CACHE_HOT_EVICTION` | `=1` | off | Lossless A/B: evicts the worst historical popularity rank first instead of pure LRU, with recency as the tie-breaker. It protects the hot set from one-off scans but can regress after workload shifts, so it remains opt-in until measured against the same usage profile. |
| `DS4_MULTI_QUANT_CACHE` | `=1` | off | Gives mixed IQ2/Q4 expert layers correctly sized per-layer pools instead of bypassing the legacy single-size-class cache. Slot counts are allocated against the legacy cache's actual byte budget, so enabling it does not silently increase the planned cache RAM. Exact numerics; keep off/on as an A/B pair until validated on the target Mac. |
| `DS4_MTLIO` | `=1` | off | Apple Metal fast resource loading for expert slot-cache misses during decode. An interleaved expert is loaded with one contiguous command directly into its destination `MTLBuffer`; prefill remains on parallel `pread`. Any failure permanently falls back to `pread`. Same bytes and numerics; A/B on the target Mac because MetalIO is not consistently faster on M1 Pro. |
| `DS4_MTLIO_MIN_GBS` | float | `1.5` | Automatic circuit-breaker threshold. Timings are aggregated into 64 MiB windows; two consecutive slow windows switch the model load to `pread`. Small batches and isolated stalls are ignored. |
| `DS4_BUNDLE_DIR` | directory path | unset (sidecar next to the GGUF) | Where the `.expbundle` sidecar is BUILT when set (`<dir>/<gguf-name>.expbundle`); reading always tries the sibling `<gguf>.expbundle` first, then the directory. The sandboxed app sets it to its Application Support (it cannot write next to a picker-selected model); to share ONE copy between demo and app, point the demo at the same dir: `DS4_BUNDLE_DIR="$HOME/Library/Application Support/DwarfStar/expert-bundle"`. |
| `DS4_POOL_INTERLEAVE` | `=0` disables | on | Slot-cache pool layout: each slot holds the expert's gate\|up\|down slabs CONTIGUOUS (same record layout as the bundle sidecar), as three views of one buffer with the record size passed as the expert stride to the MoE kernels. A cache miss with the bundle becomes ONE ~7 MB pread straight into the slot (one syscall instead of three, larger I/Os at the same queue depth). Same bytes, same kernels ⇒ identical numerics. `=0` restores the historical 3-buffer layout (parity checks). |
| `DS4_EXPERT_PREAD` | `=1` | off | Reads expert slabs with `pread` + `F_NOCACHE` directly into destination buffers, bypassing the system page cache. This prevents expert churn from evicting dense weights and is often useful on 16 GB systems. Numerics are unchanged. |
| `DS4_PREAD_SPLIT` | `1...8` | `3` | With `DS4_EXPERT_PREAD=1`: number of CONCURRENT preads per expert slab in the slot-cache fill. Decode misses are few per layer (~2-3 × 3 slabs ⇒ NVMe queue depth ~6-9), but the disk only reaches its ceiling at ~24 requests in flight (the DIAG "random parallelo" probe): splitting each slab into N disjoint 16 KB-aligned ranges read in parallel raises the queue depth at identical bytes. Same bytes, same numerics. |
| `DS4_PREFETCH` | `0/1` | off | Uses a CPU queue and `madvise` to read ahead non-routed weights. M1 Pro A/B: 2.33→2.21 wall tok/s and 4.22→3.74 GB/s gather, so it remains opt-in. |
| `DS4_PREFETCH_EXPERTS` | integer | `0` | With `DS4_PREFETCH=1`, also prefetches N likely experts from the usage prior. This is speculative and can waste SSD bandwidth if routing is not predictable. |
| `DS4_EXPERT_LOOKAHEAD` | integer | `0` | Speculative slot-cache look-ahead: while layer *i* computes, PREFILLS layer *i+1*'s pool with its top-N usage-prior experts (real reads in the SSD-idle window, not page-cache hints — works with `DS4_EXPERT_PREAD`/bundle). The hash-routed layers (0-2) are always prefilled EXACTLY (their selection depends only on the token id). Try `6`..`12`; a wrong guess only wastes idle bandwidth. Requires the slot cache. |
| `DS4_WILLNEED_EXPERTS` | `=0` disables | on | Non-speculative read-ahead hint: after the router selects the real experts for a token, the engine calls `madvise(WILLNEED)` on exactly those slabs. Use `=0` to compare against pure on-demand faults. |
| `DS4_RESIDENT_DENSE` | `=1` | off | Copies about 5 GB of non-expert weights into wired buffers instead of relying on mmap/page-cache residency. Helps when expert streaming evicts dense weights; can hurt on 16 GB by increasing memory pressure. |
| `DS4_DENSE_AHEAD` | `1...3` | `1` | Read-ahead depth of the dense staging ring (requires `DS4_DENSE_STREAM=1`). `2` keeps layers i+1 AND i+2 in flight while the GPU computes layer i, so the SSD starts the next read instead of idling when a layer's read finishes early. Costs one extra staging slot (~150 MB) per step. A/B against the gather: deeper dense read-ahead also CONTENDS with the expert streaming on the same disk. |
| `DS4_DENSE_STREAM` | `=1` | off | Double-buffered dense-weight STREAMING: instead of trying to keep ~6 GB of dense weights resident, each layer's dense tensors are `pread`+`F_NOCACHE` into a 2-slot staging ring, kicked one layer AHEAD so the SSD read of layer i+1 overlaps the GPU compute of layer i (the dense access pattern is fully sequential — no speculation). ~300 MB of staging instead of ~6 GB resident, zero page-cache footprint. Takes precedence over `DS4_RESIDENT_DENSE`. Identical numerics. The candidate cure when `route/attn` dominates on 16 GB. |
| `DS4_LAZY_IDX` | `=0` disables | on (requires `DS4_DENSE_STREAM=1`) | Defers the NSA indexer SCORING projections (`indexer.attn_q_b` + `indexer.proj`) according to **live used keys**, not configured `maxKeys`. Before the sparse boundary they are absent from the dense stream; at first activation they are loaded once into resident buffers and reused. The ~360 MB/token saving and threshold-1024/top-K-512 examples are Flash measurements; Pro supplies top-K 1024 from its geometry. The recurrent indexer compressor always runs. LOSSLESS; `=0` restores always-stage behaviour for A/B. |
| `DS4_RESIDENT_COMP` | `=0` disables | on (requires `DS4_DENSE_STREAM=1`) | Keeps the four NSA compressor projections RESIDENT instead of streaming them. They are read every token on 41/43 Flash layers and all 61 Pro layers; ~0.6 GB is the Flash footprint. Same bytes, identical numerics. `=0` restores full streaming (tight-RAM fallback / A/B). |
| `DS4_COMP_Q8` | `=1` | off | **Lossy, experimental.** Converts the resident attention/indexer compressor projections from F16 to Q8_0, approximately halving their RAM and GPU traffic. The first run creates a `.q8comp.Lx-y` sidecar in `DS4_Q4_CACHE_DIR` (or beside the GGUF). |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | `64`, `128`, `256`, `512`, `1024`, `2048`, `4096` | `1024` | Number of compressed rows above which decode attention switches from the dense scan to the sparse indexer path — around the ~2K frontier the sparse score/top-k setup dominates the smaller attention scan. Changes only WHICH implementation consumes the compressed rows (the 512-row indexer selection is a separate, lower bound); same env override and allowed values as the C engine. It also determines the live boundary at which `DS4_LAZY_IDX` loads the resident scorer. |
| `DS4_MLOCK` | `=1` | off | `mlock()` best-effort on the hot resident buffers: expert-cache pools, resident output head (with `DS4_DENSE_STREAM`), dense staging ring and, with `DS4_RAW_RING=1`, raw-KV ring. Shared Metal buffers are anonymous memory that macOS COMPRESSES between uses — a buffer touched once per token re-reads at ~2.4 GB/s through the compressor instead of RAM speed (the measured ~235 ms output head on a "resident" copy). Pins ~3.3 GB at the default settings; identical numerics. A/B and watch `output head` and `experts` in the profile. |
| `DS4_DENSE_Q4` | `=1` (requires `DS4_DENSE_STREAM=1`) | off | **LOSSY.** Requantizes the three giant attention projections (`q_b`, `output_a`, `output_b` — Q8_0, 107 of ~145 MB/layer) to Q4_K at load and keeps them RESIDENT (~1.4 GB, locked with `DS4_MLOCK`): half their bytes, RAM-speed reads, ~4.6 GB/token removed from the SSD stream — the biggest byte-reduction available once the decode is disk-bound. Uses the already-validated Q4_K matvec kernels (dense matvec = MoE id-kernel with k=1; the grouped `output_a` = k=8 with per-group activation rows). Quality: logit drift ~0.02%, greedy outputs occasionally diverge but stay coherent. The FIRST load pays a parallel requant pass and writes a cache next to the model (`<gguf>.q4dense`, ~1.4 GB for this base trio); QKV/shared options add records and increase its size. Later loads reuse the validated cache, with load time dependent on SSD and memory pressure. Delete the file to force a re-requant. |
| `DS4_SHARED_Q4` | `=1` (requires `DS4_DENSE_Q4=1`) | off | **LOSSY.** Also requantizes the shared-expert FFN projections (gate/up/down, Q8_0) to Q4_K and keeps them resident: their slabs leave the per-token dense stream entirely, freeing disk bandwidth for the expert gather. Same `.q4dense` cache (toggling re-requants once). A/B quality before adopting. |
| `DS4_QKV_Q4` | `=1` (requires `DS4_DENSE_Q4=1`) | off | **LOSSY.** Also requantizes the remaining mid-size attention projections (`q_a`, `kv` — Q8_0, ~16 MB/layer) to Q4_K and keeps them resident: ~0.7 GB/token off the stream for ~0.35 GB of RAM, and the Q4 matvec reads half the bytes of the Q8 one. Same `.q4dense` cache — a cache built without this knob stays valid and only the new tensors are requantized (records match per key). A/B quality before adopting. |
| `DS4_SPEC_K` | `2`..`8` | off | **Greedy SELF-SPECULATIVE decode** ([design and measurements](../../docs/SELF-SPECULATIVE.md)): per round, N-1 candidates generated with a cheap draft and verified in one full-config batch pass. It only activates with temperature `0` and repetition penalty `<=1`; otherwise the demo disables it. The path stays opt-in because current measurements show no performance advantage. |
| `DS4_SPEC_DRAFT_EXPERTS` | `1`..`k-1` | `2` | Active experts of the speculative DRAFT (renormalized route weights, same mechanism as `DS4_ACTIVE_EXPERTS`): fewer experts = cheaper draft but lower acceptance. A/B per model. |
| `DS4_Q4_CACHE_DIR` | directory path | unset (cache next to the GGUF) | Where the `.q4dense` requant cache is WRITTEN when set (`<dir>/<gguf-name>.q4dense`). Reading tries both places: a cache produced by the demo next to the GGUF is picked up and PROMOTED into the primary location, so demo and app share one conversion. The sandboxed app sets it to its Application Support (it cannot write next to a picker-selected model). |
| `DS4_FUSED_MOE` | `=0` disables | on | Uses fused MoE kernels by default. `=0` selects the non-fused path for numerical A/B and debugging; it may change rounding and output. |
| `DS4_FUSED_HC` | `=0` disables | on | Fused HC-reduce tail: split+collapse+RMSNorm in ONE dispatch instead of three. It runs twice per layer, so the fusion saves ~170 dispatches/token. Same math; only the RMSNorm reduction order differs (±1 ulp class). `=0` restores the unfused path. |
| `DS4_ASYNC_FFN` | `=0` disables | on | Commits each layer's routed-FFN command buffer WITHOUT a CPU wait: the next layer's route commit+wait lands on the same in-order queue, so the GPU stays fed while the CPU encodes and the per-layer bubble (encode time × 43) disappears. Explicitly waited at end of token (before the output head) and on every error path; correctness is by queue order, numerics identical. `DS4_PROFILE_ROUTE` keeps the synchronous wait for accurate phase timing. `=0` for A/B. |
| `DS4_PROFILE_ROUTE` | `=1` | off | Splits `route/attn` into diagnostic subphases such as compressor, Q/KV projections, attention, and output. Adds synchronization overhead; inspect ratios, not absolute tok/s. |
| `DS4_Q8_NSG` | `1...8` | `4` | Simdgroups per threadgroup for dense Q8_0 matvecs. Scheduling and occupancy change, and the K reduction is partitioned differently, so the final floating-point bits are not guaranteed identical. Sweep `2/4/6/8` on the target Mac and include output validation when parity matters. |
| `DS4_MOE_NSG` | `1`..`8` | `4` | Simdgroups per threadgroup in routed MoE kernels. Row-partitioned: every value is bit-identical, only occupancy changes; the optimum depends on the GPU core count. Re-read at decoder creation. |
| `DS4_DENSE_Q4_NSG` | `1`..`8` | inherits `DS4_MOE_NSG` | Independent row occupancy for resident dense/grouped Q4_K projections. Bit-identical; lets `q`/`out` tune separately from routed experts. |

## What Should I Try First?

| Profile Symptom | First Knobs to Try | Reason |
|---|---|---|
| `expert gather` dominates and effective gather bandwidth is far below SSD ceiling | Compare default vs `DS4_WILLNEED_EXPERTS=0`, then tune `DS4_PREAD_SPLIT` around the default `3` | Separates read-ahead, page-cache churn, and direct-GGUF read concurrency. |
| `route/attn`, `embed`, or `head` stay slow after warm-up | `DS4_DENSE_STREAM=1`, `DS4_MLOCK=1`, then `DS4_DENSE_Q4=1` if lossy speed is acceptable | Dense weights are likely rereading from SSD or memory-compressed buffers. Streaming + pinning reduces churn; Q4 removes large Q8 projections from the hot path. |
| Expert-cache hit-rate is low but routing is concentrated | Sweep `DS4_EXPERT_CACHE_SLOTS=8/12/16/20/22`, with a fixed `DS4_USAGE_FILE` | More slots may retain hot experts and reduce SSD reads; the measured GUI preset uses 22 on an M1 Pro 16 GB with the full Q4 profile, but the optimum depends on free RAM. |
| Expert cache is enabled but not helping | `DS4_EXPERT_CACHE_UNIFORM=1` A/B | Tests whether usage-driven redistribution is helping or whether the budget is wrong. |
| Short runs are noisy | `DS4_WARMUP=4` and fixed `DS4_USAGE_FILE=<path>` | Separates cold costs from steady state and makes routing history repeatable. |
| You want best single-machine throughput | sweep `DS4_Q8_NSG=2/4/6/8` | Dense Q8 scheduling optimum is SoC- and memory-pressure-specific. |

## Examples

### 1. Metal bring-up only

```sh
swift run DS4Demo
# DS4Demo: Metal runtime up on Apple M1 Pro, N kernels compiled
# DS4Demo: GPU self-test PASSED
```

### 2. GGUF audit without decode

```sh
DS4_TYPES_ONLY=1 swift run DS4Demo /path/DeepSeek-V4-Flash-...-imatrix.gguf
#   TYPE blk.2.ffn_gate_exps.weight = iq2_xxs (code ...)
#   SPECIAL bos=... eos=... user=... assistant=...
#   PROMPT ids = [...]
```

### 3. Forward smoke test only

```sh
swift run DS4Demo /path/model.gguf 0
# DS4Demo: 1 forward in 3.2s - logits[...] finite=YES argmax=...
```

### 4. Real generation with a custom prompt

```sh
swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly."
# prefill logs, streamed answer, token timings, decode profile
```

### 5. Low-RAM experiment

```sh
DS4_EXPERT_PREAD=1 DS4_DENSE_STREAM=1 DS4_MLOCK=1 DS4_EXPERT_CACHE_SLOTS=22 \
  swift run DS4Demo /path/model.gguf 16
```

Add `DS4_DENSE_Q4=1` when you accept a lossy speed path, and
The direct-GGUF path uses `DS4_PREAD_SPLIT=3` by default.

### 6. Fused vs non-fused MoE comparison

```sh
swift run DS4Demo /path/model.gguf 8 "1+1?"
DS4_FUSED_MOE=0 swift run DS4Demo /path/model.gguf 8 "1+1?"
```

### 7. Full streaming diagnostic

Use at least 48 generated tokens when you want meaningful cache-allocation
diagnostics.

```sh
DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=8 \
  swift run DS4Demo /path/model.gguf 48 "Tell me the history of Rome."
```

Useful A/B comparisons with `DS4_DIAG=1`:

- default `DS4_WILLNEED_EXPERTS` vs `DS4_WILLNEED_EXPERTS=0`;
- default usage-driven allocation vs `DS4_EXPERT_CACHE_UNIFORM=1`;
- `DS4_EXPERT_CACHE_SLOTS=8/12/16/20/22`;
- `DS4_Q8_NSG=2/4/6/8`.

### 8. Correctness + performance A/B for one Metal knob

Use a prompt file so both child processes receive exactly the same bytes. The
harness overrides the chosen knob, forces greedy sampling, disables speculative
decode and prevents the baseline run from mutating usage history consumed by
the candidate:

```sh
scripts/metal_ab.sh /path/model.gguf prompt.txt DS4_ADAPTIVE_SPLITK 0 1 8
```

The report distinguishes bit-exact parity from tolerance-based numerical
parity, compares generated token ids and every retained logit, then reports
prefill and decode tok/s. Trace files are bounded and analyzed through `mmap`;
the comparator never loads all vectors into Python RAM. For an optimization
that promises exact numerics use:

```sh
DS4_AB_ATOL=0 DS4_AB_RTOL=0 \
  scripts/metal_ab.sh /path/model.gguf prompt.txt DS4_FUSED_ROUTER_FINALIZE 0 1 8
```

Repeat once with `DS4_AB_ORDER=candidate-first` before promoting a result: page
cache warmth, thermal state and memory pressure can favor either process.

### 9. Clean steady-state profile

```sh
DS4_WARMUP=4 DS4_USAGE_FILE=/tmp/ds4-demo-usage.json \
  swift run DS4Demo /path/model.gguf 32 "Write three sentences about Metal."

DS4_USAGE_FILE=off swift run DS4Demo /path/model.gguf 8 "Run without history."
```

## Output Streams

`stderr` receives engine logs: detected quantization, prefill timings, token
timings, diagnostic reports, and the final decode profile. Generated text goes
to `stdout` and is streamed unbuffered.

```sh
swift run DS4Demo /path/model.gguf 8 > answer.txt
```

The file receives only the generated answer; logs remain on screen.

## Source layout

- [`Command/README.md`](Command/README.md) documents the CLI entry point and its
  dependency boundary.
- [`Diagnostics/README.md`](Diagnostics/README.md) documents logging, GGUF
  inspection, and disk-measurement helpers.

Keep this README as the authoritative command and environment-variable
reference. When a knob or positional argument changes, update it in the same
change as the implementation.
