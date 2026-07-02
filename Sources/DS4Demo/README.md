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

`main.swift` opens a GGUF with no-copy mmap, detects MoE quantization, runs one
forward pass, and when requested performs layer-major prompt prefill followed by
greedy streaming decode.

The demo uses a fixed context window of `maxKeys = 4096`. Sampling is greedy
(`temperature = 0`). It is intended for engine measurement, not for chat UX; use
the DwarfStar app for normal conversation.

## Positional Arguments

```sh
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| # | Argument | Default | Meaning |
|---|---|---|---|
| none | *(none)* | none | Metal bring-up + GPU self-test only. No model is required. |
| 1 | `gguf-path` | none | Path to a `.gguf` file. Opens the model, prints detected MoE quantization, and runs one forward smoke test. |
| 2 | `maxNew` | `4` | Number of tokens to generate. `0` means smoke-test forward only, no streaming generation. |
| 3 | `prompt` | `"ciao come stai? rispondi in 1 parola"` | User text rendered through the model chat template. |

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
| `DS4_TYPES_ONLY` | present, usually `=1` | off | GGUF audit mode. Prints critical tensor dtypes, tokenizer special ids, and prompt tokenization, then exits before constructing the decoder. Use it first when a model produces nonsense or when validating a new quantization. |
| `DS4_DIAG` | `=1` | off | Full streaming diagnostic run. Before generation it prints active knobs, measures disk bandwidth with `F_NOCACHE`, and checks whether MTP weights exist in the GGUF. After generation it prints per-layer routing, expert concentration, cache-slot allocation, and effective gather bandwidth versus measured SSD ceiling. |
| `DS4_ACTIVE_EXPERTS` | `1...6` | `6` | Reduces how many routed MoE experts are actually used per token. This lowers I/O and gather time but changes quality because Flash is trained for 6 experts. Useful as a degraded low-RAM mode or to estimate expert-I/O cost. |
| `DS4_USAGE_FILE` | path or `off` | `<gguf-path>.usage.json` | JSON file for the usage imatrix, i.e. the historical expert choices made by the router. Keeping it enabled lets the next run pre-warm the cache with historically hot experts. Use a dedicated path for repeatable benchmarks; use `off` for cold runs. |
| `DS4_WARMUP` | integer `>=0` | `0`; with `DS4_DIAG`, `min(4, maxNew-1)` | Excludes the first N generated tokens from the decode profile. Early tokens often pay one-time costs such as cold cache, buffer wiring, and memory settling. |

### Engine Knobs

`DS4Demo` builds a `StreamingDecoder`, so it inherits the same runtime knobs used
by the app. Unless stated otherwise, they are performance or memory experiments
and should not change numerics. `DS4_ACTIVE_EXPERTS` and `DS4_FUSED_MOE=0` are
intentional exceptions.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `DS4_RAW_RING` | `=1` | off | Stores raw KV in an `nSWA` ring instead of the whole context. Sliding-window attention reads only the latest 128 rows, so this makes raw-KV memory constant. It does not eliminate every compressed KV cache. |
| `DS4_PREFILL_UNION` | integer | `64` | Maximum number of experts grouped together in layer-major prefill I/O. Higher values reduce I/O rounds but use more temporary memory; lower values reduce peak memory but may slow prefill. Never below `k` (6). |
| `DS4_EXPERT_CACHE_SLOTS` | integer | `0` (off) | Enables a per-layer LRU GPU cache for MoE experts. Each slot costs about 6.9 MB wired per layer on the 2-bit model. `8` is the effective minimum when enabled. If hit-rate rises, SSD gather drops; if RAM pressure rises, performance may worsen. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1` | off | Disables usage-driven slot redistribution. By default, layers with concentrated routing receive more slots at the same total budget. Use this for A/B comparisons. |
| `DS4_EXPERT_PREAD` | `=1` | off | Reads expert slabs with `pread` + `F_NOCACHE` directly into destination buffers, bypassing the system page cache. This prevents expert churn from evicting dense weights and is often useful on 16 GB systems. Numerics are unchanged. |
| `DS4_PREFETCH` | `=1` | off | Uses `madvise` to read ahead non-routed weights for the next layer while the current layer computes. It may help if compute overlaps I/O; it may hurt if the SSD is already saturated by expert gather. |
| `DS4_PREFETCH_EXPERTS` | integer | `0` | With `DS4_PREFETCH=1`, also prefetches N likely experts from the usage prior. This is speculative and can waste SSD bandwidth if routing is not predictable. |
| `DS4_WILLNEED_EXPERTS` | `=0` disables | on | Non-speculative read-ahead hint: after the router selects the real experts for a token, the engine calls `madvise(WILLNEED)` on exactly those slabs. Use `=0` to compare against pure on-demand faults. |
| `DS4_RESIDENT_DENSE` | `=1` | off | Copies about 5 GB of non-expert weights into wired buffers instead of relying on mmap/page-cache residency. Helps when expert streaming evicts dense weights; can hurt on 16 GB by increasing memory pressure. |
| `DS4_FUSED_MOE` | `=0` disables | on | Uses fused MoE kernels by default. `=0` selects the non-fused path for numerical A/B and debugging; it may change rounding and output. |
| `DS4_PROFILE_ROUTE` | `=1` | off | Splits `route/attn` into diagnostic subphases such as compressor, Q/KV projections, attention, and output. Adds synchronization overhead; inspect ratios, not absolute tok/s. |
| `DS4_Q8_NSG` | `1...8` | `4` | Simdgroups per threadgroup for dense Q8_0 matvecs. Results are identical; only scheduling, occupancy, and latency hiding change. Sweep `2/4/6/8` on the target Mac. |

## What Should I Try First?

| Profile Symptom | First Knobs to Try | Reason |
|---|---|---|
| `expert gather` dominates and effective gather bandwidth is far below SSD ceiling | Compare default vs `DS4_WILLNEED_EXPERTS=0`, then try `DS4_EXPERT_PREAD=1` | Determines whether read-ahead helps and whether bypassing page cache stabilizes decode. |
| `route/attn`, `embed`, or `head` stay slow after warm-up | `DS4_EXPERT_PREAD=1`, then `DS4_RESIDENT_DENSE=1` on RAM-rich systems | Likely dense-weight re-faults caused by expert streaming churn. |
| Expert-cache hit-rate is low but routing is concentrated | `DS4_EXPERT_CACHE_SLOTS=8/12/16` | More slots may retain hot experts and reduce SSD reads. |
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
DS4_ACTIVE_EXPERTS=4 DS4_RAW_RING=1 DS4_EXPERT_CACHE_SLOTS=8 \
  swift run DS4Demo /path/model.gguf 16
```

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
- `DS4_EXPERT_CACHE_SLOTS=8/12/16`;
- `DS4_Q8_NSG=2/4/6/8`.

### 8. Clean steady-state profile

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
