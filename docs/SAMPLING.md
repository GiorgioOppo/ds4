# The sampler

How the engine turns a `[1, vocab]` row of logits into the next token
id. Open this when you want to know what each sampling flag does, in
what order, and how to add a new one.

The companion docs cover the surrounding pieces:

- [`MODEL.md`](MODEL.md) — what produces the logits.
- [`MODULES.md`](MODULES.md) — per-file index of `Sources/`.
- [`USAGE.md`](USAGE.md) — CLI flags that map to `SamplingOptions`.
- [`EXAMPLES.md`](EXAMPLES.md) — recipes for common sampler configs.

> 🇮🇹 La versione italiana è [`SAMPLING.it.md`](SAMPLING.it.md).

---

## 1. What the sampler is

A pipeline of independent transforms over a vocab-size logits array,
ending with one token id. Reference: `Sources/DeepSeekKit/Sampling.swift:35`
(struct `SamplingOptions`) and `Sources/DeepSeekKit/Sampling.swift:108`
(`enum Sampler`).

Every stage has a "neutral" default that disables it. When *every*
stage is neutral and `temperature == 0`, the pipeline shortcuts to a
GPU-side argmax. Otherwise the whole logits buffer is read to host
once and the rest runs in Swift (with `Accelerate` SIMD for the hot
loops).

The Python reference implements only Gumbel-max + temperature
(`Reference/inference/generate.py:19-24`). Every other stage in this
file is added by the Swift port to match the de-facto LLM toolbox
(llama.cpp, vLLM, koboldcpp).

```
logits [1, vocab] f32 on GPU
        │
        ├─ allFiltersDisabled?  ── yes ──►  Sampler.argmax(logits) → id (early return)
        ▼
read to host as [Float]
        │
        ▼
1. temperature scaling          (vDSP_vsmul, vectorised)
        │
        ▼
2a. repetition penalty          (per-history-id)
2b. frequency + presence        (per-history-id with counts)
        │
        ▼
options.mirostatTau > 0?  ── yes ──►  mirostatV2Sample(...)  → id (terminal)
        │ no
        ▼
3. top-K filter                 (quickselect nth-largest)
4. min-P filter                 (vectorised softmax + threshold)
5. tail-free                    (second-derivative on sorted probs)
6. locally-typical              (sort by |surprise - entropy|)
7. top-P (nucleus)              (sorted cumulative)
        │
        ▼
8. Gumbel-max multinomial       (argmax(log p + g) with g ~ Gumbel(0,1))
        │
        ▼
next token id
```

`options.rngState` (an LCG seed) and `options.mirostatMu` (Mirostat's
running estimate) are mutated in place across calls — that's why
`Sampler.sample(_:history:options:)` takes `options` as `inout`.

---

## 2. `SamplingOptions`

```swift
public struct SamplingOptions {
    public var temperature: Float = 1.0
    public var topK: Int = 0                    // 0 = disabled
    public var topP: Float = 1.0                // 1.0 = disabled
    public var minP: Float = 0.0                // 0 = disabled
    public var tailFree: Float = 1.0            // 1 = disabled
    public var typical: Float = 1.0             // 1 = disabled
    public var repetitionPenalty: Float = 1.0   // 1.0 = disabled
    public var frequencyPenalty: Float = 0.0    // 0 = disabled
    public var presencePenalty: Float = 0.0     // 0 = disabled
    public var mirostatTau: Float = 0.0         // 0 = disabled (use steps 3-8 instead)
    public var mirostatEta: Float = 0.1
    public var mirostatMu: Float = 10.0         // running estimate, updated in place
    public var rngState: UInt64 = defaultSamplerSeed()
}
```

Every field is `var` so the caller can tune values between turns —
the GUI's Generation tab, the agent's sampling defaults, and the CLI
flags all share this struct.

### Default seed

`defaultSamplerSeed()` mixes wall-clock nanoseconds and the process
id through an LCG step. Two consecutive runs of the CLI with the same
prompt and `temperature > 0` therefore produce different streams. For
reproducibility, set `rngState` explicitly.

### Cheap shortcut

`allFiltersDisabled` is true iff every filter is at the neutral
default *and* `temperature == 0`. The sample call detects this and
short-circuits to `argmax(logits)` on GPU, skipping the read-to-host
entirely. The CLI's `--temperature 0` invocation lands here when the
other flags are also at defaults.

---

## 3. The pipeline, step by step

The stage order is fixed and matches what llama.cpp and koboldcpp do.
Reordering would change the math (e.g. top-P after min-P sees a
different distribution than top-P before).

### 3.1 Argmax (GPU-only shortcut)

`Sampler.argmax(_:)` — `Sources/DeepSeekKit/Sampling.swift:136`.

Two paths, selected by vocab size `V`:

- **Single threadgroup** (`V < 8192`): one `argmax_f32` dispatch with
  256 threads doing a tree reduction over the logits buffer. One
  command-buffer commit, one `UInt32` written to a shared buffer.
- **Multi-stage** (`V ≥ 8192`): stage 1 splits the vocab into tiles of
  `argmaxTileSize = 2048` and produces `M = ceil(V / tileSize)`
  partial `(val, idx)` pairs in private-storage buffers. Stage 2
  reduces the `M` partials into the final answer. Two encoder
  invocations, one commit. For V=129 280 (V4-Flash), M = 64
  threadgroups in parallel — saturates an Apple GPU at 10–40 cores
  and cuts argmax latency ~5–8× vs the single-threadgroup path.

The threshold (8192) and tile size (2048) are tuned empirically;
below 8192 the kernel-dispatch overhead exceeds the parallelism gain.

### 3.2 GPU-side temperature scaling

`Sampler.applyTemperature(_:_:)` exists but the full
`Sampler.sample(...)` pipeline doesn't use it — `T == 0` shortcuts to
argmax, and any other temperature is applied host-side via
`vDSP_vsmul` after the logits are read.

The helper is kept for callers that want to scale logits in-place
without then sampling (e.g. visualisers).

### 3.3 Temperature (host-side, vectorised)

```
arr[i] *= 1 / max(T, 1e-5)
```

Done via `vDSP_vsmul` (Accelerate). ~3–5× faster than a Swift
per-element loop at V=130 k.

Skipped when `T == 1.0`. The `1e-5` floor stops a `T == 0` request
from blowing up (in practice it's filtered earlier via
`allFiltersDisabled`).

### 3.4 Repetition penalty (HuggingFace-style)

For every token id seen in `history`:

```
if arr[id] >= 0: arr[id] /= rep_penalty
else:            arr[id] *= rep_penalty
```

Multiplicative — sign-dependent so the penalty consistently makes the
logit smaller in magnitude no matter the sign. `rep_penalty == 1.0`
disables.

### 3.5 Frequency + presence penalties (OpenAI-style)

Both are additive in logit space; both run in the same loop after
counting `counts[id]` for each id in `history`:

```
if counts[id] > 0:
    arr[id] -= freq_pen * counts[id]
    arr[id] -= pres_pen
```

`freq_pen == 0 && pres_pen == 0` disables (the whole branch is
skipped).

These compose with `repetitionPenalty` — they are NOT mutually
exclusive. Practical advice: use one *or* the other in a given
agent's preset.

### 3.6 Mirostat v2 (terminal)

`mirostatV2Sample(...)` — `Sampler.swift:457`. When
`options.mirostatTau > 0`, the entire `(top-K | min-P | tail-free |
typical | top-P | Gumbel-max)` chain is replaced by:

1. Softmax to get `probs[]` (numerically stable, via
   `softmaxDouble(...)`).
2. Sort tokens by descending probability.
3. Truncate: keep the prefix where surprise `-log(p_i) < μ`.
   (`options.mirostatMu` is the running estimate of acceptable
   surprise, initialised to `2τ` by convention.)
4. Gumbel-max sample over the kept set.
5. Update μ in place: `μ ← μ − η · (S_t − τ)`, where `S_t = -log(p_selected)`.
   Clamped to `≥ 0.01`.

The Mirostat update keeps the *average* surprise of generated tokens
close to the target `τ`, which translates to perplexity ≈ `2^τ`. A
typical τ value is 5.0 (target perplexity ≈ 32).

Reference: Basu et al. 2020, "Mirostat: A Neural Text Decoding
Algorithm that Directly Controls Perplexity".

### 3.7 Top-K filter

`nthLargest(arr, k: options.topK)` finds the K-th largest value via
quickselect (O(N) average). Everything strictly smaller becomes
`-Float.infinity` — which Gumbel-max then ignores, and which
softmax-based downstream steps treat as zero mass.

Skipped when `topK == 0` (the disabled default) or `topK >= V`.

### 3.8 Min-P filter (vectorised)

`applyMinP(...)` — keeps only tokens whose probability is at least
`minP × max_prob`. Implemented entirely with Accelerate primitives:

1. `vDSP_maxv` → max logit.
2. `vDSP_vsadd` → shift by `-m`.
3. `vDSP_vspdp` → promote Float to Double (softmax sum is computed in
   double for numerical stability).
4. `vvexp` → element-wise exponential.
5. `vDSP_sveD` → sum.
6. `vDSP_vsmulD` → divide by sum (normalised probs).
7. `vDSP_maxvD` → max prob.
8. Threshold = `pMax · minP`; entries with prob < threshold → `-inf`.

Typical `minP` values: 0.05 – 0.10 (the llama.cpp range).

### 3.9 Tail-free sampling

`applyTailFree(...)` — the "tail" is defined as the part of the sorted
probability curve where the second derivative `|p_{i+2} - 2p_{i+1} +
p_i|` has accumulated past `z` of its total. Everything past that cutoff
is masked.

In practice this preserves the "head" plus a short transition zone and
clips the long flat tail. Disabled at `z == 1`; useful values: 0.95.

### 3.10 Locally-typical sampling

`applyTypical(...)` — keeps the tokens whose surprise `-log(p_i)` is
closest to the distribution's entropy `H = -Σ p log p`, sorted by
`|s_i − H|`, until cumulative mass reaches `p`.

Disabled at `typical == 1`. Typical values: 0.95.

The intuition: a fair model should produce tokens whose information
content matches the distribution's entropy. Tokens at the tails (too
common = boring; too rare = noisy) are clipped.

### 3.11 Top-P (nucleus) filter

`applyTopP(...)` — sorts the softmaxed probabilities, walks the
cumulative sum, and masks every token below the cutoff probability
that first makes the cumulative mass reach `topP`.

Disabled at `topP == 1`.

### 3.12 Gumbel-max multinomial

`Sampler.sample(...)` lines 308-326. After every filter has run, the
sampler picks the token whose `log(p_i) + g_i` is largest, where
`g_i ~ Gumbel(0, 1)`.

Gumbel-max is mathematically equivalent to a categorical sample from
`softmax(logits)`, but it avoids constructing the explicit
distribution: no normalisation, no CDF, just one `argmax` after
adding noise. Reference: `Reference/inference/generate.py:19-24`.

Per-step computation (in the engine):

```
for i in 0..<V:
    if arr[i] == -inf: continue          # masked out by earlier filters
    u ~ Uniform(0, 1)
    g = -log(-log(u))                    # Gumbel(0,1)
    key = (arr[i] - max_logit) + g
    track argmax(key)
```

The constant `max_logit` shift doesn't change the argmax — it's just
to keep the arithmetic away from FP32 overflow. The defensive
fallback (`if !anyFinite`) catches the rare combo where every token
got masked: it returns the GPU argmax of the *original* logits.

### 3.13 RNG

`nextUnit(_:)` is an inline LCG step (`state = state * 6364136223846793005
+ 1442695040888963407`, then take the high 53 bits). One state per
`SamplingOptions`; the caller is responsible for keeping the same
`SamplingOptions` instance across decode steps if they want a
single stream.

---

## 4. Per-instance state

Two fields mutate across calls:

- `rngState` — the LCG seed, advanced by `nextUnit` on every sampled
  token. Updated before `Sampler.sample` returns.
- `mirostatMu` — Mirostat's running surprise estimate. Updated only
  inside `mirostatV2Sample` after the Gumbel-max pick.

Everything else (`temperature`, `topK`, …) is "settings": you can
tweak them between turns without breaking continuity.

The chat surface holds one `SamplingOptions` per conversation; the
CLI builds one from command-line flags at startup and reuses it for
the whole decode loop. Reproducibility is therefore "fix `rngState`
to a constant; keep `temperature`, `topK`, etc. constant; reset
`mirostatMu` to `2 * mirostatTau` if Mirostat is on".

---

## 5. Recommended settings

These are the values the README and the Settings UI suggest by
default:

| Use case | T | top-K | top-P | min-P | rep_pen |
|---|---|---|---|---|---|
| Short Q&A | 0.7 | 0 | 0.9 | 0 | 1.0 |
| Coding | 0.6 | 0 | 0.85 | 0 | 1.0 |
| Long-form / creative | 0.85 | 0 | 0.95 | 0.05 | 1.05 |
| Determinism (CI) | 0 (argmax) | 0 | 1 | 0 | 1 |

**Critical gotcha for DeepSeek-V4-Flash**: at `--temperature 0` the
MoE router falls into self-reinforcing loops (`好的好的好的…`). The
GUI clamps the temperature slider to `[0.5, 1.0]` for that reason;
the CLI accepts 0 but the README recommends 0.7. See the README's
Recommended values section.

**Mirostat vs static filters**: don't mix. Mirostat's path doesn't
respect top-K / top-P / min-P / tail-free / typical — turning it on
disables that whole branch. If you set `mirostatTau > 0` together
with `topK == 50`, the topK value is silently ignored.

---

## 6. Performance

The whole sampler runs in O(V) host-side after one CPU-GPU sync to
read the logits. Vectorisation via Accelerate is the load-bearing
optimisation:

- **softmax (Double precision)**: `vDSP_maxv` + `vDSP_vsadd` +
  `vDSP_vspdp` + `vvexp` + `vDSP_sveD` + `vDSP_vsmulD`. ~3-5× faster
  than the equivalent Swift loop on V=130 k.
- **temperature**: `vDSP_vsmul` (~3-5× speedup).
- **min-P**: same softmax pipeline + a single threshold pass.

The remaining stages (`top-K nth-largest`, `tail-free` second
derivative, `typical` per-token surprise) are O(V log V) due to the
sort. For V=130 k that's still under a millisecond on Apple Silicon,
small compared to the model's forward pass.

The argmax shortcut path skips the read-to-host entirely — at greedy
sampling the only host work is a single 4-byte buffer load.

---

## 7. Mapping CLI flags

The local CLI exposes every sampling parameter as a flag. See
[`USAGE.md`](USAGE.md) for the canonical reference; quick map:

| Flag | Field |
|---|---|
| `--temperature T` | `temperature` |
| `--top-k K` | `topK` |
| `--top-p P` | `topP` |
| `--min-p P` | `minP` |
| `--tfs Z` | `tailFree` |
| `--typical P` | `typical` |
| `--repetition-penalty P` | `repetitionPenalty` |
| `--frequency-penalty P` | `frequencyPenalty` |
| `--presence-penalty P` | `presencePenalty` |
| `--mirostat τ` | `mirostatTau` |
| `--mirostat-eta η` | `mirostatEta` |

The remote OpenRouter path passes whatever subset OpenRouter accepts
in the JSON body (`temperature`, `top_p`, `frequency_penalty`,
`presence_penalty`); Mirostat / tail-free / typical / min-P are
local-only.

---

## 8. Adding a new sampler

The pipeline lives in `Sampler.sample(_:history:options:)`. New
filters slot in between an existing pair, after the temperature /
penalty block and before Gumbel-max.

Checklist:

1. Add a field to `SamplingOptions` with a *neutral default* that
   leaves the distribution unchanged. Update `allFiltersDisabled` so
   the new field's neutral value is required for the GPU-argmax
   shortcut.
2. Write the filter as a `private static func applyXxx(_ arr: inout
   [Float], vocabSize V: Int, ...)`. Match the in-place `arr[i] =
   -.infinity` masking convention so Gumbel-max ignores filtered
   tokens for free.
3. If the filter needs softmax probabilities, call `softmaxDouble`
   (already vectorised with Accelerate); don't re-implement.
4. Add the CLI flag in `Sources/deepseek/main.swift` and the GUI
   slider in `Sources/DeepSeekUI/Views/Settings/GenerationSettingsTab.swift`.
5. Add an XCTest in `Tests/DeepSeekKitTests/SamplerTests.swift` that
   verifies the disabled-default produces identical output to "no
   filter" and that an extreme value produces the expected
   degenerate behaviour (e.g. top-K=1 forces argmax).

---

## 9. Source map

| Topic | File |
|---|---|
| `SamplingOptions` struct | `Sources/DeepSeekKit/Sampling.swift:35` |
| `Sampler.argmax` (GPU multi-stage) | `Sources/DeepSeekKit/Sampling.swift:136` |
| `Sampler.applyTemperature` | `Sources/DeepSeekKit/Sampling.swift:204` |
| `Sampler.sample` (full pipeline) | `Sources/DeepSeekKit/Sampling.swift:224` |
| `applyMinP` / `applyTailFree` / `applyTypical` / `applyTopP` | same file, §"Filter helpers" |
| `mirostatV2Sample` | `Sources/DeepSeekKit/Sampling.swift:457` |
| `softmaxDouble` (vectorised) | `Sources/DeepSeekKit/Sampling.swift:509` |
| `nthLargest` / `nextUnit` | `Sources/DeepSeekKit/Sampling.swift:562` / `:583` |
| Metal kernels (`argmax_f32`, `apply_temperature`) | `Sources/DeepSeekKit/Kernels/sampling.metal` |
| Python reference (`sample` only) | `Reference/inference/generate.py:19-24` |
| CLI flag plumbing | `Sources/deepseek/main.swift` |
| GUI sliders | `Sources/DeepSeekUI/Views/Settings/GenerationSettingsTab.swift` |
| XCTests | `Tests/DeepSeekKitTests/SamplerTests.swift` |
