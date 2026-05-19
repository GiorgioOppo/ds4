# The converter

Offline transcoder that takes a HuggingFace DeepSeek-V4 release and
produces a directory the Swift runtime can load. Renames tensors,
fuses FP8/FP4 + scales into BF16/F16, or quantises Linear weights to
INT8/INT4/INT2 — depending on the target dtype.

Companion docs:

- [`DTYPES.md`](DTYPES.md) — bit layouts of FP4 / FP8 / E8M0 / BF16 /
  F16, the fusion math, and the conversion code excerpts.
- [`MODEL.md`](MODEL.md) — what the converter output gets loaded as.
- [`LOADING.md`](LOADING.md) — how the loader reads the shards.
- [`MODULES.md`](MODULES.md) — per-file index.
- [`USAGE.md`](USAGE.md) — the operator's reference (running the
  binary, flag table).

> 🇮🇹 La versione italiana è [`CONVERTER.it.md`](CONVERTER.it.md).

---

## 1. What the converter does

Three jobs:

1. **Rename** the HuggingFace tensor names into the canonical
   inference-side naming convention.
2. **Transcode** non-native dtypes into formats Apple Silicon's GPU
   actually has hardware for: FP8 + E8M0 scale → BF16, FP4 + E8M0
   scale → BF16, or fresh INT8/INT4/INT2 quantisation when the user
   wants to shrink the disk footprint.
3. **Re-shard** the output into layer-aligned safetensors files of a
   configurable size, plus a `model.safetensors.index.json` so other
   tooling can find each tensor.

Tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
`config.json`, `generation_config.json`, `special_tokens_map.json`)
are copied verbatim alongside.

The CLI is `Sources/converter/main.swift` (top-level
straight-line script — no class wrapper). The reusable pieces
(rename logic, dtype packers, fusion functions, ConversionSpec)
live in the `DeepSeekConverter` library
(`Sources/DeepSeekConverter/`).

### Why convert

Apple GPUs handle these natively in MSL:

- `F32`, `F16`, `BF16` (Metal 3+ / macOS 14+)
- Integer types (`i8`, `i16`, `i32`)
- `simdgroup_matrix<float|half|bfloat>` on M1+ / M3+

They do **not** have native types or arithmetic for:

- FP8-E4M3, FP4-E2M1, E8M0 (MX scale)

Inference on FP8/FP4 needs per-element dequant in shader on every
GEMM. Cheaper to pay the cost once at convert time and ship
native-dtype weights — at the price of bigger files (FP8 → BF16
doubles, FP4 → BF16 quadruples).

The INT8/INT4/INT2 paths go the opposite direction: pay extra
quantisation cost up front to *shrink* the disk footprint vs BF16,
trading a bit of accuracy.

---

## 2. Invocation

```
swift run -c release converter \
    --hf-ckpt-path /path/V4-Flash-HF \
    --save-path /path/V4-Flash-converted \
    --n-experts <N> \
    [--model-parallel 1] \
    [--target-dtype bf16|f16|int8|int4|int2|keep] \
    [--shard-size-gb 5]
```

Required flags:

| Flag | Meaning |
|---|---|
| `--hf-ckpt-path` | Source HuggingFace directory. Must contain `*.safetensors` shards + `model.safetensors.index.json`. |
| `--save-path` | Output directory. Created if missing; resume-safe (see §9). |
| `--n-experts` | Total expert count. Sanity-checked against the input; mismatch is a hard error. |

Optional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model-parallel` | 1 | Inherited from the Python reference; the Swift port is single-rank only, so this stays 1. |
| `--target-dtype` | `bf16` | Output dtype mode. See §3. |
| `--shard-size-gb` | 5 | Soft cap per output shard. Auto-capped to ~95% of `MTLDevice.maxBufferLength` so the runtime can mmap each shard as one MTLBuffer. |

### Target dtype matrix

| `--target-dtype` | Linear weights | Other tensors | Disk vs HF native | Inference speed |
|---|---|---|---|---|
| `keep` | FP8 / FP4 preserved | unchanged | smallest | slower (shader dequant) |
| `bf16` (default) | BF16 fused | BF16 | ~2× FP8 input, ~4× FP4 input | fastest (native simdgroup_matrix) |
| `f16` | F16 fused | F16 | same as bf16 | fastest |
| `int8` | INT8 W8A16 + F16 group scales | BF16 (fallthrough) | ~½ × BF16 | slightly slower (dequant in shader) |
| `int4` | INT4 W4A16 + F16 group scales, packed 2-per-byte | BF16 | ~¼ × BF16 | similar to int8 |
| `int2` | INT2 W2A16 + F16 group scales, packed 4-per-byte | BF16 | ~⅛ × BF16 | similar; brutal accuracy hit |

The `keep` mode is the "shortest path" — only `wo_a` gets fused
(since MLA reads it through `Einsum.bsgdGrd` directly without going
through a `Linear` dispatch, the einsum kernel expects a BF16-style
input). Everything else is relabelled at most.

---

## 3. Naming convention

`Sources/DeepSeekConverter/Rename.swift` — `renameKey(_:)`.

Three phases:

1. Strip the `model.` prefix.
2. Replace common phrases:
   - `self_attn → attn`
   - `mlp → ffn`
   - `weight_scale_inv → scale`
   - `e_score_correction_bias → bias`
3. Rewrite the leaf via `leafMapping[String:String]`:
   - `embed_tokens → embed`
   - `input_layernorm → attn_norm`
   - `post_attention_layernorm → ffn_norm`
   - `q_proj → wq`
   - `q_a_proj → wq_a`
   - `q_a_layernorm → q_norm`
   - `q_b_proj → wq_b`
   - `kv_a_proj_with_mqa → wkv_a`
   - `kv_a_layernorm → kv_norm`
   - `kv_b_proj → wkv_b`
   - `o_proj → wo`
   - `gate_proj → w1`, `down_proj → w2`, `up_proj → w3`
   - `lm_head → head`
   - Identity entries for already-canonical names so a re-conversion
     is a no-op.

There's also a leaf-detection guard: tensors whose parent is `hc`,
`attn_sink`, `tie2eid`, `ape`, or starts with `hc_` use the last
path component as the leaf instead of the second-to-last.

`shouldSkip(_:)` drops MTP-tied aliases: `mtp.*.emb*` and
`mtp.*.head.weight` exist in the HF checkpoint as references to the
main embed/head, so we discard them at convert time. The runtime's
`MTPBlock` holds weak references to the shared embed/head instead.

### Scale companions

`scaleNameFor(_:)` returns `<base>.scale` for `<base>.weight`. The
HF native release names them `<base>.weight_scale_inv` (note the `_inv`
suffix); the rename pass strips the `_inv` part. The runtime loader
accepts both forms via `WeightLoader.tryLoad`.

---

## 4. Indexing the input

`Sources/converter/main.swift:122` (walk + collect):

```swift
for inputURL in inputs {
    try autoreleasepool {
        let stf = try SafeTensorsFile(url: inputURL)
        for (origName, entry) in stf.entries {
            if shouldSkip(origName) { continue }
            let newName = renameKey(origName)
            let absOffset = try absoluteOffset(of: entry, in: inputURL)
            plan[newName] = PendingTensor(
                url: inputURL, offset: absOffset,
                byteCount: entry.dataOffsets[1] - entry.dataOffsets[0],
                dtype: entry.dtype, shape: entry.shape)
        }
    }
}
```

`autoreleasepool` matters: every `SafeTensorsFile` mmaps the whole
shard and wraps it as an MTLBuffer. Without draining the pool
between shards, those mappings stay alive for the entire indexing
phase and the process's virtual memory climbs by ~14 GB per
indexed shard.

The output is a flat `[String: PendingTensor]` map: new name → input
file + byte range + dtype + shape. No data has been read yet —
the file mapping is referenced for the parse pass and released.

All tensors land in the same map. An earlier refactor segregated
`wo_a.scale` into its own dict but never looked it up again, which
silently broke wo_a fusion in every mode (FP8 wo_a was passed through
verbatim instead of being fused to BF16 / INT8). Treating it like
any other scale lets the existing fusion paths find it via
`plan[scaleName]`.

---

## 5. Building write entries

For each tensor in the plan, the converter decides what to emit. The
output structure is `WriteEntry { name, dtype, shape, byteCount,
source: SafeTensorsWriter.Source }`. `Source` is one of:

- `.data(Data)` — pre-computed bytes in memory.
- `.file(url:offset:byteCount:)` — copy directly from another file.
- `.compute(byteCount:closure)` — lazy producer, called when the
  writer needs the bytes.

Use of `.compute` is essential for the fusion / quantisation paths:
the closure is called by the writer just before it streams that
tensor to disk, so peak memory stays bounded to `nCores × tensor
size`.

The decision tree (`Sources/converter/main.swift:236-622`):

```
For each tensor t in plan, in sorted name order:
  if t is a standalone .scale tensor and its parent .weight is in the plan:
    skip (its parent will consume it)
    
  if target == int8:
    if t is a Linear weight that should be quantised:
      emit two write entries:
        <name>.weight  (INT8 packed)
        <name>.scale   (F16 group scales)
      pick the source-dependent compute closure:
        FP8 in → quantizeFP8ToInt8
        FP4 in → quantizeFP4ToInt8
        BF16 in → quantizeBF16ToInt8
        F32 in → quantizeF32ToInt8
      continue
      
  if target == int4: …same shape, packed two-per-byte, INT4 ranges
  if target == int2: …same shape, packed four-per-byte, INT2 ranges

  effectiveTarget = bf16 if target ∈ {int8, int4, int2} else target

  if effectiveTarget != keep:
    if t is FP8 with a scale companion:
      emit <name> as BF16 (or F16), source = fuseFP8ToNative (compute)
      mark scale as consumed
      continue
    if t is FP4 (or packed I8/U8 in an experts dir) with a scale companion:
      emit <name> as BF16 (or F16), source = fuseFP4ToNative (compute)
      mark scale as consumed
      continue

  if effectiveTarget == keep:
    only fuse wo_a (BF16); relabel experts' I8/U8 as F4_E2M1
    everything else passes through .file(url:offset:byteCount:)
    continue

  default: pass-through .file
```

### Which Linear weights get quantised

`Sources/DeepSeekKit/Int8Quant.swift` exposes
`shouldQuantizeToInt8(name, lastDim:)`. The default whitelist (when
no `--int8-whitelist` is given):

- Every `layers.*.attn.*.weight` (MLA: wq_a, wq_b, wkv, wo_a, wo_b).
- Every `layers.*.ffn.experts.*.{w1,w2,w3}.weight` (routed experts).
- Every `layers.*.ffn.shared_experts.{w1,w2,w3}.weight`.
- Every `layers.*.attn.compressor.{wkv,wgate}.weight`.
- Every `layers.*.attn.indexer.{wq_b,weights_proj}.weight`.
- Every MTP layer's analogous tensors.

Excluded (always BF16): embed, head, RMSNorm gains, attn_sink, gate
weight (must stay F32 — see [`MODEL.md`](MODEL.md#48-moe-feed-forward)),
HC fn/base/scale, biases. The exclusions matter because INT8
quantisation drops the gate's logits below the precision the
sqrt(softplus) + topK gating tolerates; the same rationale applies
in `--target-dtype bf16` where `Linear.castOutputToBF16` is set to
`false` only on the gate and LM head.

`shouldQuantizeToInt4` and `shouldQuantizeToInt2` mirror this with
their own group-K constraints (must be divisible by `kInt4GroupK =
128` and `kInt2GroupK = 128` respectively).

### Quantisation kernels

`Sources/DeepSeekConverter/` packs the routines per source dtype:

| Function | What it does |
|---|---|
| `quantizeFP8ToInt8(weightURL:, scaleURL:, ...)` | Read FP8 bytes + per-(128,128) E8M0 scale, dequant inline (LUT), then symmetric RTN to INT8 with per-(row, K-128) F16 group scale. |
| `quantizeFP4ToInt8(...)` | Same shape but for FP4 packed two-per-byte + per-(row, K-32) E8M0 scale (the FP4 storage format). |
| `quantizeBF16ToInt8(...)` | Direct symmetric RTN from BF16. Used when the input is already a converted BF16 directory and you want to re-quantise. |
| `quantizeF32ToInt8(...)` | Same for F32 input. |
| `quantizeFP8ToInt4` / `quantizeFP4ToInt4` / `quantizeBF16ToInt4` / `quantizeInt8ToInt4` / `quantizeF32ToInt4` | INT4 variants. The `Int8ToInt4` path is useful for "re-quantise an existing INT8 directory" without going back to the HF native. |
| Same series for `…ToInt2` | INT2 variants, 4-per-byte packing. |
| `fuseFP8ToNative(..., target: TargetDType)` | FP8 + E8M0 → BF16 (or F16) per element. The hot loop uses precomputed LUTs (`e4m3LUT[256]`, `e8m0LUT[256]`); the per-block scale is hoisted out of the inner loop. |
| `fuseFP4ToNative(...)` | FP4 + E8M0 → BF16 (or F16). Per-row scales over `[1, 32]` K-blocks. |

All quantisation uses **symmetric RTN** (round-to-nearest): the
group max-absolute is computed, the scale is `max_abs / int_max`
(where `int_max` is 127 for INT8, 7 for INT4, 1 for INT2), values are
rounded and clamped to the range. Per-(row × K-128) scales for INT4
and INT8, per-(row × K-128) for INT2.

There is no calibration today (no AWQ / GPTQ / SmoothQuant). The
output is "fast and reasonable" up to INT4; INT2 takes a sharp
accuracy hit because the symmetric range `[-2, 1]` is too coarse for
many activation outliers. The CLI prints a warning for INT2 that
"calibration recommended for production".

### Lazy compute pairs

For `--target-dtype int8|int4|int2`, every Linear weight produces
*two* `WriteEntry` records (the packed weight + the F16 scale). Both
closures capture the same `var cached: (weight: Data, scale: Data)?`
by reference. The writer processes entries in declaration order:

1. The weight entry runs the quantisation closure → caches both
   results → returns the weight bytes.
2. The scale entry consumes the cached results → niles the cache out
   (releasing the weight buffer immediately) → returns the scale
   bytes.

So the converter never holds both the weight and scale
simultaneously after they've been written, which matters at 4-core
× 8 GB tensor peak.

---

## 6. Sharding the output

`Sources/converter/main.swift:638` (depth bucketing).

Strategy: each transformer layer goes into its own shard whenever
possible. If a layer is bigger than the shard cap, split it across
consecutive shards (the next layer always starts a fresh shard).
Top-level tensors (embed, head, norm, hc_head_*) go into shard 0.

```
depthKey(name):
  if name starts with "layers." → layer index from the second component
  if name starts with "mtp."    → 100_000 + index   (sort MTP after main)
  otherwise                     → -1                 (top-level)
```

Within each depth bucket, tensors are sorted by name for determinism.
The packer walks them in order and starts a new shard when adding
the next tensor would push the running total past `shard-size-gb`.

### Why layer-aligned

The forward pass touches layers in strict depth order. With layer-
aligned shards, one forward pass reads shard 0, then 1, then 2, …,
never going backwards. The OS page cache prefetches the next shard
while the GPU is reading the current one, and evicts already-
consumed shards under memory pressure without disturbing the layer
being actively read. The streaming-pool loader builds on this:
each per-layer shard is permanently assigned to a rotating slot,
and the pool kicks `pread`s ahead of the layer it's about to need.

### Shard size cap

`--shard-size-gb 5` is the default. The CLI also caps this to ~95%
of the device's `maxBufferLength` (MTLDevice.maxBufferLength): the
runtime mmaps each shard as one MTLBuffer, so a shard larger than
this is unloadable on that machine. The 95% margin leaves room for
the safetensors header + page alignment rounding.

### Output filenames

`expectedFilename(i:total:)` returns
`model-<i+1 zero-padded 5>-of-<total zero-padded 5>.safetensors`.
Together with `model.safetensors.index.json` this matches the
HuggingFace format other tooling expects.

---

## 7. Writing each shard

`SafeTensorsWriter` (`Sources/DeepSeekKit/SafeTensorsWriter.swift`)
is the streaming writer. For each tensor:

- Builds the safetensors JSON header lazily as `add(name:dtype:shape:source:)`
  is called.
- When `write(to: URL)` is invoked, writes the header (after padding
  to 8 bytes), then streams each tensor's payload from its `Source`:
  - `.data(Data)` — write directly.
  - `.file(...)` — open the source, copy in 64 MB chunks.
  - `.compute(byteCount:closure)` — call the closure, write the
    returned `Data`. The closure is responsible for memory
    management (the lazy-compute pair above ensures the cached
    payload is released after the second consumer reads it).

`autoreleasepool` wraps each shard write so transient allocations
from one shard don't survive into the next.

---

## 8. `model.safetensors.index.json`

After all shards are written, the converter assembles:

```json
{
  "metadata": { "total_size": <bytes> },
  "weight_map": {
    "embed.weight": "model-00001-of-00046.safetensors",
    "norm.weight": "model-00001-of-00046.safetensors",
    "layers.0.attn.wq_a.weight": "model-00001-of-00046.safetensors",
    "layers.0.attn.wq_a.scale": "model-00001-of-00046.safetensors",
    ...
  }
}
```

The Swift loader doesn't strictly need this (it walks `*.safetensors`
files directly), but other HF tooling does. Same format as the input
HF directory's index.

---

## 9. Resume

Conversion of V4-Flash can take 30+ minutes on a 16-core Mac. The
converter is resume-safe:

`Sources/converter/main.swift:735`:

```swift
var resumeFromShard = 0
for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    let url = saveDir.appendingPathComponent(fileName)
    guard let attrs = try? fm.attributesOfItem(atPath: url.path),
          let size = attrs[.size] as? Int,
          size >= shard.totalBytes else { break }
    for e in shard.entries { weightMap[e.name] = fileName }
    resumeFromShard = i + 1
}
```

A shard is "complete" iff:
- Its filename matches `model-(i+1)-of-(total).safetensors` exactly
  (so a partial write under a different `total` is *not* skipped).
- Its size is at least `shard.totalBytes` (the header adds a few KB
  of JSON, so `>=` means full write).

The `weightMap` for skipped shards is populated so the final
`index.json` reflects them correctly.

Sharding is deterministic given the same input + same flags
(deterministic name → depth bucket assignment + deterministic sort
within each bucket), so re-running with identical args produces
identical files at identical positions. Changing `--shard-size-gb`
or `--target-dtype` invalidates the resume — manually rm the output
dir first.

---

## 10. Concurrent fusion

Each tensor's compute closure is sequential, but the input shards
are read in parallel. The bottleneck during fusion is the per-tensor
SIMD loop (Float ops on every element); on a 16-core M3 Max, fusion
saturates around 4 concurrent threads before SSD contention starts
dominating.

Peak resident memory during fusion ≈ `nThreads × maxTensorSize`. For
V4-Flash that's a few GB.

---

## 11. Quantisation kernels (host-side)

The INT8 / INT4 / INT2 quantisation kernels live in
`Sources/DeepSeekKit/{Int8Quant,Int4Quant,Int2Quant,CalibratedQuant}.swift`
plus `Sources/DeepSeekConverter/DTypePacking.swift`. Hot loops:

### Symmetric RTN to INT8

```
for each row r in [0, outDim):
    for each K-block bk in [0, inDim/128):
        block_start = bk * 128
        // Find max-abs in the K-block.
        maxAbs = 0
        for k in block_start..<block_start+128:
            maxAbs = max(maxAbs, abs(w[r, k]))
        scale = maxAbs / 127        // f16
        // Round and clamp.
        for k in block_start..<block_start+128:
            q = round(w[r, k] / scale)
            q = clamp(q, -127, 127)  // -128 reserved for the symmetric "no-value"
            packed_weight[r, k] = q
        weight_scale[r, bk] = scale.bf16ToF16()
```

The `clamp(-127, 127)` is symmetric (leaving -128 unused) so the GEMM
kernel can use unsigned arithmetic for the `|x|` value range.

INT4 packs nibbles two-per-byte: low nibble = column `2k`, high nibble
= column `2k+1`. INT2 packs four 2-bit values per byte: `[7:6]` = col
`4k+3`, `[5:4]` = col `4k+2`, `[3:2]` = col `4k+1`, `[1:0]` = col `4k`.

### LUT-driven FP8 / FP4 dequant

`fuseFP8ToNative` and `fuseFP4ToNative` use precomputed LUTs to
avoid per-element bit twiddling:

```swift
let e4m3LUT: [Float] = (0..<256).map { dequantE4M3(UInt8($0)) }
let e2m1LUT: [Float] = (0..<16).map  { dequantE2M1(UInt8($0)) }
let e8m0LUT: [Float] = (0..<256).map { dequantE8M0(UInt8($0)) }
```

Each LUT fits in L1. The hot loop becomes "load FP8 byte → LUT
lookup → multiply by scale → write BF16". The per-(block_o, block_i)
scale is hoisted out of the inner loop so it's amortised across 128²
weights.

See [`DTYPES.md`](DTYPES.md) for the encoding details.

---

## 12. Source map

| Topic | File |
|---|---|
| CLI entry point | `Sources/converter/main.swift` |
| Rename + leaf mapping | `Sources/DeepSeekConverter/Rename.swift` |
| Dtype packing helpers (BF16 / F16 conversions) | `Sources/DeepSeekConverter/DTypePacking.swift` |
| FP8 / FP4 → BF16/F16 fusion | `Sources/DeepSeekConverter/NativeFusion.swift` |
| INT8 quant + whitelist | `Sources/DeepSeekKit/Int8Quant.swift` |
| INT4 quant | `Sources/DeepSeekKit/Int4Quant.swift` |
| INT2 quant | `Sources/DeepSeekKit/Int2Quant.swift` |
| Calibration helpers (stub today) | `Sources/DeepSeekKit/CalibratedQuant.swift` |
| ConversionSpec / ConversionTarget | `Sources/DeepSeekConverter/ConversionSpec.swift` |
| Streaming safetensors writer | `Sources/DeepSeekKit/SafeTensorsWriter.swift` |
| Safetensors reader (input side) | `Sources/DeepSeekKit/SafeTensors.swift` |
| ConvertSheet (GUI driver) | `Sources/DeepSeekUI/Views/Convert/ConvertSheet.swift` |
| Python reference | `Reference/inference/convert.py` |

---

## 13. The GUI variant

The desktop app exposes the converter from the toolbar's **Convert**
menu (wand icon) and Settings:

- `Sources/DeepSeekUI/Views/Convert/ConvertSheet.swift` — the SwiftUI
  sheet.
- `Sources/DeepSeekUI/State/ConvertViewModel.swift` — driver:
  assembles argv, spawns the converter binary, streams stdout/stderr
  back into the sheet's log pane.

The binary that the GUI runs is the same `swift run -c release converter`
output; the GUI just provides a UI for picking the input / output
directories and the flags. No code is shared between the SwiftUI side
and the converter — the GUI is a wrapper around the CLI.

---

## 14. Limitations and deferred work

Tracked in `TODO.md` (§0 Quantizzazione + §1 Parità). At a glance:

- **`cast_e2m1fn_to_e4m3fn`** in the Python reference
  (`Reference/inference/convert.py:17-52`) is NOT ported. The
  `--target-dtype keep` path with `--expert-dtype fp8` falls back to
  relabel-only. With the default `bf16` fusion path this is moot —
  the experts get fused to BF16 anyway.
- **Calibration (AWQ / GPTQ / SmoothQuant)** is not implemented. The
  INT4 and INT2 paths use symmetric RTN, which is the simplest
  scheme; calibration would tighten the per-row scales and recover
  some accuracy. `Sources/DeepSeekKit/CalibratedQuant.swift` is the
  scaffold.
- **W8A8 conversion mode** (separate from the runtime's
  `useW8A8Activations` Linear flag): the converter doesn't emit a
  W8A8-specific layout because the runtime can take any W8A16 layout
  and opt into W8A8 at dispatch time via `Linear.useW8A8Activations`.
- **Per-layer dtype mixing** (e.g. INT4 on experts, BF16 on attention)
  is not a CLI option but would be useful for accuracy-sensitive
  layers. Add a per-pattern allowlist to `shouldQuantizeToInt4` if
  the use case shows up.
- **Multi-GPU sharding** (the `--model-parallel` flag from the
  Python convert.py) is not implemented; the Swift port is
  single-rank.

---

## 15. End-to-end example

```bash
# 1. Download the HF release (one-time, ~142 GB).
pip install --upgrade huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir ~/Downloads/V4-Flash-HF

# 2. Convert to BF16 (default). Takes ~30 min on M3 Max.
swift run -c release converter \
    --hf-ckpt-path ~/Downloads/V4-Flash-HF \
    --save-path ~/Downloads/V4-Flash-BF16 \
    --n-experts 256

# 3. The runtime now reads the converted directory directly.
swift run -c release deepseek ~/Downloads/V4-Flash-BF16 \
    "Explain nuclear fusion in two sentences." \
    --mode chat --temperature 0.7 --max-tokens 256

# Disk footprint: ~600 GB (BF16). Quartered if you instead pass
# --target-dtype int4. The runtime auto-detects the dtype from each
# tensor's safetensors header — no loader-side flag is needed.
```
