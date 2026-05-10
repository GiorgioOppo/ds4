# Numeric data types

Bit-level reference for the formats the project handles. Open this
before touching anything in `Sources/DeepSeekKit/Quantization.swift`,
`Sources/converter/main.swift`, or any quant-aware kernel.

## At a glance

| Type | Bits | Encoding | Range | Native on Apple Silicon? |
|---|---|---|---|---|
| F32 | 32 | IEEE 754 binary32 | ±3.4e38 | yes |
| F16 | 16 | IEEE 754 binary16 | ±65504 | yes |
| BF16 | 16 | F32 truncated top 16 bits | ±3.4e38 (F16 mantissa precision) | yes (Metal 3+) |
| FP8-E4M3 | 8 | 1 sign + 4 exp + 3 mantissa, bias 7 | ±448 | **no** |
| FP4-E2M1 | 4 | 1 sign + 2 exp + 1 mantissa, bias 1 | ±6 | **no** |
| E8M0 | 8 | unbiased exponent, value = `2^(b-127)` | `2^-127` to `2^128` | **no** |

The non-native types are unpacked in shader on every GEMM that consumes
them. The Swift converter can transcode them to BF16/F16 at convert
time to avoid this — see [USAGE.md](USAGE.md) and §3.

## 1. Bit layouts

### F32 (IEEE 754 binary32)

```
sign (1)  | exponent (8, bias 127)         | mantissa (23)
[31]      | [30..23]                       | [22..0]
```

Used as the accumulator type in every GEMM and as the working type
inside Layers (after dequant or directly when input is f32).

### F16 (IEEE 754 binary16)

```
sign (1)  | exponent (5, bias 15)  | mantissa (10)
[15]      | [14..10]               | [9..0]
```

Range ±65504. Subnormals near zero (2^-24). Smaller exponent range than
BF16 — easier to overflow.

### BF16 (Brain Float 16)

```
sign (1)  | exponent (8, bias 127)  | mantissa (7)
[15]      | [14..7]                 | [6..0]
```

Top 16 bits of an F32 (with rounding). Same exponent range as F32, less
mantissa precision (7 fractional bits vs F32's 23). Standard for
transformer training and the preferred 16-bit format on Apple Silicon.

### FP8-E4M3FN

```
sign (1)  | exponent (4, bias 7)  | mantissa (3)
[7]       | [6..3]                | [2..0]
```

- Normals: value = `(-1)^s * 2^(e-7) * (1 + m/8)`, for `e ∈ [1, 14]`.
- Subnormals: `e = 0, m > 0`, value = `(-1)^s * 2^-6 * (m/8)`.
- ±0: `e = 0, m = 0`.
- **NaN**: `e = 0xF, m = 0x7` only — no other NaN encodings. The "FN"
  suffix means "Finite or NaN" (no infinities).
- Max finite value: `e = 0xF, m = 0x6` = `1.75 * 2^8` = **448**.
- Min positive normal: `2^-6` ≈ 0.0156.
- Min positive subnormal: `2^-9` ≈ 0.00195.

Dequant logic (from `Sources/DeepSeekKit/Quantization.swift`):

```swift
func dequantE4M3(_ b: UInt8) -> Float {
    let sign = (UInt32(b) >> 7) & 1
    let exp  = (UInt32(b) >> 3) & 0xF
    let mant = UInt32(b) & 0x7
    if exp == 0 && mant == 0 { return sign == 1 ? -0.0 : 0.0 }
    if exp == 0xF && mant == 0x7 { return .nan }
    if exp == 0 { return (sign == 1 ? -1 : 1) * Float(mant) * 0x1p-9 }
    let bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return Float(bitPattern: bits)
}
```

### FP4-E2M1

```
sign (1)  | exponent (2, bias 1)  | mantissa (1)
[3]       | [2..1]                | [0]
```

Only 16 possible nibbles ⇒ 8 magnitudes. Full grid:

| Nibble | Bits | Value |
|---|---|---|
| 0 | `0 00 0` | +0 |
| 1 | `0 00 1` | +0.5 |
| 2 | `0 01 0` | +1 |
| 3 | `0 01 1` | +1.5 |
| 4 | `0 10 0` | +2 |
| 5 | `0 10 1` | +3 |
| 6 | `0 11 0` | +4 |
| 7 | `0 11 1` | +6 |
| 8…15 | sign=1 + same magnitudes | negatives |

Notice the irregular spacing: finer near zero (0.5 step), coarser at
top (2 step). This matches typical weight distributions in trained
networks.

Stored packed two-per-byte. Convention in this codebase:
- low nibble (`byte & 0xF`) is the element at K-index `2*i`
- high nibble (`byte >> 4`) is the element at K-index `2*i + 1`

Matches the PyTorch `float4_e2m1fn_x2` ordering.

Dequant logic (from `Sources/DeepSeekKit/Quantization.swift`):

```swift
func dequantE2M1(_ nibble: UInt8) -> Float {
    let s = (nibble >> 3) & 1
    let mag: Float
    switch nibble & 7 {
    case 0: mag = 0;   case 1: mag = 0.5
    case 2: mag = 1;   case 3: mag = 1.5
    case 4: mag = 2;   case 5: mag = 3
    case 6: mag = 4;   default: mag = 6
    }
    return s == 1 ? -mag : mag
}
```

### E8M0

```
exponent (8, unbiased)
[7..0]
```

Pure exponent: `value = 2^(b - 127)`. No sign bit (always positive).
`b = 0x00` → `2^-127` (smallest). `b = 0xFE` → `2^127` (largest finite).
`b = 0xFF` is reserved as NaN by the MX spec.

Used as a *block scale* — never a standalone tensor value. The fused
expression `x_E8M0 * y_quant` provides the dynamic range that the
quant value alone can't cover.

Dequant:

```swift
func dequantE8M0(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    return Float(bitPattern: UInt32(b) << 23)
}
```

## 2. Block layouts in safetensors

Each quantized weight tensor is paired with a `.scale` tensor of the
same base name in the safetensors header (e.g. `layers.0.attn.wkv.weight`
+ `layers.0.attn.wkv.scale`). The Swift converter consumes both at
once when transcoding.

### FP8 weight blocks

```
weight: [out_features, in_features]                    fp8_e4m3   bytes
scale:  [ceil(out_features/128), ceil(in_features/128)] e8m0       bytes
```

One scale per 128×128 block. Used for most non-expert linears in V4.

### FP4 weight blocks (experts)

```
weight: [out_features, in_features / 2]    fp4_e2m1     (packed)
scale:  [out_features, in_features / 32]   e8m0         bytes
```

One scale per 1×32 K-block. Each row has its own scale series — fine
granularity along the contracted dimension, none on output dim.

Used for the MoE expert weights in V4-Flash. V4-Pro has the same layout.

### Activation quant

Activations are quantized inline at runtime (when the FP8/FP4 GEMM path
is used). The activation quant produces 1×128 (FP8) or 1×32 (FP4)
groups along the K axis, with E8M0 scales.

Implementation: `Sources/DeepSeekKit/Layers/ActQuant.swift` (Swift
wrapper around `Kernels/act_quant.metal`).

## 3. Conversions

### F32 → BF16 (round-to-nearest-even)

```swift
@inline(__always)
func floatToBF16(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
    return UInt16(truncatingIfNeeded: rounded >> 16)
}
```

Implemented in `Sources/converter/main.swift:155`. Three-line truncate
with RTNE. Works because BF16 is exactly F32 with the bottom 16 bits
chopped — no exponent rebias needed.

### F32 → F16 (saturating + RTNE)

Full IEEE conversion: exponent rebias from 127 to 15, mantissa
truncation with rounding, saturating to ±inf on overflow, subnormal
flushing. ~25 lines of bit manipulation in
`Sources/converter/main.swift:162-193`.

### Dequant lookup tables (converter)

The converter precomputes 256-entry (FP8, E8M0) and 16-entry (FP4)
lookup tables on startup:

```swift
let e4m3LUT: [Float] = (0..<256).map { dequantE4M3(UInt8($0)) }
let e2m1LUT: [Float] = (0..<16).map  { dequantE2M1(UInt8($0)) }
let e8m0LUT: [Float] = (0..<256).map { dequantE8M0(UInt8($0)) }
```

Each LUT fits in L1 cache. The hot conversion loop becomes a load +
multiply instead of bit twiddling per element.

## 4. Fusion math (converter)

### FP8 + E8M0 → BF16

For each `(i, j)` in the logical `[out, in]` weight:

```
block_o, block_i = i / 128, j / 128
scale = dequantE8M0(scaleByte[block_o][block_i])
value = dequantE4M3(weightByte[i][j])
out_bf16[i][j] = floatToBF16(value * scale)
```

The per-128-block scale is hoisted out of the inner loop in the
parallel implementation — once per `(block_o, block_i)`, reused across
128² weights. See `fuseFP8ToNative` in
`Sources/converter/main.swift:260-310`.

### FP4 + E8M0 → BF16

For each `(i, j)` in the logical `[out, in]` (with packed weight
`[out, in/2]`):

```
half_idx = j / 2
high     = (j & 1) == 1
nib = (weightByte[i][half_idx] >> 4) if high else (weightByte[i][half_idx] & 0xF)
scale = dequantE8M0(scaleByte[i][j / 32])
value = dequantE2M1(nib)
out_bf16[i][j] = floatToBF16(value * scale)
```

Per-row scales (one row per `out` index), inner loop processes 16
packed bytes = 32 logical values per scale block. See `fuseFP4ToNative`
in `Sources/converter/main.swift:312-380`.

### `wo_a` special case

Even with `--target-dtype keep`, the converter always fuses `wo_a +
scale → BF16` because the MLA forward path uses
`Einsum.bsgdGrd(...)` directly on `wo_a.weight` (not a `Linear`
dispatch). That einsum expects F32-compatible weights, so wo_a must
land as BF16 either way.

See the `--target-dtype keep` branch in the converter for the special
handling.

## 5. What Apple Silicon supports natively

| Type | F32 ops | F16 ops | BF16 ops | INT4/INT8 ops |
|---|---|---|---|---|
| Add/mul/etc | yes | yes | yes (Metal 3+) | yes (i8/i16/i32, no i4) |
| `simdgroup_matrix` | yes (M1+) | yes (M1+) | yes (M3+) | no |

There is **no** native FP8 or FP4 arithmetic on any Apple GPU as of
M4. Every FP8/FP4 element has to be unpacked to F32 (or H, then
promoted) in shader before the math. INT4 also has no matmul
instructions on Apple, so reformatting FP4 → INT4 wouldn't change the
runtime cost.

### Conversion costs in shader

- FP8 → F32 in shader: 1 load + ~5 bit-ops + 1 reinterpret. Per
  element ~3 ns at 1 TFLOP equivalent.
- FP4 → F32 in shader: 1 load + 1 shift + 1 mask + 1 LUT lookup (in
  threadgroup memory or constant memory). Similar cost.
- BF16 → F32 in shader: implicit on `simdgroup_matrix<bfloat>`
  operations. Effectively free.

This is why converting to BF16 at convert time (paying the cost once)
typically beats keeping FP8/FP4 on disk for inference performance —
even though it 2× / 4× the disk footprint.

## 6. Sources of truth

- Runtime decoder helpers: `Sources/DeepSeekKit/Quantization.swift`
- Converter LUT + pack helpers: `Sources/converter/main.swift:155-260`
- Reference Python: `Reference/inference/kernel.py` (the
  `act_quant_kernel`, `fp4_quant_kernel`, `fp8_gemm_kernel`,
  `fp4_gemm_kernel` and the `cast_e2m1fn_to_e4m3fn` in
  `Reference/inference/convert.py:17-52`)
- MX spec for E8M0 / E4M3 / E2M1: see the OCP Microscaling Formats v1.0
  whitepaper (referenced by DeepSeek-V4 paper).

See [PYTHON-MAPPING.md](PYTHON-MAPPING.md) for the line-by-line
Swift ↔ Python correspondence.
