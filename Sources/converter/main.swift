import Foundation
import Metal
import DeepSeekKit
import DeepSeekConverter

// Native Swift port of `Reference/inference/convert.py`, extended to
// transcode every non-native dtype into a type Apple Silicon supports
// natively (BF16 by default).
//
// Apple Silicon GPUs handle these natively in MSL:
//   F32, F16, BF16 (Metal 3+ / macOS 14+), and integer types.
// They do NOT have native types or arithmetic for:
//   FP8-E4M3, FP4-E2M1, E8M0 (MX scale).
// Inference paths on those would need per-element dequant in shader.
// Cheaper to pay the cost once at convert time and ship native-dtype
// weights — at the price of bigger files (FP8 → BF16 doubles, FP4 →
// BF16 quadruples).
//
// Usage:
//   swift run -c release converter \
//       --hf-ckpt-path /path/V4-Flash-HF \
//       --save-path /path/V4-Flash-converted \
//       --n-experts <N> \
//       [--model-parallel 1] \
//       [--target-dtype bf16|f16|int8|int4|int2|keep]  # default: bf16
//       [--shard-size-gb 5]                   # default: 5 GB per shard
//       [--quant-method rtn|awq|smoothQuant|gptq]  # default: rtn
//       [--calib-stats <dir-or-stats.json>]   # required for non-RTN methods
//
// `int8` is a W8A16 weight-only quantization: every Linear weight that
// passes `shouldQuantizeToInt8` (DeepSeekKit/Int8Quant.swift) gets
// symmetric RTN quantization into INT8 with per-row per-group-128 F16
// scales. Other tensors (embed/head/norm/attn_sink/hc_*/bias) fall through
// to BF16 like in `--target-dtype bf16`.
//
// `--quant-method awq|smoothQuant|gptq` consults `--calib-stats` to
// pick per-tensor calibration. Per-tensor lookup key is the OUTPUT
// (renamed) tensor name minus the trailing `.weight`. Tensors with
// no stats entry silently fall back to RTN. The inverseChannelScale
// produced by AWQ / SmoothQuant is currently DISCARDED — the runtime
// loader doesn't read it yet, so emitting it would just bloat the
// output safetensors. Documented follow-up.

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: converter --hf-ckpt-path <dir> --save-path <dir> --n-experts <N>
                     [--model-parallel 1] [--target-dtype bf16|f16|int8|int4|int2|keep]
                     [--shard-size-gb 5]

    """.utf8))
    exit(2)
}

// `TargetDType` is now `ConversionTarget` and lives in the
// DeepSeekConverter library. Imported above.
typealias TargetDType = ConversionTarget

// ---------- Argument parsing ----------

var hfPath: String?
var savePath: String?
var nExperts: Int = 0
var mp: Int = 1
var targetDType: TargetDType = .bf16
var shardSizeGB: Double = 5.0
// Calibrated quantization (TODO §1). `--quant-method` defaults to
// .rtn (the existing behavior). When set to .awq / .smoothQuant /
// .gptq, the converter consults `--calib-stats <path>` to pick the
// per-tensor stats; if a tensor isn't represented in the stats it
// falls back to RTN silently (matches what an under-specified
// calibration set would imply).
var quantMethod: QuantMethod = .rtn
var calibStatsPath: String?

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--hf-ckpt-path":
        guard !args.isEmpty else { usage() }
        hfPath = args.removeFirst()
    case "--save-path":
        guard !args.isEmpty else { usage() }
        savePath = args.removeFirst()
    case "--n-experts":
        guard !args.isEmpty, let n = Int(args.removeFirst()) else { usage() }
        nExperts = n
    case "--model-parallel":
        guard !args.isEmpty, let n = Int(args.removeFirst()) else { usage() }
        mp = n
    case "--target-dtype":
        guard !args.isEmpty, let t = TargetDType(rawValue: args.removeFirst()) else { usage() }
        targetDType = t
    case "--shard-size-gb":
        guard !args.isEmpty, let v = Double(args.removeFirst()) else { usage() }
        shardSizeGB = v
    case "--quant-method":
        guard !args.isEmpty else { usage() }
        let raw = args.removeFirst()
        guard let m = QuantMethod(rawValue: raw) else {
            FileHandle.standardError.write(Data(
                "--quant-method must be one of rtn|awq|smoothQuant|gptq (got '\(raw)')\n".utf8))
            exit(2)
        }
        quantMethod = m
    case "--calib-stats":
        guard !args.isEmpty else { usage() }
        calibStatsPath = args.removeFirst()
    default:
        usage()
    }
}

guard let hf = hfPath, let save = savePath, nExperts > 0 else { usage() }
precondition(nExperts % mp == 0, "n_experts must be divisible by model_parallel")
precondition(mp == 1, "model_parallel > 1 not supported by this Swift port (single-rank)")
// Load the calibration bundle if --calib-stats was supplied. Once
// here, per-tensor lookups are cheap (dict access). When the user
// asks for a non-RTN method but doesn't pass --calib-stats, error
// out — silently demoting to RTN would surprise them.
let calibDir: CalibrationDir?
if let path = calibStatsPath {
    do {
        calibDir = try CalibrationDir(url: URL(fileURLWithPath: path))
        FileHandle.standardError.write(Data(
            "Loaded calibration stats: \(calibDir!.statsByName.count) "
            + "layers\n".utf8))
    } catch {
        FileHandle.standardError.write(Data(
            "--calib-stats failed to load: \(error)\n".utf8))
        exit(1)
    }
} else {
    calibDir = nil
    if quantMethod != .rtn {
        FileHandle.standardError.write(Data(
            "--quant-method \(quantMethod.rawValue) requires --calib-stats "
            + "<path>\n".utf8))
        exit(2)
    }
}
// Cap shard size to ~95% of the device's per-MTLBuffer limit. The
// runtime mmaps each shard as one MTLBuffer, so a shard larger than
// `maxBufferLength` is unloadable on this machine. The 95% margin
// leaves room for safetensors header + page alignment rounding.
var shardSizeBytes = Int(shardSizeGB * 1_000_000_000)
if let dev = MTLCreateSystemDefaultDevice() {
    let cap = Int(Double(dev.maxBufferLength) * 0.95)
    if shardSizeBytes > cap {
        let gib = 1024.0 * 1024.0 * 1024.0
        print("Capping --shard-size-gb \(shardSizeGB) to "
              + String(format: "%.2f", Double(cap) / gib)
              + " GiB (device maxBufferLength = "
              + String(format: "%.2f", Double(dev.maxBufferLength) / gib)
              + " GiB).")
        shardSizeBytes = cap
        shardSizeGB = Double(cap) / 1_000_000_000.0
    }
}

let hfDir = URL(fileURLWithPath: hf)
let saveDir = URL(fileURLWithPath: save)
try FileManager.default.createDirectory(at: saveDir, withIntermediateDirectories: true)

// Rename mapping (`leafMapping`, `renameKey`, `shouldSkip`),
// dtype packers (`floatToBF16`, `floatToF16`, `packNative`,
// `deqE4M3`, `deqE2M1`, `deqE8M0`), LUTs (`e4m3LUT`, `e2m1LUT`,
// `e8m0LUT`), dtype probes (`isFP8DType`, `isFP4DType`,
// `isE8M0DType`), and the FP8/FP4 → BF16/F16 fusion functions
// (`fuseFP8ToNative`, `fuseFP4ToNative`) now live in the
// DeepSeekConverter library — imported above.
//
// `scaleNameFor` is also there (Rename.swift); the local
// duplicate below is removed.


// ---------- Walk + collect ----------

let fm = FileManager.default
let inputs = try fm.contentsOfDirectory(at: hfDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "safetensors" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

guard !inputs.isEmpty else {
    FileHandle.standardError.write(Data("no .safetensors files in \(hf)\n".utf8))
    exit(1)
}

print("Converter: \(inputs.count) input shard(s)")
let dtypeNote: String
switch targetDType {
case .keep: dtypeNote = "(non-native FP8/FP4 preserved)"
case .int8: dtypeNote = "(Linear weights → INT8 W8A16 + F16 group scales; other tensors → BF16)"
case .int4: dtypeNote = "(Linear weights → INT4 W4A16 packed 2-per-byte + F16 group scales; other tensors → BF16)"
case .int2: dtypeNote = "(Linear weights → INT2 W2A16 packed 4-per-byte + F16 group scales; other tensors → BF16)"
case .bf16, .f16: dtypeNote = "(FP8/FP4+scale fused into native)"
}
print("  target dtype:    \(targetDType.rawValue) \(dtypeNote)")
print("  shard size cap:  \(String(format: "%.1f", shardSizeGB)) GB")
print("  sharding:        layer-aligned (top-level + one bucket per layer)")
switch targetDType {
case .keep:
    break
case .int8:
    print("  note:            INT8 quant covers Linear weights only; net disk ≈ ½ × BF16.")
case .int4:
    print("  note:            INT4 quant covers Linear weights only; net disk ≈ ¼ × BF16. " +
          "Accuracy lower than INT8 — RTN only, no calibration.")
case .int2:
    print("  note:            INT2 quant covers Linear weights only; net disk ≈ ⅛ × BF16. " +
          "Brutal accuracy hit at this bit-width — RTN only, calibration recommended for production.")
case .bf16, .f16:
    print("  note:            FP8 → 2× size, FP4 → 4× size; expect ~3-4× the " +
          "input directory's footprint on disk.")
}
print("Indexing input …")

// Mapping: new_name → (sourceURL, byte_offset, byte_count, dtype, shape)
struct PendingTensor {
    let url: URL
    let offset: Int
    let byteCount: Int
    let dtype: String
    let shape: [Int]
}

var plan: [String: PendingTensor] = [:]

for inputURL in inputs {
    // `SafeTensorsFile` mmaps the whole shard and wraps it as an `MTLBuffer`.
    // `MTLBuffer` is an ObjC protocol — its release (and our `munmap`
    // deallocator) can ride the autorelease pool. Without an explicit drain
    // every shard's mapping stays alive for the whole indexing phase: at
    // ~14 GB virtual per shard with header pages faulted in, that's real
    // RSS that goes to swap under pressure.
    try autoreleasepool {
        let stf = try SafeTensorsFile(url: inputURL)
        for (origName, entry) in stf.entries {
            if shouldSkip(origName) { continue }
            let newName = renameKey(origName)
            // Compute absolute byte range in the input file.
            // SafeTensorsFile keeps the data section start internally; we need
            // to recompute it here by re-reading the header length.
            let absOffset = try absoluteOffset(of: entry, in: inputURL)
            let pt = PendingTensor(url: inputURL, offset: absOffset,
                                    byteCount: entry.dataOffsets[1] - entry.dataOffsets[0],
                                    dtype: entry.dtype, shape: entry.shape)
            // All tensors (wo_a included) go into a single map. An earlier
            // refactor segregated `wo_a.scale` into its own dict but never
            // looked it up again, which silently broke wo_a fusion in
            // every mode (FP8 wo_a was passed through verbatim instead of
            // being fused into BF16 / INT8). Treating it like any other
            // scale lets the existing FP8/INT8 fusion paths find it via
            // `plan[scaleName]`. The dispatch loop's E8M0-skip check
            // guarantees the standalone scale is dropped (parent weight
            // is in plan → `continue`).
            plan[newName] = pt
        }
    }
}

print("Collected \(plan.count) tensors.")

// ---------- Build per-tensor write entries ----------
//
// For each tensor in the input:
//   - If it's a non-native dtype (FP8 or FP4) AND has a companion .scale,
//     fuse them into a single tensor of `targetDType` (BF16 or F16).
//   - If it's an E8M0 scale tensor, drop it: it's been consumed by its
//     parent weight's fusion above.
//   - Otherwise pass through unchanged (BF16/F16/F32/I32/I8 are all
//     native or trivially native-compatible).
//
// When `targetDType == .keep`, fall back to the original convert.py
// behaviour: only fuse wo_a (which the reference inference path expects
// in BF16) and relabel I8 expert weights as F4_E2M1.

struct WriteEntry {
    let name: String
    let dtype: String
    let shape: [Int]
    let byteCount: Int
    let source: SafeTensorsWriter.Source
}

// `scaleNameFor` now lives in DeepSeekConverter (Rename.swift).
// Imported above; the original definition uses
// `replacingOccurrences(of:.weight, with:.scale)` which is
// equivalent for the names we emit (`<base>.weight`).

var writeEntries: [WriteEntry] = []
var scalesConsumed = Set<String>()    // .scale entries that have been folded in
var int8QuantizedCount = 0
var int4QuantizedCount = 0
var int2QuantizedCount = 0

for newName in plan.keys.sorted() {
    let pt = plan[newName]!

    // Skip standalone scale tensors when the parent weight is going to
    // consume them (or already has).
    if isE8M0DType(pt.dtype) {
        if scalesConsumed.contains(newName) { continue }
        // If the parent weight isn't in the plan (rare), pass the scale
        // through as-is so nothing is silently dropped.
        let parent = newName.replacingOccurrences(of: ".scale", with: ".weight")
        if plan[parent] != nil { continue }
    }

    // ----- INT8 W8A16 quantization branch -----
    // Non-whitelisted tensors fall through to BF16 fusion (effectiveTarget
    // below). The branch emits two `WriteEntry`s per Linear weight: the
    // INT8 weight under `<base>.weight` and the F16 group scale under
    // `<base>.scale`. The runtime loader (`Assembly.loadLinear`) already
    // pairs them — no assembly changes needed.
    if targetDType == .int8 {
        let outDim = pt.shape[0]
        // Compute logical inDim, special-casing packed FP4 layout.
        let logicalInDim: Int
        let isPackedFP4 = isFP4DType(pt.dtype) || (newName.contains(".experts.") &&
                                                    (pt.dtype.uppercased() == "I8" ||
                                                     pt.dtype.uppercased() == "U8"))
        if isPackedFP4 {
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] * 2 : 0
        } else {
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] : 0
        }

        if pt.shape.count == 2 &&
           shouldQuantizeToInt8(newName, lastDim: logicalInDim) {
            let inDim = logicalInDim
            let blocksIn = inDim / kInt8GroupK
            let weightBytes = outDim * inDim
            let scaleBytes = outDim * blocksIn * 2
            let logicalShape = [outDim, inDim]
            let scaleShape = [outDim, blocksIn]
            let scaleName = scaleNameFor(newName)
            let srcDtype = pt.dtype
            let companion = plan[scaleName]
            if companion != nil &&
                (isFP8DType(srcDtype) || isPackedFP4) {
                scalesConsumed.insert(scaleName)
            }

            // Lazy compute shared between the two write entries. Both
            // closures capture this `var` by reference; the writer
            // dispatches entries in declaration order, so the weight closure
            // populates `cached`, the scale closure consumes it and nils
            // it out to release the weight buffer immediately.
            var cached: (weight: Data, scale: Data)? = nil
            let upper = srcDtype.uppercased()
            let weightURL = pt.url
            let weightOffset = pt.offset
            let scaleURL = companion?.url
            let scaleOffset = companion?.offset ?? 0
            let isFP8 = isFP8DType(srcDtype)
            let isFP4 = isPackedFP4
            // For calibrated dispatch, we look up CalibrationStats by
            // the OUTPUT (renamed) tensor name without the trailing
            // ".weight" suffix — matches the convention
            // `deepseek_calibrate` writes. If no entry is found we
            // fall back to RTN (silently for non-quantized methods on
            // tensors out of the calibration corpus's scope, like
            // freshly-added bias / norm Linears).
            let calibStem: String = {
                if newName.hasSuffix(".weight") {
                    return String(newName.dropLast(".weight".count))
                }
                return newName
            }()
            let calibStats = calibDir?.statsByName[calibStem]
            let calibHessian = (quantMethod == .gptq)
                ? calibDir?.hessian(for: calibStem) : nil
            let effectiveMethod: QuantMethod = {
                guard quantMethod != .rtn else { return .rtn }
                // Without stats we can't run AWQ / SmoothQuant /
                // GPTQ; downgrade to RTN silently for this tensor.
                if calibStats == nil { return .rtn }
                if quantMethod == .gptq && calibHessian == nil { return .rtn }
                return quantMethod
            }()
            let compute: () throws -> (weight: Data, scale: Data) = {
                if let c = cached { return c }
                let r: (weight: Data, scale: Data)
                if isFP8, let sURL = scaleURL {
                    r = try quantizeFP8ToInt8(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e4m3LUT: e4m3LUT, e8m0LUT: e8m0LUT)
                } else if isFP4, let sURL = scaleURL {
                    r = try quantizeFP4ToInt8(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e2m1LUT: e2m1LUT, e8m0LUT: e8m0LUT)
                } else if upper == "BF16" {
                    switch effectiveMethod {
                    case .rtn:
                        r = try quantizeBF16ToInt8(
                            srcURL: weightURL, srcOffset: weightOffset,
                            outDim: outDim, inDim: inDim)
                    case .awq:
                        let stats = calibStats!  // forced by effectiveMethod resolution
                        let res = try awqQuantizeBF16ToInt8(
                            srcURL: weightURL, srcOffset: weightOffset,
                            outDim: outDim, inDim: inDim,
                            actAbsMax: stats.perChannelAbsMax,
                            alpha: 0.5)
                        r = (res.weight, res.scale)
                        // The inverseChannelScale produced by AWQ
                        // would belong as a sidecar tensor consumed
                        // by `Linear.inverseChannelScale` at load
                        // time. Not yet written here — the loader
                        // doesn't read it either, so emitting it
                        // would just bloat the safetensors output.
                        // Documented follow-up.
                        _ = res.inverseChannelScale
                    case .smoothQuant:
                        let stats = calibStats!
                        let res = try smoothQuantBF16ToInt8(
                            srcURL: weightURL, srcOffset: weightOffset,
                            outDim: outDim, inDim: inDim,
                            actAbsMax: stats.perChannelAbsMax,
                            alpha: 0.5)
                        r = (res.weight, res.scale)
                        _ = res.inverseChannelScale
                    case .gptq:
                        let H = calibHessian!  // forced by effectiveMethod
                        r = try gptqQuantizeBF16ToInt8(
                            srcURL: weightURL, srcOffset: weightOffset,
                            outDim: outDim, inDim: inDim,
                            hessian: H)
                    }
                } else if upper == "F32" {
                    r = try quantizeF32ToInt8(srcURL: weightURL, srcOffset: weightOffset,
                                               outDim: outDim, inDim: inDim)
                } else {
                    throw NSError(domain: "Int8Quant", code: 1, userInfo: [
                        NSLocalizedDescriptionKey:
                            "INT8 quant: unsupported source dtype \(srcDtype) for \(newName)"
                    ])
                }
                cached = r
                return r
            }

            writeEntries.append(WriteEntry(
                name: newName, dtype: "I8", shape: logicalShape, byteCount: weightBytes,
                source: .compute(byteCount: weightBytes) { try compute().weight }))
            writeEntries.append(WriteEntry(
                name: scaleName, dtype: "F16", shape: scaleShape, byteCount: scaleBytes,
                source: .compute(byteCount: scaleBytes) {
                    let s = try compute().scale
                    cached = nil   // release the weight buffer
                    return s
                }))
            int8QuantizedCount += 1
            continue
        }
        // Fall through: non-whitelisted tensors get BF16 fusion below.
    }

    // ----- INT4 W4A16 quantization branch -----
    // Same whitelist + same per-128-block layout as INT8; the only
    // differences are nibble packing and the [-8, 7] quantization range.
    if targetDType == .int4 {
        let outDim = pt.shape[0]
        let logicalInDim: Int
        let scaleName = scaleNameFor(newName)
        let companion = plan[scaleName]
        let scaleDtype = companion?.dtype.uppercased() ?? ""
        // Disambiguate two cases of `pt.dtype == "I8"`:
        //   - HF V4 ships FP4 expert weights packed two-per-byte
        //     with safetensors dtype I8/U8 and an E8M0 scale.
        //   - Our previous INT8 conversion ships real signed-int8
        //     weights with an F16 scale.
        // Use the scale dtype to tell them apart so we run the
        // correct row dequantizer.
        let isPackedFP4 = isFP4DType(pt.dtype) || (newName.contains(".experts.") &&
                                                    (pt.dtype.uppercased() == "I8" ||
                                                     pt.dtype.uppercased() == "U8") &&
                                                    isE8M0DType(scaleDtype))
        let isRealInt8 = pt.dtype.uppercased() == "I8" && scaleDtype == "F16"
        if isPackedFP4 {
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] * 2 : 0
        } else {
            // INT8 and BF16/F32 use the canonical 2D layout.
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] : 0
        }

        if pt.shape.count == 2 &&
           shouldQuantizeToInt4(newName, lastDim: logicalInDim) {
            let inDim = logicalInDim
            let blocksIn = inDim / kInt4GroupK
            let packedRowBytes = inDim / 2
            let weightBytes = outDim * packedRowBytes
            let scaleBytes = outDim * blocksIn * 2
            let logicalShape = [outDim, inDim]
            let scaleShape = [outDim, blocksIn]
            let srcDtype = pt.dtype
            if companion != nil &&
                (isFP8DType(srcDtype) || isPackedFP4 || isRealInt8) {
                scalesConsumed.insert(scaleName)
            }

            var cached: (weight: Data, scale: Data)? = nil
            let upper = srcDtype.uppercased()
            let weightURL = pt.url
            let weightOffset = pt.offset
            let scaleURL = companion?.url
            let scaleOffset = companion?.offset ?? 0
            let isFP8 = isFP8DType(srcDtype)
            let isFP4 = isPackedFP4
            let isInt8 = isRealInt8
            let compute: () throws -> (weight: Data, scale: Data) = {
                if let c = cached { return c }
                let r: (weight: Data, scale: Data)
                if isFP8, let sURL = scaleURL {
                    r = try quantizeFP8ToInt4(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e4m3LUT: e4m3LUT, e8m0LUT: e8m0LUT)
                } else if isFP4, let sURL = scaleURL {
                    r = try quantizeFP4ToInt4(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e2m1LUT: e2m1LUT, e8m0LUT: e8m0LUT)
                } else if isInt8, let sURL = scaleURL {
                    r = try quantizeInt8ToInt4(weightURL: weightURL, weightOffset: weightOffset,
                                                scaleURL: sURL, scaleOffset: scaleOffset,
                                                outDim: outDim, inDim: inDim)
                } else if upper == "BF16" {
                    r = try quantizeBF16ToInt4(srcURL: weightURL, srcOffset: weightOffset,
                                                outDim: outDim, inDim: inDim)
                } else if upper == "F32" {
                    r = try quantizeF32ToInt4(srcURL: weightURL, srcOffset: weightOffset,
                                               outDim: outDim, inDim: inDim)
                } else {
                    throw NSError(domain: "Int4Quant", code: 1, userInfo: [
                        NSLocalizedDescriptionKey:
                            "INT4 quant: unsupported source dtype \(srcDtype) for \(newName)"
                    ])
                }
                cached = r
                return r
            }

            writeEntries.append(WriteEntry(
                name: newName, dtype: "I4", shape: logicalShape, byteCount: weightBytes,
                source: .compute(byteCount: weightBytes) { try compute().weight }))
            writeEntries.append(WriteEntry(
                name: scaleName, dtype: "F16", shape: scaleShape, byteCount: scaleBytes,
                source: .compute(byteCount: scaleBytes) {
                    let s = try compute().scale
                    cached = nil
                    return s
                }))
            int4QuantizedCount += 1
            continue
        }
    }

    // ----- INT2 W2A16 quantization branch -----
    // Same shape conventions as INT4 with 4-per-byte packing and the
    // [-2, 1] quantization range.
    if targetDType == .int2 {
        let outDim = pt.shape[0]
        let logicalInDim: Int
        let scaleName = scaleNameFor(newName)
        let companion = plan[scaleName]
        let scaleDtype = companion?.dtype.uppercased() ?? ""
        // Same FP4-vs-INT8 disambiguation as the INT4 branch above —
        // see comments there. INT8-source path lets users
        // re-quantize an existing INT8 checkpoint without going back
        // to the HF native weights.
        let isPackedFP4 = isFP4DType(pt.dtype) || (newName.contains(".experts.") &&
                                                    (pt.dtype.uppercased() == "I8" ||
                                                     pt.dtype.uppercased() == "U8") &&
                                                    isE8M0DType(scaleDtype))
        let isRealInt8 = pt.dtype.uppercased() == "I8" && scaleDtype == "F16"
        if isPackedFP4 {
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] * 2 : 0
        } else {
            logicalInDim = pt.shape.count >= 2 ? pt.shape[1] : 0
        }

        if pt.shape.count == 2 &&
           shouldQuantizeToInt2(newName, lastDim: logicalInDim) {
            let inDim = logicalInDim
            let blocksIn = inDim / kInt2GroupK
            let packedRowBytes = inDim / 4
            let weightBytes = outDim * packedRowBytes
            let scaleBytes = outDim * blocksIn * 2
            let logicalShape = [outDim, inDim]
            let scaleShape = [outDim, blocksIn]
            let srcDtype = pt.dtype
            if companion != nil &&
                (isFP8DType(srcDtype) || isPackedFP4 || isRealInt8) {
                scalesConsumed.insert(scaleName)
            }

            var cached: (weight: Data, scale: Data)? = nil
            let upper = srcDtype.uppercased()
            let weightURL = pt.url
            let weightOffset = pt.offset
            let scaleURL = companion?.url
            let scaleOffset = companion?.offset ?? 0
            let isFP8 = isFP8DType(srcDtype)
            let isFP4 = isPackedFP4
            let isInt8 = isRealInt8
            let compute: () throws -> (weight: Data, scale: Data) = {
                if let c = cached { return c }
                let r: (weight: Data, scale: Data)
                if isFP8, let sURL = scaleURL {
                    r = try quantizeFP8ToInt2(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e4m3LUT: e4m3LUT, e8m0LUT: e8m0LUT)
                } else if isFP4, let sURL = scaleURL {
                    r = try quantizeFP4ToInt2(weightURL: weightURL, weightOffset: weightOffset,
                                               scaleURL: sURL, scaleOffset: scaleOffset,
                                               outDim: outDim, inDim: inDim,
                                               e2m1LUT: e2m1LUT, e8m0LUT: e8m0LUT)
                } else if isInt8, let sURL = scaleURL {
                    r = try quantizeInt8ToInt2(weightURL: weightURL, weightOffset: weightOffset,
                                                scaleURL: sURL, scaleOffset: scaleOffset,
                                                outDim: outDim, inDim: inDim)
                } else if upper == "BF16" {
                    r = try quantizeBF16ToInt2(srcURL: weightURL, srcOffset: weightOffset,
                                                outDim: outDim, inDim: inDim)
                } else if upper == "F32" {
                    r = try quantizeF32ToInt2(srcURL: weightURL, srcOffset: weightOffset,
                                               outDim: outDim, inDim: inDim)
                } else {
                    throw NSError(domain: "Int2Quant", code: 1, userInfo: [
                        NSLocalizedDescriptionKey:
                            "INT2 quant: unsupported source dtype \(srcDtype) for \(newName)"
                    ])
                }
                cached = r
                return r
            }

            writeEntries.append(WriteEntry(
                name: newName, dtype: "I2", shape: logicalShape, byteCount: weightBytes,
                source: .compute(byteCount: weightBytes) { try compute().weight }))
            writeEntries.append(WriteEntry(
                name: scaleName, dtype: "F16", shape: scaleShape, byteCount: scaleBytes,
                source: .compute(byteCount: scaleBytes) {
                    let s = try compute().scale
                    cached = nil
                    return s
                }))
            int2QuantizedCount += 1
            continue
        }
    }

    // Effective target for non-INT* (or non-whitelisted) tensors: when the
    // user asked for INT8/INT4/INT2, anything that didn't quantize falls
    // back to BF16; otherwise honor the user's target verbatim.
    let effectiveTarget: TargetDType =
        (targetDType == .int8 || targetDType == .int4 || targetDType == .int2) ? .bf16 : targetDType

    // Native-dtype transcoding: FP8/FP4 weight + scale → BF16/F16.
    if effectiveTarget != .keep {
        let scaleName = scaleNameFor(newName)
        if isFP8DType(pt.dtype), let sc = plan[scaleName] {
            let outDim = pt.shape[0]
            let inDim = pt.shape[1]
            let n = outDim * inDim * effectiveTarget.bytesPerElement
            scalesConsumed.insert(scaleName)
            let target = effectiveTarget
            writeEntries.append(WriteEntry(
                name: newName, dtype: target.safetensorsTag, shape: pt.shape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP8ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: target)
                }))
            continue
        }
        if isFP4DType(pt.dtype) || (newName.contains(".experts.") &&
                                     (pt.dtype.uppercased() == "I8" ||
                                      pt.dtype.uppercased() == "U8")),
           let sc = plan[scaleName] {
            // FP4 weights ship as packed [out, in/2]. The logical inDim is
            // double the stored last-dim.
            let outDim = pt.shape[0]
            let inDim = pt.shape[1] * 2
            let n = outDim * inDim * effectiveTarget.bytesPerElement
            scalesConsumed.insert(scaleName)
            let target = effectiveTarget
            let logicalShape = [outDim, inDim]
            writeEntries.append(WriteEntry(
                name: newName, dtype: target.safetensorsTag, shape: logicalShape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP4ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: target)
                }))
            continue
        }
    } else {
        // Legacy convert.py-style behaviour: only fuse wo_a, relabel experts.
        if newName.hasSuffix(".wo_a.weight"),
           let sc = plan[scaleNameFor(newName)] {
            let outDim = pt.shape[0]
            let inDim = pt.shape[1]
            let n = outDim * inDim * 2
            scalesConsumed.insert(scaleNameFor(newName))
            writeEntries.append(WriteEntry(
                name: newName, dtype: "BF16", shape: pt.shape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP8ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: .bf16)
                }))
            continue
        }
        var dtype = pt.dtype
        if newName.contains(".experts.") && newName.hasSuffix(".weight") {
            if pt.dtype.uppercased() == "I8" || pt.dtype.uppercased() == "U8" {
                dtype = "F4_E2M1"
            }
        }
        writeEntries.append(WriteEntry(
            name: newName, dtype: dtype, shape: pt.shape, byteCount: pt.byteCount,
            source: .file(url: pt.url, offset: pt.offset, byteCount: pt.byteCount)))
        continue
    }

    // Default pass-through (native dtype already, or unknown name).
    writeEntries.append(WriteEntry(
        name: newName, dtype: pt.dtype, shape: pt.shape, byteCount: pt.byteCount,
        source: .file(url: pt.url, offset: pt.offset, byteCount: pt.byteCount)))
}

// ---------- Group tensors by depth, then pack ----------
//
// Strategy: each transformer layer goes into its own shard whenever
// possible. If a layer is bigger than the shard cap, split it across
// consecutive shards (the next layer always starts a fresh shard).
// Top-level tensors (embed, head, norm, hc_head_*) go into shard 0.
//
// Why: the forward pass touches layers in strict depth order. Aligning
// shards to layer boundaries means that during one forward pass we read
// shard 0 once, then 1, then 2, ..., never going backwards. The OS page
// cache can prefetch the next shard while we're still computing on the
// current one, and evict already-consumed shards under memory pressure
// without disturbing the layer being actively read.

func depthKey(of name: String) -> Int {
    // Returns -1 for top-level tensors, otherwise the layer index.
    if name.hasPrefix("layers.") {
        let parts = name.split(separator: ".")
        if parts.count >= 2, let i = Int(parts[1]) { return i }
    }
    if name.hasPrefix("mtp.") {
        let parts = name.split(separator: ".")
        if parts.count >= 2, let i = Int(parts[1]) {
            return 100_000 + i   // sort MTP after main layers
        }
    }
    return -1
}

// Bucket entries by depth.
var bucketsByDepth: [Int: [WriteEntry]] = [:]
for e in writeEntries { bucketsByDepth[depthKey(of: e.name), default: []].append(e) }

// Sanity log: report what depth structure we discovered just from the
// tensor names. Useful to cross-check against config.json's `n_layers`
// and `n_mtp_layers` if you have them.
let layerDepths = bucketsByDepth.keys.filter { $0 >= 0 && $0 < 100_000 }.sorted()
let mtpDepths = bucketsByDepth.keys.filter { $0 >= 100_000 }.sorted().map { $0 - 100_000 }
let topLevelTensors = bucketsByDepth[-1]?.count ?? 0
print("Discovered structure (from tensor names, no config.json read):")
print("  top-level tensors:  \(topLevelTensors)")
if let lo = layerDepths.first, let hi = layerDepths.last {
    print("  layers:             \(layerDepths.count) (indices \(lo)…\(hi))")
}
if !mtpDepths.isEmpty {
    print("  mtp blocks:         \(mtpDepths.count)")
}

struct Shard {
    var entries: [WriteEntry]
    var totalBytes: Int
}

var shards: [Shard] = []
let depths = bucketsByDepth.keys.sorted()
for depth in depths {
    let bucket = bucketsByDepth[depth]!
    // Within a depth bucket, sort tensors by name for determinism.
    let sorted = bucket.sorted { $0.name < $1.name }
    var current = Shard(entries: [], totalBytes: 0)
    for e in sorted {
        if !current.entries.isEmpty && current.totalBytes + e.byteCount > shardSizeBytes {
            shards.append(current)
            current = Shard(entries: [], totalBytes: 0)
        }
        current.entries.append(e)
        current.totalBytes += e.byteCount
    }
    if !current.entries.isEmpty { shards.append(current) }
}

print("Packing \(writeEntries.count) tensors into \(shards.count) shard(s) " +
      "(layer-aligned; max \(String(format: "%.1f", shardSizeGB)) GB/shard)")
if targetDType == .int8 {
    print("  INT8-quantized:  \(int8QuantizedCount) Linear weight(s) " +
          "(each splits into <name>.weight + <name>.scale).")
}
if targetDType == .int4 {
    print("  INT4-quantized:  \(int4QuantizedCount) Linear weight(s) " +
          "(each splits into <name>.weight + <name>.scale, packed 2-per-byte).")
}
if targetDType == .int2 {
    print("  INT2-quantized:  \(int2QuantizedCount) Linear weight(s) " +
          "(each splits into <name>.weight + <name>.scale, packed 4-per-byte).")
}

// ---------- Write each shard + index.json ----------

let total = shards.count
var weightMap: [String: String] = [:]      // tensor name → shard filename
let totalBytes = shards.reduce(0) { $0 + $1.totalBytes }

// Resume detection: if the output dir already contains shards from a
// previous (interrupted) run, skip the ones that are present and the
// right size. Sharding is deterministic given the same input + flags,
// so file `model-(i+1)-of-total.safetensors` produced now must match
// any file with the same name from a prior run.
//
// A shard is considered "complete" iff:
//   - its filename matches model-(i+1)-of-(total).safetensors exactly
//     (so a partial write under a different `total` is skipped)
//   - its size on disk is at least `shard.totalBytes` (the header adds a
//     few KB of JSON, so >= totalBytes implies full write)
//
// We also pre-populate `weightMap` for the skipped shards so the final
// index.json comes out correct.

func expectedFilename(i: Int, total: Int) -> String {
    return String(format: "model-%05d-of-%05d.safetensors", i + 1, total)
}

var resumeFromShard = 0
for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    let url = saveDir.appendingPathComponent(fileName)
    guard let attrs = try? fm.attributesOfItem(atPath: url.path),
          let size = attrs[.size] as? Int,
          size >= shard.totalBytes else { break }
    // Mark as already-written, populate the index.
    for e in shard.entries { weightMap[e.name] = fileName }
    resumeFromShard = i + 1
}
if resumeFromShard > 0 {
    let skippedBytes = shards[..<resumeFromShard].reduce(0) { $0 + $1.totalBytes }
    print("Resume: \(resumeFromShard)/\(total) shard(s) already on disk " +
          "(\(String(format: "%.1f", Double(skippedBytes) / 1_000_000_000)) GB skipped).")
}

for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    if i < resumeFromShard { continue }
    // Each shard write churns through many GB of NSData-backed buffers
    // (input slurps, fused BF16 outputs, output `FileHandle` write chunks).
    // Drain the pool at the shard boundary so transient allocations from
    // one shard don't survive into the next.
    try autoreleasepool {
        let outURL = saveDir.appendingPathComponent(fileName)
        let w = SafeTensorsWriter()
        for e in shard.entries {
            w.add(name: e.name, dtype: e.dtype, shape: e.shape, source: e.source)
            weightMap[e.name] = fileName
        }
        print("  [\(i + 1)/\(total)] \(fileName) — " +
              "\(shard.entries.count) tensors, " +
              "\(String(format: "%.2f", Double(shard.totalBytes) / 1_000_000_000)) GB")
        try w.write(to: outURL)
    }
}

// HuggingFace-style index.json, useful for tooling and reproducibility.
let indexObj: [String: Any] = [
    "metadata": ["total_size": totalBytes] as [String: Any],
    "weight_map": weightMap,
]
let indexData = try JSONSerialization.data(withJSONObject: indexObj,
                                            options: [.sortedKeys, .prettyPrinted])
let indexURL = saveDir.appendingPathComponent("model.safetensors.index.json")
try indexData.write(to: indexURL)
print("Wrote \(weightMap.count) tensors across \(total) shard(s); " +
      "\(String(format: "%.1f", Double(totalBytes) / 1_000_000_000)) GB total.")

// ---------- Copy tokenizer + config ----------

for f in ["tokenizer.json", "tokenizer_config.json", "config.json",
          "generation_config.json", "special_tokens_map.json"] {
    let src = hfDir.appendingPathComponent(f)
    if fm.fileExists(atPath: src.path) {
        let dst = saveDir.appendingPathComponent(f)
        if fm.fileExists(atPath: dst.path) { try? fm.removeItem(at: dst) }
        try fm.copyItem(at: src, to: dst)
    }
}

print("Done.")
print("Run:  swift run -c release deepseek \(saveDir.path) \"<prompt>\" --mode chat")

// ---------- Helpers ----------

/// Re-reads the header of `url` to compute the absolute byte position of
/// `entry.dataOffsets[0]` within the file. SafeTensorsFile already
/// computed this internally but doesn't expose it; this function repeats
/// the small header read to get it deterministically.
func absoluteOffset(of entry: SafeTensorsFile.Entry, in url: URL) throws -> Int {
    guard let fh = FileHandle(forReadingAtPath: url.path) else {
        throw NSError(domain: "absoluteOffset", code: 1)
    }
    defer { try? fh.close() }
    guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
        throw NSError(domain: "absoluteOffset", code: 2)
    }
    let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
    return 8 + Int(headerLen) + entry.dataOffsets[0]
}
