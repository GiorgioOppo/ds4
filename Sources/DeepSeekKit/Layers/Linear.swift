import Foundation
import MLX
import MLXNN
import MLXFast

// Dequantization LUTs in bf16 so the unpack tensor (one entry per
// weight element — [outFeatures, inFeatures]) is half the size of the
// f32 equivalent. bf16 covers the full FP4-E2M1 value range
// exactly, and represents the realistic E8M0 scale range used by
// DeepSeek-V4 (powers of 2 between ~2⁻²⁰ and ~2¹⁰). Extreme E8M0
// exponents (2⁻¹²⁷, 2¹²⁸) saturate, which the reference checkpoint
// does not produce.
private let fp4Lut = MLXArray((0..<16).map { Float(dequantE2M1(UInt8($0))) }).asType(.bfloat16)
private let e8m0Lut = MLXArray((0..<256).map { Float(dequantE8M0(UInt8($0))) }).asType(.bfloat16)
// FP8 E4M3 byte → float LUT. Without this, the W8A8 path would multiply
// the raw uint8 byte (treated as integer 0..255) by the block scale,
// producing weights ~33× larger than intended for typical magnitudes.
// NaN encodings (0x7F, 0xFF) are replaced with 0 — they never appear
// in a trained checkpoint and would otherwise propagate NaN through
// the matmul.
private let fp8E4M3Lut: MLXArray = {
    let values = (0..<256).map { (i: Int) -> Float in
        let v = dequantE4M3(UInt8(i))
        return v.isNaN ? 0 : v
    }
    return MLXArray(values).asType(.bfloat16)
}()

public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    
    // Eager mode: weight stored directly
    private var _weight: Tensor?
    private var _scale: Tensor?
    
    // Lazy mode: load from disk on demand
    private var _weightName: String?
    private var _scaleName: String?
    private weak var _loader: WeightLoader?
    
    public var weight: Tensor {
        if let w = _weight { return w }
        // Lazy load
        guard let loader = _loader, let name = _weightName else {
            fatalError("Linear: no weight and no loader configured for lazy loading")
        }
        let w = (try? loader.load(name)) ?? Tensor.empty(shape: [outFeatures, inFeatures], dtype: .f32)
        return w
    }
    
    public var scale: Tensor? {
        if _weight != nil { return _scale }
        // Lazy load
        guard let loader = _loader, let name = _scaleName else { return nil }
        return try? loader.load(name)
    }

    public let castOutputToBF16: Bool
    public let useW8A8Activations: Bool
    public var inverseChannelScale: Tensor? = nil

    /// Original dtype of the weight (needed for FP4 detection even in lazy mode)
    public let weightDType: DType

    /// When true, `callAsFunction` routes through
    /// `MLXFast.quantizedMatmul` over a re-quantized (groupSize=64,
    /// bits=4) form of the weight. The triplet is computed lazily on
    /// first call (running the existing `getDequantizedWeight()` once
    /// to bf16 then `MLX.quantized(...)`), stored in the loader cache
    /// under synthetic keys, and reused. `releaseExperts` drops those
    /// keys via the layer-prefix sweep, so streaming semantics are
    /// preserved.
    ///
    /// Default false. Wired to `true` for routed-expert linears in
    /// Assembly.load when DEEPSEEK_MLX_QUANT=1. The DeepSeek-specific
    /// FP4-E2M1 + E8M0-block-scale and FP8-E4M3 + 128×128-block-scale
    /// formats are not natively supported by MLX's quantized GEMM
    /// kernel, hence the one-time re-quant at load time.
    public var useMLXQuant: Bool = false

    // Eager init (existing API)
    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?,
                castOutputToBF16: Bool = false,
                useW8A8Activations: Bool = false,
                inverseChannelScale: Tensor? = nil) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self._weight = weight
        self._scale = scale
        self.weightDType = weight.dtype
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = useW8A8Activations
        self.inverseChannelScale = inverseChannelScale
    }
    
    // Lazy init (streaming mode)
    public init(inFeatures: Int, outFeatures: Int,
                weightName: String, scaleName: String?,
                weightDType: DType,
                loader: WeightLoader,
                castOutputToBF16: Bool = false) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self._weight = nil
        self._scale = nil
        self._weightName = weightName
        self._scaleName = scaleName
        self._loader = loader
        self.weightDType = weightDType
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = false
    }
    
    /// Re-quantize this Linear's weight into MLX's native 4-bit linear
    /// quantization format (groupSize=64) and cache the triplet in the
    /// loader cache under synthetic keys. Returns the cached triplet on
    /// subsequent calls.
    ///
    /// The DeepSeek checkpoint formats (FP4-E2M1 + E8M0 32-element
    /// block scales for experts, FP8-E4M3 + 128×128 block scales for
    /// MLA) are not natively supported by `MLXFast.quantizedMatmul`,
    /// which expects MLX's standard linear `(int4|int8) + fp scales +
    /// fp biases` layout. So we do one round-trip at first use:
    ///
    ///   raw weight (FP4/FP8) → getDequantizedWeight() → bf16 [out, in]
    ///                       → quantized(_, groupSize: 64, bits: 4)
    ///                       → (qw, scales, biases)
    ///
    /// The triplet lives in the loader cache under `<base>.mlxq.{w,s,b}`.
    /// The expert-streaming `releaseExperts` already releases everything
    /// matching `layers.K.ffn.experts.E.` so the synthetic keys are
    /// dropped on layer release without any extra wiring.
    ///
    /// Returns nil for eager-mode Linears (no `_loader`) — the caller
    /// must fall back to the standard path.
    public func getMLXQuant() -> (w: MLXArray, scales: MLXArray, biases: MLXArray)? {
        guard let loader = _loader, let weightName = _weightName else {
            return nil
        }
        let dotWeight = ".weight"
        let base = weightName.hasSuffix(dotWeight)
            ? String(weightName.dropLast(dotWeight.count))
            : weightName

        // Path A — MLX-native checkpoint: the triplet is ALREADY stored
        // on disk under `<base>.{weight,scales,biases}`. No re-quant
        // needed; just load. This is the mlx-community-style format.
        let nativeScalesName = "\(base).scales"
        let nativeBiasesName = "\(base).biases"
        if loader.dtype(of: nativeScalesName) != nil,
           loader.dtype(of: nativeBiasesName) != nil {
            if let qW = (try? loader.load(weightName))?.array,
               let qS = (try? loader.load(nativeScalesName))?.array,
               let qB = (try? loader.load(nativeBiasesName))?.array {
                return (qW, qS, qB)
            }
            return nil
        }

        // Path B — old custom-format checkpoint (FP4/FP8 + block scale).
        // Synthetic key namespace derived from the weight name (e.g.
        // "layers.0.ffn.experts.5.w1.weight" → base
        // "layers.0.ffn.experts.5.w1"). The triplet keys sit under the
        // same expert prefix so `releaseExperts(layer: K, indices: [E])`
        // picks them up.
        let qWKey = "\(base).mlxq.w"
        let qSKey = "\(base).mlxq.scales"
        let qBKey = "\(base).mlxq.biases"

        if let qW = loader.lookupRaw(qWKey),
           let qS = loader.lookupRaw(qSKey),
           let qB = loader.lookupRaw(qBKey) {
            return (qW, qS, qB)
        }

        // Cache miss: dequant to bf16 (existing code path), then
        // re-quantize. The bf16 temp is dropped at end of this scope.
        let bf16 = getDequantizedWeight().asType(.bfloat16)
        let (qW, qS, qB) = quantized(bf16, groupSize: 64, bits: 4)
        // Force materialization so the bf16 temp can be freed by the
        // next clearCache (called by MoE after dispatch).
        MLX.eval(qW)
        MLX.eval(qS)
        MLX.eval(qB)
        loader.storeRaw(qWKey, qW)
        loader.storeRaw(qSKey, qS)
        loader.storeRaw(qBKey, qB)
        return (qW, qS, qB)
    }

    /// True when the underlying checkpoint has the MLX-native quant
    /// triplet (`<base>.scales` + `<base>.biases`) for this Linear's
    /// weight. Cheap probe — does not load anything. Used by
    /// `callAsFunction` to auto-enable the `MLXFast.quantizedMatmul`
    /// fast path even when `useMLXQuant` wasn't explicitly set.
    private func hasMLXNativeTriplet() -> Bool {
        guard let loader = _loader, let weightName = _weightName,
              weightName.hasSuffix(".weight") else { return false }
        let base = String(weightName.dropLast(".weight".count))
        return loader.dtype(of: "\(base).scales") != nil
            && loader.dtype(of: "\(base).biases") != nil
    }

    public func getDequantizedWeight() -> MLXArray {
        let w = self.weight
        var wArr = w.array
        let isFP4 = weightDType == .fp4E2M1
        let isFP8 = weightDType == .fp8E4M3

        if isFP4 {
            let u8Arr = wArr.asType(.uint8)
            let low = bitwiseAnd(u8Arr, 15)
            let high = rightShift(u8Arr, 4)
            let stackedArr = MLX.stacked([low, high], axis: -1)
            let indices = stackedArr.reshaped([outFeatures, inFeatures])

            // `fp4Lut` is bf16 → take returns bf16, so the dequantized
            // weight footprint is half of the previous f32 path.
            var unpacked = take(fp4Lut, indices)

            if let s = self.scale {
                var sArr = s.array
                if s.dtype == .e8m0 {
                    sArr = take(e8m0Lut, sArr.asType(.uint8))   // bf16
                } else if sArr.dtype != .bfloat16 {
                    sArr = sArr.asType(.bfloat16)
                }

                let sExpanded = sArr.expandedDimensions(axes: [-1])
                let unpackedReshaped = unpacked.reshaped([outFeatures, inFeatures / 32, 32])
                let scaled = unpackedReshaped * sExpanded
                unpacked = scaled.reshaped([outFeatures, inFeatures])
            }
            wArr = unpacked                                        // bf16
        } else if isFP8 {
            // FP8 E4M3 weights: each byte encodes a float in [-448, +448]
            // via the IEEE-style sign/exp/mantissa layout (see
            // dequantE4M3). Decode through the LUT FIRST — the previous
            // code skipped this step and ran the matmul against the raw
            // uint8 byte values, producing weights ~33× too large.
            var wDecoded = take(fp8E4M3Lut, wArr.asType(.uint8))   // bf16

            if let s = self.scale, s.shape.count == 2 {
                // Block scale [outBlocks, inBlocks] (DeepSeek W8A8 layout).
                var sArr = s.array
                if s.dtype == .e8m0 {
                    sArr = take(e8m0Lut, sArr.asType(.uint8))     // bf16
                } else if sArr.dtype == .uint8 {
                    sArr = take(e8m0Lut, sArr)                     // bf16
                } else if sArr.dtype != .bfloat16 {
                    sArr = sArr.asType(.bfloat16)
                }

                let outBlocks = s.shape[0]
                let inBlocks = s.shape[1]
                let outBlockSize = outFeatures / outBlocks
                let inBlockSize = inFeatures / inBlocks

                let wReshaped = wDecoded
                    .reshaped([outBlocks, outBlockSize, inBlocks, inBlockSize])
                let sReshaped = sArr.reshaped([outBlocks, 1, inBlocks, 1])
                wDecoded = (wReshaped * sReshaped)
                    .reshaped([outFeatures, inFeatures])
            }
            // For FP8 with 1D scale (per-row) or no scale: leave wDecoded
            // as-is. callAsFunction's tail will multiply by the per-row
            // scale after the matmul.
            wArr = wDecoded
        }
        return wArr
    }

    public func callAsFunction(_ xIn: Tensor) -> Tensor {
        var xArr = xIn.array

        if let invScale = inverseChannelScale {
            xArr = xArr * invScale.array
        }

        // Fast path: pre-quantized triplet + fused MLX kernel. Avoids
        // the [outFeatures, inFeatures] bf16 dequant temporary entirely
        // (≈32 MB for a routed expert, larger for MLA matrices).
        //
        // Activated in two cases:
        //   1. MLX-native checkpoint: the triplet (.weight + .scales +
        //      .biases) is already on disk. Auto-detected via
        //      `hasMLXNativeTriplet()`, no env flag needed.
        //   2. Old custom-format checkpoint with DEEPSEEK_MLX_QUANT=1:
        //      Linear re-quantizes once on first use, caches.
        let preferMLXQuant = useMLXQuant || hasMLXNativeTriplet()
        if preferMLXQuant, let triple = getMLXQuant() {
            // MLXFast expects bf16/fp16 activations against the
            // quantized weight. Match the dtype of the scales.
            let xBf16 = xArr.dtype == .bfloat16 ? xArr : xArr.asType(.bfloat16)
            let yArr = MLXFast.quantizedMatmul(
                xBf16, triple.w,
                scales: triple.scales,
                biases: triple.biases,
                transpose: true,
                groupSize: 64,
                bits: 4)
            var outArr = yArr
            // Per-row scale (1D) is applied post-matmul — same as the
            // dequant path. 2D block scales are baked into the
            // requantized weight.
            if let s = scale, s.shape.count != 2 {
                let sBf16 = s.array.dtype == .bfloat16
                    ? s.array : s.array.asType(.bfloat16)
                outArr = outArr * sBf16
            }
            if castOutputToBF16 {
                if outArr.dtype != .bfloat16 { outArr = outArr.asType(.bfloat16) }
            } else if outArr.dtype != .float32 {
                outArr = outArr.asType(.float32)
            }
            return Tensor(array: outArr, dtype: castOutputToBF16 ? .bf16 : .f32)
        }

        let wArr = getDequantizedWeight()

        // When the dequant path produced bf16 (i.e. the weight is
        // quantized) run the matmul in bf16: that keeps the dequant
        // temporary at its bf16 size instead of upcasting back to f32
        // here. Unquantized weights (gate, embeddings projections,
        // hc params) keep their original dtype path so precision-
        // sensitive paths like gate routing aren't perturbed.
        let computeDtype: MLX.DType = wArr.dtype == .bfloat16
            ? .bfloat16
            : xArr.dtype
        let xComp = xArr.dtype == computeDtype ? xArr : xArr.asType(computeDtype)
        let wComp = wArr.dtype == computeDtype ? wArr : wArr.asType(computeDtype)

        // In FP4, the last dimension is inFeatures / 2 because of packing.
        let isFP4 = weightDType == .fp4E2M1
        let transposeNeeded = isFP4
            ? (self.weight.array.shape.last == inFeatures / 2)
            : (wComp.shape.last == inFeatures)

        let yArr: MLXArray = transposeNeeded
            ? matmul(xComp, wComp.transposed())
            : matmul(xComp, wComp)

        var outArr = yArr
        if !isFP4, let s = scale, s.shape.count != 2 {
            let sArr = s.array.dtype == outArr.dtype
                ? s.array
                : s.array.asType(outArr.dtype)
            outArr = outArr * sArr
        }

        // Honor the public contract: f32 by default, bf16 only if
        // castOutputToBF16 was set. Downstream layers (RMSNorm,
        // sinkhorn, sampling) expect f32.
        if castOutputToBF16 {
            if outArr.dtype != .bfloat16 { outArr = outArr.asType(.bfloat16) }
        } else if outArr.dtype != .float32 {
            outArr = outArr.asType(.float32)
        }

        return Tensor(array: outArr, dtype: castOutputToBF16 ? .bf16 : .f32)
    }
}
