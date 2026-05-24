import Foundation
import MLX
import MLXNN

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
