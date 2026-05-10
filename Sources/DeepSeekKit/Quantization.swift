import Foundation

/// Quantization layouts used by DeepSeek-V4. These match the reference in
/// `Original/DeepSeek-V4-Pro/inference/kernel.py` and the FP4/FP8 mixed
/// checkpoint format. Apple Metal does not support FP8 or FP4 natively, so
/// every GEMM that reads these dtypes must dequantize through FP16/FP32 in
/// the shader before computing.
public enum Quant {
    /// FP8-E4M3 weight blocks: 128×128 with one E8M0 scale per block.
    /// Layout in safetensors:
    ///   weight: [out, in]   in fp8_e4m3
    ///   scale:  [ceil(out/128), ceil(in/128)]  in float8_e8m0fnu
    public enum FP8 {
        public static let blockOut = 128
        public static let blockIn = 128
    }

    /// FP4-E2M1 weight blocks: 1×32 with one E8M0 scale per block.
    /// Layout in safetensors:
    ///   weight: [out, in/2]  packed (two FP4 per byte, packed along K)
    ///   scale:  [out, in/32] in float8_e8m0fnu
    public enum FP4 {
        public static let blockOut = 1
        public static let blockIn = 32
    }

    /// FP8 activation quant block size used by `act_quant` along the K axis.
    public static let actBlockSizeFP8 = 128
    /// FP4 activation quant block size used by `fp4_act_quant`.
    public static let actBlockSizeFP4 = 32

    /// FP8-E4M3 numeric range.
    public static let fp8Min: Float = -448.0
    public static let fp8Max: Float = 448.0
    /// FP4-E2M1 numeric range.
    public static let fp4Max: Float = 6.0
}

/// E4M3 dequantization: 1 sign + 4 exponent + 3 mantissa, bias 7.
@inlinable
public func dequantE4M3(_ b: UInt8) -> Float {
    let s: UInt32 = UInt32((b >> 7) & 1)
    let e: UInt32 = UInt32((b >> 3) & 0xF)
    let m: UInt32 = UInt32(b & 0x7)
    if e == 0 && m == 0 { return s == 1 ? -0.0 : 0.0 }
    if e == 0xF && m == 0x7 { return Float.nan }
    let bits: UInt32
    if e == 0 {
        // subnormal: value = m * 2^(1 - bias - 3) = m * 2^-9
        let f = Float(m) * 0x1p-9
        return s == 1 ? -f : f
    }
    bits = (s << 31) | ((e &- 7 &+ 127) << 23) | (m << 20)
    return Float(bitPattern: bits)
}

/// E2M1 dequantization (one nibble): 1 sign + 2 exponent + 1 mantissa, bias 1.
/// Values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6.
@inlinable
public func dequantE2M1(_ nibble: UInt8) -> Float {
    let s = (nibble >> 3) & 1
    let e = (nibble >> 1) & 0x3
    let m = nibble & 0x1
    let mag: Float
    switch (e, m) {
    case (0, 0): mag = 0
    case (0, 1): mag = 0.5
    case (1, 0): mag = 1.0
    case (1, 1): mag = 1.5
    case (2, 0): mag = 2.0
    case (2, 1): mag = 3.0
    case (3, 0): mag = 4.0
    case (3, 1): mag = 6.0
    default: mag = 0
    }
    return s == 1 ? -mag : mag
}

/// E8M0 dequantization: unsigned 8-bit unbiased exponent, value = 2^(b - 127).
/// 0xFF is reserved as NaN by the MX spec.
@inlinable
public func dequantE8M0(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    let bits: UInt32 = UInt32(b) << 23
    return Float(bitPattern: bits)
}
