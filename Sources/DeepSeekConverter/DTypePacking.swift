import Foundation

// Pure dtype-encoding helpers shared between the historical CLI
// converter and the new library entry points. Each function is
// `@inline(__always)` to keep hot loops tight when called inside
// per-row fusion / quantization workers.

@inline(__always)
public func floatToBF16(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
    return UInt16(truncatingIfNeeded: rounded >> 16)
}

@inline(__always)
public func floatToF16(_ f: Float) -> UInt16 {
    // IEEE 754 single → half with round-to-nearest-even, saturating to
    // ±inf on overflow. Subnormals are flushed via the standard formula.
    let bits = f.bitPattern
    let sign = (bits >> 31) & 1
    let exp = (bits >> 23) & 0xFF
    let mant = bits & 0x7FFFFF
    if exp == 0 { return UInt16(truncatingIfNeeded: sign << 15) }
    if exp == 0xFF {
        let m: UInt32 = mant != 0 ? 0x200 : 0   // NaN if mantissa, else inf
        return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10) | m)
    }
    let unbiased = Int(exp) - 127
    if unbiased > 15 { return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10)) }
    if unbiased < -14 {
        let shift = -14 - unbiased + 13
        if shift > 24 { return UInt16(truncatingIfNeeded: sign << 15) }
        let full = (mant | 0x800000) >> (shift - 1)
        let halfMant = (full + 1) >> 1
        return UInt16(truncatingIfNeeded: (sign << 15) | halfMant)
    }
    let halfExp = UInt32(unbiased + 15)
    let halfMant = (mant + 0x1000) >> 13
    if halfMant >= 0x400 {
        if halfExp + 1 >= 0x1F {
            return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10))
        }
        return UInt16(truncatingIfNeeded: (sign << 15) | ((halfExp + 1) << 10))
    }
    return UInt16(truncatingIfNeeded: (sign << 15) | (halfExp << 10) | halfMant)
}

/// Pack one float into 16 bits using the BF16 or F16 path of
/// `target`. INT* targets are not handled here — they go through
/// the dedicated quantizers in DeepSeekKit (Int{8,4,2}Quant).
@inline(__always)
public func packNative(_ f: Float, _ target: ConversionTarget) -> UInt16 {
    switch target {
    case .bf16: return floatToBF16(f)
    case .f16:  return floatToF16(f)
    case .int8: fatalError("packNative requires bf16 or f16; INT8 path uses Int8Quant")
    case .int4: fatalError("packNative requires bf16 or f16; INT4 path uses Int4Quant")
    case .int2: fatalError("packNative requires bf16 or f16; INT2 path uses Int2Quant")
    case .keep: fatalError("packNative requires bf16 or f16")
    }
}

@inline(__always)
public func deqE4M3(_ b: UInt8) -> Float {
    let sign = (UInt32(b) >> 7) & 1
    let exp = (UInt32(b) >> 3) & 0xF
    let mant = UInt32(b) & 0x7
    if exp == 0 && mant == 0 { return sign == 1 ? -0.0 : 0.0 }
    if exp == 0xF && mant == 0x7 { return .nan }
    if exp == 0 {
        let v = Float(mant) * 0x1p-9
        return sign == 1 ? -v : v
    }
    let bp = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return Float(bitPattern: bp)
}

@inline(__always)
public func deqE2M1(_ nibble: UInt8) -> Float {
    let mag = nibble & 7
    let v: Float
    switch mag {
    case 0: v = 0.0
    case 1: v = 0.5
    case 2: v = 1.0
    case 3: v = 1.5
    case 4: v = 2.0
    case 5: v = 3.0
    case 6: v = 4.0
    default: v = 6.0
    }
    return (nibble & 8) != 0 ? -v : v
}

@inline(__always)
public func deqE8M0(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    return Float(bitPattern: UInt32(b) << 23)
}

/// Pre-computed lookup tables for the dequant functions. Each input
/// byte (or nibble) maps to a fixed Float, so we replace per-element
/// bit twiddling with a single load. 256-entry tables fit in L1.
/// Marked `nonisolated(unsafe)` because they're read-only after the
/// module loads.
public let e4m3LUT: [Float] = (0..<256).map { deqE4M3(UInt8($0)) }
public let e2m1LUT: [Float] = (0..<16).map  { deqE2M1(UInt8($0)) }
public let e8m0LUT: [Float] = (0..<256).map { deqE8M0(UInt8($0)) }

@inline(__always)
public func isFP8DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F8_E4M3" || u == "F8E4M3" || u == "FLOAT8_E4M3FN"
}

@inline(__always)
public func isFP4DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F4_E2M1" || u == "F4E2M1" || u == "FLOAT4_E2M1FN_X2"
}

@inline(__always)
public func isE8M0DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F8_E8M0" || u == "F8E8M0" || u == "FLOAT8_E8M0FNU"
}
