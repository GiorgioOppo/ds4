import Foundation

/// Portable Float ⇄ IEEE-754 binary16 conversion.
///
/// On arm64 this is the hardware `Float16` (exact, fast). The Swift stdlib does
/// NOT provide `Float16` on macOS x86_64, so building any Intel slice (an
/// "Any Mac" destination, an Archive without pinned ARCHS) fails with
/// misleading errors ("Float16 has no member bitPattern"). This shim keeps the
/// engine compiling everywhere; the software fallback implements correct
/// round-to-nearest-even, matching the hardware conversion bit-for-bit.
public enum Half {
    /// Float32 → binary16 bit pattern (round-to-nearest-even).
    @inline(__always)
    public static func bits(_ x: Float) -> UInt16 {
        #if arch(arm64)
        return Float16(x).bitPattern
        #else
        return bitsSoftware(x)
        #endif
    }

    /// binary16 bit pattern → Float32 (exact).
    @inline(__always)
    public static func float(_ h: UInt16) -> Float {
        #if arch(arm64)
        return Float(Float16(bitPattern: h))
        #else
        return floatSoftware(h)
        #endif
    }

    // MARK: software fallback (exposed internally for tests)

    static func bitsSoftware(_ value: Float) -> UInt16 {
        let sign = UInt16((value.bitPattern >> 16) & 0x8000)
        let f = abs(value)
        if f.isNaN { return sign | 0x7E00 }                    // quiet NaN
        if f >= 65520 { return sign | 0x7C00 }                 // ≥ RNE midpoint → ±inf
        if f < 0x1p-14 {                                       // subnormal / zero
            // h = round(f · 2^24) ties-to-even; 1024 rolls into the smallest normal.
            return sign | UInt16((f * 0x1p24).rounded(.toNearestOrEven))
        }
        // Normal: rebias exponent (Int math: the field is ≥113 here, −112 ≥ 1),
        // then round 23→10 mantissa bits ties-to-even.
        let b = f.bitPattern
        let e = UInt32(Int((b >> 23) & 0xFF) - 112)
        let m = b & 0x7F_FFFF
        var h = (e << 10) | (m >> 13)
        let rem = m & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1 }   // carry-safe
        return sign | UInt16(h)
    }

    static func floatSoftware(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h >> 10) & 0x1F
        let mant = UInt32(h & 0x3FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }    // ±0
            let mag = Float(mant) * 0x1p-24                    // subnormal
            return (h & 0x8000) != 0 ? -mag : mag
        }
        if exp == 0x1F {                                       // ±inf / NaN
            return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13))
        }
        return Float(bitPattern: sign | ((exp - 15 + 127) << 23) | (mant << 13))
    }
}
