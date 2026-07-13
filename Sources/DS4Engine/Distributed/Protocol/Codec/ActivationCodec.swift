import Foundation
import DS4Core

// MARK: - Activation transport codec (32 / 16 / 8 bit)

/// Packs/unpacks a float activation vector at 32, 16 (float16) or 8 (per-vector
/// scaled int8) bits. 8-bit uses a single absmax scale prepended as Float32.
///
/// This is the WIRE HOT PATH (an HC state is tens of thousands of floats, per
/// chunk, per hop): everything moves as bulk buffer copies on the little-endian
/// arm64 host — never per-element Data appends/reads. `unpack` is STRICT: it
/// returns nil unless the payload holds exactly `count` values, so a truncated
/// frame is rejected at decode instead of surfacing as a short array that
/// crashes the consumer's slicing.
public enum ActivationCodec {
    public static func pack(_ v: [Float], bits: Int) -> Data {
        switch bits {
        case 16:
            var half = [UInt16](repeating: 0, count: v.count)
            for i in v.indices { half[i] = Half.bits(v[i]) }
            return half.withUnsafeBufferPointer { Data(buffer: $0) }
        case 8:
            let absmax = v.reduce(Float(0)) { max($0, abs($1)) }
            let scale = absmax > 0 ? absmax / 127.0 : 1
            var d = Data(capacity: 4 + v.count)
            d.appendLE(scale.bitPattern)
            var q = [UInt8](repeating: 0, count: v.count)
            for i in v.indices { q[i] = UInt8(bitPattern: Int8(clamping: Int((v[i] / scale).rounded()))) }
            d.append(contentsOf: q)
            return d
        default: // 32
            return v.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }

    public static func unpack(_ d: Data, count: Int, bits: Int) -> [Float]? {
        guard count >= 0 else { return nil }
        if count == 0 { return [] }
        switch bits {
        case 16:
            guard d.count >= count * 2 else { return nil }
            return d.withUnsafeBytes { raw -> [Float] in
                let src = raw.baseAddress!
                var half = [UInt16](repeating: 0, count: count)
                half.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, src, count * 2) }
                var out = [Float](repeating: 0, count: count)
                for i in 0..<count { out[i] = Half.float(half[i]) }
                return out
            }
        case 8:
            guard d.count >= 4 + count else { return nil }
            return d.withUnsafeBytes { raw -> [Float] in
                let scale = Float(bitPattern: raw.loadUnaligned(as: UInt32.self))
                let bytes = raw.baseAddress! + 4
                var out = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    out[i] = Float(Int8(bitPattern: bytes.load(fromByteOffset: i, as: UInt8.self))) * scale
                }
                return out
            }
        default:
            guard d.count >= count * 4 else { return nil }
            return d.withUnsafeBytes { raw -> [Float] in
                var out = [Float](repeating: 0, count: count)
                out.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, raw.baseAddress!, count * 4) }
                return out
            }
        }
    }
}

