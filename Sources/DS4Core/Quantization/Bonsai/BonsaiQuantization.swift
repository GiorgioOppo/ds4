import Foundation

/// Reference codecs for the packed Prism weights used by Ternary Bonsai 2.
/// The GPU consumes these bytes directly; no full model dequantization occurs.
public enum BonsaiQuantization {
    public static func rowBytes(type: UInt32, columns: Int) -> Int? {
        guard columns > 0, columns <= Int(UInt32.max) / 4 else { return nil }
        switch type {
        case 0: return columns * 4
        case 1, 30: return columns * 2
        case 142 where columns % 128 == 0: return columns / 128 * 34
        case 143 where columns % 128 == 0: return columns / 128 * 28
        default: return nil
        }
    }

    public static func decodeRow(_ bytes: [UInt8], type: UInt32, columns: Int) throws -> [Float] {
        guard let size = rowBytes(type: type, columns: columns), bytes.count == size else {
            throw GGUFError.message("Bonsai: invalid packed row length or encoding")
        }
        func u16(_ i: Int) -> UInt16 { UInt16(bytes[i]) | UInt16(bytes[i + 1]) << 8 }
        return (0..<columns).map { i in
            switch type {
            case 0:
                let p = i * 4
                let bits = UInt32(bytes[p]) | UInt32(bytes[p+1]) << 8 | UInt32(bytes[p+2]) << 16 | UInt32(bytes[p+3]) << 24
                return Float(bitPattern: bits)
            case 1: return Float(Float16(bitPattern: u16(i * 2)))
            case 30: return Float(bitPattern: UInt32(u16(i * 2)) << 16)
            case 142:
                let p = i / 128 * 34, j = i % 128
                let code = (Int(bytes[p + 2 + j / 4]) >> (2 * (j % 4))) & 3
                return Float(Float16(bitPattern: u16(p))) * Float(code - 1)
            default:
                let p = i / 128 * 28, j = i % 128
                let byte: Int, trit: Int
                if j < 80 { byte = j % 16; trit = j / 16 }
                else if j < 120 { byte = 16 + (j - 80) % 8; trit = (j - 80) / 8 }
                else { byte = 24 + (j - 120) % 2; trit = (j - 120) / 2 }
                // PTQ wraps its byte product BEFORE extraction. Modulo 243
                // or an ordinary base-three digit decoder is not equivalent.
                let wrapped = (Int(bytes[p + byte]) * [1, 3, 9, 27, 81][trit]) & 255
                return Float(Float16(bitPattern: u16(p + 26))) * Float(((wrapped * 3) >> 8) - 1)
            }
        }
    }

    public static func hadamard(_ input: [Float], signs: [Int32], inverse: Bool = false) throws -> [Float] {
        guard !input.isEmpty, input.count % 1024 == 0, signs.count == input.count,
              signs.allSatisfy({ $0 == 1 || $0 == -1 }), input.allSatisfy(\.isFinite) else {
            throw GGUFError.message("Bonsai: invalid normalized Hadamard input/signs")
        }
        var output = input
        if !inverse { for i in output.indices { output[i] *= Float(signs[i]) } }
        for base in stride(from: 0, to: input.count, by: 1024) {
            var step = 1
            while step < 1024 {
                for start in stride(from: base, to: base + 1024, by: step * 2) {
                    for i in start..<start + step {
                        let a = output[i], b = output[i + step]
                        output[i] = a + b; output[i + step] = a - b
                    }
                }
                step *= 2
            }
        }
        for i in output.indices { output[i] = output[i] * (1 / 32) * (inverse ? Float(signs[i]) : 1) }
        return output
    }
}
