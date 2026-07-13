import Foundation
import DS4Core

// MARK: - Little-endian Data helpers

extension Data {
    mutating func appendLE(_ v: UInt32) { var le = v.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) } }
    mutating func appendLE(_ v: UInt16) { var le = v.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) } }
    mutating func appendLE(_ v: UInt64) { var le = v.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) } }

    func readLE(_ o: inout Index) -> UInt32 {
        var r: UInt32 = 0
        for i in 0..<4 { r |= UInt32(self[o + i]) << (8 * i) }
        o += 4
        return r
    }

    func readLE(_ o: inout Index) -> UInt16 {
        var r: UInt16 = 0
        for i in 0..<2 { r |= UInt16(self[o + i]) << (8 * i) }
        o += 2
        return r
    }

    func readLE(_ o: inout Index) -> UInt64 {
        var r: UInt64 = 0
        for i in 0..<8 { r |= UInt64(self[o + i]) << (8 * i) }
        o += 8
        return r
    }
}

