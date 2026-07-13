import Foundation
import DS4Core

// MARK: - KV control payloads (kvQuery / kvLengths / kvRestore / kvSave / kvAck)

/// Payload helpers for the distributed disk-KV control frames. Token lists are
/// `u32 count + count × u32`; caps mirror the rest of the protocol (a hostile
/// count is rejected instead of allocating gigabytes).
public enum DistKV {
    static let maxTokens = 1_000_000
    static let maxLengths = 4096

    public static func encodeTokens(_ ids: [Int]) -> Data {
        var d = Data(capacity: 4 + ids.count * 4)
        d.appendLE(UInt32(ids.count))
        for t in ids { d.appendLE(UInt32(truncatingIfNeeded: t)) }
        return d
    }

    public static func decodeTokens(_ d: Data) -> [Int]? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let count = Int(d.readLE(&o) as UInt32)
        guard count >= 0, count <= maxTokens, o + count * 4 <= d.endIndex else { return nil }
        var ids = [Int](); ids.reserveCapacity(count)
        for _ in 0..<count { ids.append(Int(d.readLE(&o) as UInt32)) }
        return ids
    }

    /// kvSave payload: the token prefix to checkpoint + the eviction reason
    /// ("cold" marks a conversation's first checkpoint, 2× protected).
    public static func encodeSave(tokens: [Int], cold: Bool) -> Data {
        var d = Data()
        d.appendLE(UInt32(cold ? 1 : 0))
        d.append(encodeTokens(tokens))
        return d
    }

    public static func decodeSave(_ d: Data) -> (tokens: [Int], cold: Bool)? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let cold = (d.readLE(&o) as UInt32) != 0
        guard let tokens = decodeTokens(Data(d[o..<d.endIndex])) else { return nil }
        return (tokens, cold)
    }

    public static func encodeLengths(_ lengths: [Int]) -> Data {
        var d = Data(capacity: 4 + lengths.count * 4)
        d.appendLE(UInt32(lengths.count))
        for l in lengths { d.appendLE(UInt32(truncatingIfNeeded: l)) }
        return d
    }

    public static func decodeLengths(_ d: Data) -> [Int]? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let count = Int(d.readLE(&o) as UInt32)
        guard count >= 0, count <= maxLengths, o + count * 4 <= d.endIndex else { return nil }
        var out = [Int](); out.reserveCapacity(count)
        for _ in 0..<count { out.append(Int(d.readLE(&o) as UInt32)) }
        return out
    }

    public static func encodeAck(ok: Bool, message: String = "") -> Data {
        var d = Data()
        d.appendLE(UInt32(ok ? 1 : 0))
        let m = Data(message.utf8)
        d.appendLE(UInt32(m.count)); d.append(m)
        return d
    }

    public static func decodeAck(_ d: Data) -> (ok: Bool, message: String)? {
        var o = d.startIndex
        guard d.count >= 8 else { return nil }
        let ok = (d.readLE(&o) as UInt32) != 0
        let len = Int(d.readLE(&o) as UInt32)
        guard len >= 0, o + len <= d.endIndex else { return nil }
        return (ok, String(decoding: d[o..<o+len], as: UTF8.self))
    }
}

/// WORK payload: `nTokens` consecutive tokens' HC states (concatenated) to
/// evaluate through `[layerStart, layerEnd]` starting at absolute position `pos`.
/// When `route` is non-empty the workers forward the result downstream
/// (worker→worker) and the terminal worker replies to `returnHost:returnPort`;
/// when empty, each worker replies on the same connection (coordinator relay).

