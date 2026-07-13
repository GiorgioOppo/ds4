import Foundation
import DS4Core

public struct DistWork: Sendable {
    public var session: UInt32        // per-turn id, echoed by workers in RESULT
    public var pos: Int               // absolute position of the FIRST token
    public var nTokens: Int           // tokens in this chunk (hc holds nTokens states)
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var flags: Dist.WorkFlags
    public var hcBits: Int            // 32/16/8
    public var route: [DistRouteEntry]
    public var routeIndex: Int        // which route entry THIS work is for
    public var returnHost: String
    public var returnPort: UInt16
    public var hc: [Float]            // nTokens * (nHC*nEmbd) floats
    public var tokenIds: [Int32]      // v7: the chunk's token ids (hash-layer routing)

    public init(session: UInt32, pos: Int, nTokens: Int, layerStart: Int, layerEnd: Int,
                flags: Dist.WorkFlags, hcBits: Int, route: [DistRouteEntry] = [], routeIndex: Int = 0,
                returnHost: String = "", returnPort: UInt16 = 0, hc: [Float], tokenIds: [Int32] = []) {
        self.session = session
        self.pos = pos; self.nTokens = nTokens; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.flags = flags; self.hcBits = hcBits; self.route = route; self.routeIndex = routeIndex
        self.returnHost = returnHost; self.returnPort = returnPort; self.hc = hc
        self.tokenIds = tokenIds
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(session)
        d.appendLE(UInt32(bitPattern: Int32(pos)))
        d.appendLE(UInt32(nTokens))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(flags.rawValue)
        d.appendLE(UInt32(hcBits))
        d.appendLE(UInt32(route.count)); d.appendLE(UInt32(routeIndex))
        for e in route { e.encode(into: &d) }
        let rh = Data(returnHost.utf8)
        d.appendLE(UInt32(rh.count)); d.append(rh)
        d.appendLE(UInt32(returnPort))
        d.appendLE(UInt32(tokenIds.count))
        for t in tokenIds { d.appendLE(UInt32(bitPattern: t)) }
        d.appendLE(UInt32(hc.count))
        d.append(ActivationCodec.pack(hc, bits: hcBits))
        return d
    }

    public static func decode(_ d: Data) -> DistWork? {
        var o = d.startIndex
        guard d.count >= 36 else { return nil }
        let session = d.readLE(&o) as UInt32
        let pos = Int(Int32(bitPattern: d.readLE(&o)))
        let nTokens = Int(d.readLE(&o) as UInt32)
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let flags = Dist.WorkFlags(rawValue: d.readLE(&o))
        let bits = Int(d.readLE(&o) as UInt32)
        let routeCount = Int(d.readLE(&o) as UInt32)
        let routeIndex = Int(d.readLE(&o) as UInt32)
        guard routeCount <= Dist.maxRouteEntries else { return nil }
        var route: [DistRouteEntry] = []
        for _ in 0..<routeCount {
            guard let e = DistRouteEntry.decode(d, &o) else { return nil }
            route.append(e)
        }
        guard o + 4 <= d.endIndex else { return nil }
        let rhLen = Int(d.readLE(&o) as UInt32)
        guard rhLen >= 0, o + rhLen + 8 <= d.endIndex else { return nil }
        let returnHost = String(decoding: d[o..<o+rhLen], as: UTF8.self); o += rhLen
        let returnPort = UInt16(clamping: d.readLE(&o) as UInt32)
        guard o + 4 <= d.endIndex else { return nil }
        // v7: token ids. Either absent (0 — legacy control paths) or exactly
        // one per token; anything else is a malformed frame.
        let idCount = Int(d.readLE(&o) as UInt32)
        guard idCount == 0 || idCount == nTokens, o + idCount * 4 + 4 <= d.endIndex else { return nil }
        var tokenIds: [Int32] = []
        tokenIds.reserveCapacity(idCount)
        for _ in 0..<idCount { tokenIds.append(Int32(bitPattern: d.readLE(&o) as UInt32)) }
        let count = Int(d.readLE(&o) as UInt32)
        // Strict: a chunk whose payload does not hold EXACTLY `count` values is
        // rejected here, so no consumer ever slices a silently-short array.
        guard let hc = ActivationCodec.unpack(Data(d[o..<d.endIndex]), count: count, bits: bits) else {
            return nil
        }
        return DistWork(session: session, pos: pos, nTokens: nTokens, layerStart: ls, layerEnd: le,
                        flags: flags, hcBits: bits, route: route, routeIndex: routeIndex,
                        returnHost: returnHost, returnPort: returnPort, hc: hc, tokenIds: tokenIds)
    }
}

/// RESULT payload: the produced HC state (forward to the next slice) or final
/// logits. `session` echoes the WORK's session id: the coordinator discards any
/// result from a turn it has already abandoned (Stop mid-chunk leaves the reply
/// in a TCP buffer — without the echo it would be read as the NEXT turn's answer).
public struct DistResult: Sendable {
    public var session: UInt32
    public var kind: Dist.ResultKind
    public var bits: Int
    public var values: [Float]

    public init(session: UInt32, kind: Dist.ResultKind, bits: Int, values: [Float]) {
        self.session = session; self.kind = kind; self.bits = bits; self.values = values
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(session)
        d.appendLE(kind.rawValue)
        d.appendLE(UInt32(bits))
        d.appendLE(UInt32(values.count))
        d.append(ActivationCodec.pack(values, bits: bits))
        return d
    }

    public static func decode(_ d: Data) -> DistResult? {
        var o = d.startIndex
        guard d.count >= 16 else { return nil }
        let session = d.readLE(&o) as UInt32
        guard let kind = Dist.ResultKind(rawValue: d.readLE(&o)) else { return nil }
        let bits = Int(d.readLE(&o) as UInt32)
        let count = Int(d.readLE(&o) as UInt32)
        guard let values = ActivationCodec.unpack(Data(d[o..<d.endIndex]), count: count, bits: bits) else {
            return nil
        }
        return DistResult(session: session, kind: kind, bits: bits, values: values)
    }
}

