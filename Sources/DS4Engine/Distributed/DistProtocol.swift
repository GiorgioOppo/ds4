import Foundation
import DS4Core

/// Wire protocol for DwarfStar distributed inference (pipeline parallelism by
/// contiguous layer ranges), modelled on ds4_distributed.c but Swift-native:
/// every node runs the same DwarfStar build, so the framing is our own (no C
/// byte-compatibility needed).
///
/// Topology: a COORDINATOR owns the embedding, the sampling loop and the API/UI.
/// Each WORKER owns a contiguous layer slice `[start, end]` and its KV/compressor
/// shard. Per token the coordinator embeds → sends the hidden (HC) state to the
/// first worker → each worker runs its layers and forwards the HC state → the
/// coordinator runs the output head and samples. The HC state is `nHC*nEmbd`
/// floats; it can be transported at 32/16/8-bit width to save bandwidth.
public enum Dist {
    /// Bump when the framing or semantics change incompatibly.
    public static let protocolVersion: UInt32 = 1
    static let magic: UInt32 = 0x44_53_34_44   // "DS4D"

    public enum MsgType: UInt32, Sendable {
        case hello   = 1    // worker → coordinator on connect (slice + model identity)
        case work    = 3    // coordinator → worker: embed/HC input for a layer slice
        case result  = 4    // worker → coordinator: HC state or logits
        case error   = 2
    }

    /// Flags on a WORK message.
    public struct WorkFlags: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        public static let resetSession = WorkFlags(rawValue: 1 << 0)  // pos==0: reset compressor/KV
        public static let outputLogits = WorkFlags(rawValue: 1 << 1)  // last slice: this worker also runs the head
    }

    public enum ResultKind: UInt32, Sendable { case hidden = 0, logits = 1, ack = 2 }
}

// MARK: - Frames

/// Fixed header preceding every framed message: magic + type + payload length.
public struct DistFrameHeader {
    public static let byteSize = 12
    public var type: Dist.MsgType
    public var length: UInt32     // payload bytes following the header

    public func encoded() -> Data {
        var d = Data(capacity: DistFrameHeader.byteSize)
        d.appendLE(Dist.magic)
        d.appendLE(type.rawValue)
        d.appendLE(length)
        return d
    }

    public static func decode(_ d: Data) -> DistFrameHeader? {
        guard d.count >= byteSize else { return nil }
        var o = d.startIndex
        guard d.readLE(&o) == Dist.magic,
              let type = Dist.MsgType(rawValue: d.readLE(&o)) else { return nil }
        let length = d.readLE(&o) as UInt32
        return DistFrameHeader(type: type, length: length)
    }
}

/// HELLO payload: a worker announces its model identity and the slice it serves.
public struct DistHello: Sendable {
    public var modelName: String
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // also owns the output head
    public var nLayers: Int
    public var contextSize: Int

    public init(modelName: String, layerStart: Int, layerEnd: Int, hasOutput: Bool,
                nLayers: Int, contextSize: Int) {
        self.modelName = modelName; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.hasOutput = hasOutput; self.nLayers = nLayers; self.contextSize = contextSize
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        d.appendLE(UInt32(nLayers)); d.appendLE(UInt32(contextSize))
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        return d
    }

    public static func decode(_ d: Data) -> DistHello? {
        var o = d.startIndex
        guard d.count >= 24 else { return nil }
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let nl = Int(d.readLE(&o) as UInt32), ctx = Int(d.readLE(&o) as UInt32)
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard o + nameLen <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self)
        return DistHello(modelName: name, layerStart: ls, layerEnd: le, hasOutput: ho,
                         nLayers: nl, contextSize: ctx)
    }
}

/// WORK payload: `nTokens` consecutive tokens' HC states (concatenated) to
/// evaluate through `[layerStart, layerEnd]` starting at absolute position `pos`.
/// When `route` is non-empty the workers forward the result downstream
/// (worker→worker) and the terminal worker replies to `returnHost:returnPort`;
/// when empty, each worker replies on the same connection (coordinator relay).
public struct DistWork: Sendable {
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

    public init(pos: Int, nTokens: Int, layerStart: Int, layerEnd: Int, flags: Dist.WorkFlags,
                hcBits: Int, route: [DistRouteEntry] = [], routeIndex: Int = 0,
                returnHost: String = "", returnPort: UInt16 = 0, hc: [Float]) {
        self.pos = pos; self.nTokens = nTokens; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.flags = flags; self.hcBits = hcBits; self.route = route; self.routeIndex = routeIndex
        self.returnHost = returnHost; self.returnPort = returnPort; self.hc = hc
    }

    public func encoded() -> Data {
        var d = Data()
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
        d.appendLE(UInt32(hc.count))
        d.append(ActivationCodec.pack(hc, bits: hcBits))
        return d
    }

    public static func decode(_ d: Data) -> DistWork? {
        var o = d.startIndex
        guard d.count >= 32 else { return nil }
        let pos = Int(Int32(bitPattern: d.readLE(&o)))
        let nTokens = Int(d.readLE(&o) as UInt32)
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let flags = Dist.WorkFlags(rawValue: d.readLE(&o))
        let bits = Int(d.readLE(&o) as UInt32)
        let routeCount = Int(d.readLE(&o) as UInt32)
        let routeIndex = Int(d.readLE(&o) as UInt32)
        var route: [DistRouteEntry] = []
        for _ in 0..<routeCount {
            guard let e = DistRouteEntry.decode(d, &o) else { return nil }
            route.append(e)
        }
        guard o + 4 <= d.endIndex else { return nil }
        let rhLen = Int(d.readLE(&o) as UInt32)
        guard o + rhLen + 8 <= d.endIndex else { return nil }
        let returnHost = String(decoding: d[o..<o+rhLen], as: UTF8.self); o += rhLen
        let returnPort = UInt16(clamping: d.readLE(&o) as UInt32)
        let count = Int(d.readLE(&o) as UInt32)
        let hc = count == 0 ? [] : ActivationCodec.unpack(Data(d[o..<d.endIndex]), count: count, bits: bits)
        return DistWork(pos: pos, nTokens: nTokens, layerStart: ls, layerEnd: le, flags: flags,
                        hcBits: bits, route: route, routeIndex: routeIndex,
                        returnHost: returnHost, returnPort: returnPort, hc: hc)
    }
}

/// RESULT payload: the produced HC state (forward to the next slice) or final logits.
public struct DistResult: Sendable {
    public var kind: Dist.ResultKind
    public var bits: Int
    public var values: [Float]

    public init(kind: Dist.ResultKind, bits: Int, values: [Float]) {
        self.kind = kind; self.bits = bits; self.values = values
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(kind.rawValue)
        d.appendLE(UInt32(bits))
        d.appendLE(UInt32(values.count))
        d.append(ActivationCodec.pack(values, bits: bits))
        return d
    }

    public static func decode(_ d: Data) -> DistResult? {
        var o = d.startIndex
        guard d.count >= 12, let kind = Dist.ResultKind(rawValue: d.readLE(&o)) else { return nil }
        let bits = Int(d.readLE(&o) as UInt32)
        let count = Int(d.readLE(&o) as UInt32)
        let values = count == 0 ? [] : ActivationCodec.unpack(Data(d[o..<d.endIndex]), count: count, bits: bits)
        return DistResult(kind: kind, bits: bits, values: values)
    }
}

// MARK: - Activation transport codec (32 / 16 / 8 bit)

/// Packs/unpacks a float activation vector at 32, 16 (float16) or 8 (per-vector
/// scaled int8) bits. 8-bit uses a single absmax scale prepended as Float32.
public enum ActivationCodec {
    public static func pack(_ v: [Float], bits: Int) -> Data {
        switch bits {
        case 16:
            var d = Data(capacity: v.count * 2)
            for x in v { d.appendLE(Half.bits(x)) }
            return d
        case 8:
            let absmax = v.reduce(Float(0)) { max($0, abs($1)) }
            let scale = absmax > 0 ? absmax / 127.0 : 1
            var d = Data(capacity: 4 + v.count)
            d.appendLE(scale.bitPattern)
            for x in v {
                let q = Int(( x / scale).rounded())
                d.append(UInt8(bitPattern: Int8(clamping: q)))
            }
            return d
        default: // 32
            var d = Data(capacity: v.count * 4)
            for x in v { d.appendLE(x.bitPattern) }
            return d
        }
    }

    public static func unpack(_ d: Data, count: Int, bits: Int) -> [Float] {
        var o = d.startIndex
        var out = [Float](); out.reserveCapacity(count)
        switch bits {
        case 16:
            for _ in 0..<count {
                guard o + 2 <= d.endIndex else { break }
                out.append(Half.float(d.readLE(&o) as UInt16))
            }
        case 8:
            guard o + 4 <= d.endIndex else { return out }
            let scale = Float(bitPattern: d.readLE(&o) as UInt32)
            for _ in 0..<count {
                guard o < d.endIndex else { break }
                let q = Int8(bitPattern: d[o]); o += 1
                out.append(Float(q) * scale)
            }
        default:
            for _ in 0..<count {
                guard o + 4 <= d.endIndex else { break }
                out.append(Float(bitPattern: d.readLE(&o) as UInt32))
            }
        }
        return out
    }
}

// MARK: - Little-endian Data helpers

extension Data {
    mutating func appendLE(_ v: UInt32) { var le = v.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) } }
    mutating func appendLE(_ v: UInt16) { var le = v.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) } }

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
}
