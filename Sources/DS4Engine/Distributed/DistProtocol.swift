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
    /// v2: HELLO carries the version; WORK/RESULT carry a per-turn `session`
    /// id echoed by the workers, so a result left in a TCP buffer by a
    /// cancelled turn can never be mistaken for the next turn's reply.
    /// v3: the COORDINATOR defines each worker's job. Workers start idle
    /// (listening, no model loaded); the coordinator sends ASSIGN (gguf,
    /// context, layer slice, cache slots) and the worker replies READY once
    /// its engine is loaded. HELLO gained the `assigned` state.
    public static let protocolVersion: UInt32 = 3
    static let magic: UInt32 = 0x44_53_34_44   // "DS4D"
    /// Sanity cap on the route length in a WORK frame (a hostile frame could
    /// otherwise declare 4G entries and spin the decoder).
    static let maxRouteEntries = 256

    public enum MsgType: UInt32, Sendable {
        case hello   = 1    // worker → coordinator on connect (state + model identity)
        case work    = 3    // coordinator → worker: embed/HC input for a layer slice
        case result  = 4    // worker → coordinator: HC state or logits
        case error   = 2
        case assign  = 5    // coordinator → worker: gguf + settings + layer slice
        case ready   = 6    // worker → coordinator: assignment loaded (HELLO payload)
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

/// HELLO payload: a worker announces its protocol version, its state
/// (`assigned` — an engine is loaded for a slice) and, when assigned, the model
/// identity and slice it serves. Sent on connect and echoed as the READY
/// payload after an ASSIGN completes. The coordinator validates the version
/// FIRST — a mixed cluster fails with a clear error instead of garbled frames.
public struct DistHello: Sendable {
    public var version: UInt32
    public var assigned: Bool         // an engine is loaded (slice fields valid)
    public var modelName: String      // loaded gguf (assigned) or local hint (idle)
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // also owns the output head
    public var nLayers: Int
    public var contextSize: Int       // 0 while idle

    public init(modelName: String, layerStart: Int, layerEnd: Int, hasOutput: Bool,
                nLayers: Int, contextSize: Int, assigned: Bool = true,
                version: UInt32 = Dist.protocolVersion) {
        self.version = version
        self.assigned = assigned
        self.modelName = modelName; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.hasOutput = hasOutput; self.nLayers = nLayers; self.contextSize = contextSize
    }

    /// An unconfigured worker waiting for an ASSIGN.
    public static func idle(localModelName: String, nLayers: Int) -> DistHello {
        DistHello(modelName: localModelName, layerStart: 0, layerEnd: 0, hasOutput: false,
                  nLayers: nLayers, contextSize: 0, assigned: false)
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(version)
        d.appendLE(UInt32(assigned ? 1 : 0))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        d.appendLE(UInt32(nLayers)); d.appendLE(UInt32(contextSize))
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        return d
    }

    public static func decode(_ d: Data) -> DistHello? {
        var o = d.startIndex
        guard d.count >= 32 else { return nil }
        let ver = d.readLE(&o) as UInt32
        let assigned = (d.readLE(&o) as UInt32) != 0
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let nl = Int(d.readLE(&o) as UInt32), ctx = Int(d.readLE(&o) as UInt32)
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen >= 0, o + nameLen <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self)
        return DistHello(modelName: name, layerStart: ls, layerEnd: le, hasOutput: ho,
                         nLayers: nl, contextSize: ctx, assigned: assigned, version: ver)
    }
}

/// ASSIGN payload: the coordinator defines a worker's whole job — WHICH gguf
/// (full coordinator-side path + filename, resolved locally by the worker),
/// the context size, the expert-cache budget, and the layer slice to own.
/// The worker loads (or reuses) its engine and replies READY (HELLO payload).
public struct DistAssign: Sendable {
    public var modelPath: String      // coordinator's path (verbatim, tried first)
    public var modelName: String      // gguf filename (fallback resolution key)
    public var contextSize: Int
    public var expertCacheSlots: Int  // 0 = no expert slot-cache on the worker
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // last slice: also runs the output head

    public init(modelPath: String, modelName: String, contextSize: Int, expertCacheSlots: Int,
                layerStart: Int, layerEnd: Int, hasOutput: Bool) {
        self.modelPath = modelPath; self.modelName = modelName
        self.contextSize = contextSize; self.expertCacheSlots = expertCacheSlots
        self.layerStart = layerStart; self.layerEnd = layerEnd; self.hasOutput = hasOutput
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(contextSize))
        d.appendLE(UInt32(expertCacheSlots))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        let path = Data(modelPath.utf8)
        d.appendLE(UInt32(path.count)); d.append(path)
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        return d
    }

    public static func decode(_ d: Data) -> DistAssign? {
        var o = d.startIndex
        guard d.count >= 28 else { return nil }
        let ctx = Int(d.readLE(&o) as UInt32)
        let slots = Int(d.readLE(&o) as UInt32)
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let pathLen = Int(d.readLE(&o) as UInt32)
        guard pathLen >= 0, o + pathLen + 4 <= d.endIndex else { return nil }
        let path = String(decoding: d[o..<o+pathLen], as: UTF8.self); o += pathLen
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen >= 0, o + nameLen <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self)
        return DistAssign(modelPath: path, modelName: name, contextSize: ctx,
                          expertCacheSlots: slots, layerStart: ls, layerEnd: le, hasOutput: ho)
    }
}

/// WORK payload: `nTokens` consecutive tokens' HC states (concatenated) to
/// evaluate through `[layerStart, layerEnd]` starting at absolute position `pos`.
/// When `route` is non-empty the workers forward the result downstream
/// (worker→worker) and the terminal worker replies to `returnHost:returnPort`;
/// when empty, each worker replies on the same connection (coordinator relay).
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

    public init(session: UInt32, pos: Int, nTokens: Int, layerStart: Int, layerEnd: Int,
                flags: Dist.WorkFlags, hcBits: Int, route: [DistRouteEntry] = [], routeIndex: Int = 0,
                returnHost: String = "", returnPort: UInt16 = 0, hc: [Float]) {
        self.session = session
        self.pos = pos; self.nTokens = nTokens; self.layerStart = layerStart; self.layerEnd = layerEnd
        self.flags = flags; self.hcBits = hcBits; self.route = route; self.routeIndex = routeIndex
        self.returnHost = returnHost; self.returnPort = returnPort; self.hc = hc
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
        let count = Int(d.readLE(&o) as UInt32)
        // Strict: a chunk whose payload does not hold EXACTLY `count` values is
        // rejected here, so no consumer ever slices a silently-short array.
        guard let hc = ActivationCodec.unpack(Data(d[o..<d.endIndex]), count: count, bits: bits) else {
            return nil
        }
        return DistWork(session: session, pos: pos, nTokens: nTokens, layerStart: ls, layerEnd: le,
                        flags: flags, hcBits: bits, route: route, routeIndex: routeIndex,
                        returnHost: returnHost, returnPort: returnPort, hc: hc)
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
