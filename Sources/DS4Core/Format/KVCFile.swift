import Foundation
import CryptoKit

// Faithful Swift port of the disk KV-cache file format and policy bits in
// ds4_kvstore.c: the 48-byte KVC header (fill/parse), little-endian helpers,
// SHA1 naming, reason/key-kind mapping, default options, and the base eviction
// score. Phase 5 (serialization) of the C->Swift port.
//
// Note: the heavy KV *tensor* payload (DSV4 body) is produced by the live
// inference graph, which DS4Core does not have yet; only the portable
// header/format layer is ported here (see DSV4PayloadHeader for the documented
// payload header). The cache directory management (refresh/evict I/O,
// supersedes-continued) is deferred to a later cache-management phase.

public enum KVCFile {
    public static let fixedHeader = 48
    public static let magic0: UInt8 = UInt8(ascii: "K")
    public static let magic1: UInt8 = UInt8(ascii: "V")
    public static let magic2: UInt8 = UInt8(ascii: "C")
    public static let version: UInt8 = 1
    public static let payloadABI: UInt8 = 2
    public static let hitHalfLifeSeconds: Double = 6 * 60 * 60

    // Extension flag bits (DS4_KVSTORE_EXT_*).
    public static let extToolMap: UInt8 = 1 << 0
    public static let extResponsesVisible: UInt8 = 1 << 1
    public static let extThinkingVisible: UInt8 = 1 << 2
    public static let extSessionTitle: UInt8 = 1 << 3

    public enum Reason: UInt8, Sendable {
        case unknown = 0, cold = 1, continued = 2, evict = 3, shutdown = 4
        case agentSystem = 5, agentSession = 6
    }

    public struct Options: Sendable, Equatable {
        public var minTokens: Int = 512
        public var coldMaxTokens: Int = 30000
        public var continuedIntervalTokens: Int = 10000
        public var boundaryTrimTokens: Int = 32
        public var boundaryAlignTokens: Int = 2048
    }

    public static func defaultOptions() -> Options { Options() }

    /// Port of ds4_kvstore_reason_code.
    public static func reasonCode(_ reason: String?) -> UInt8 {
        switch reason {
        case "cold": return Reason.cold.rawValue
        case "continued": return Reason.continued.rawValue
        case "evict": return Reason.evict.rawValue
        case "shutdown": return Reason.shutdown.rawValue
        case "agent-system": return Reason.agentSystem.rawValue
        case "agent-session": return Reason.agentSession.rawValue
        default: return Reason.unknown.rawValue
        }
    }

    /// Port of ds4_kvstore_key_kind.
    public static func keyKind(extFlags: UInt8) -> String {
        if extFlags & extResponsesVisible != 0 { return "responses-visible" }
        if extFlags & extThinkingVisible != 0 { return "thinking-visible" }
        return "token-text"
    }

    // MARK: Little-endian helpers (ds4_kvstore_le_put32/get32 + le64)

    public static func lePut32(_ buf: inout [UInt8], _ off: Int, _ v: UInt32) {
        buf[off] = UInt8(v & 0xff)
        buf[off + 1] = UInt8((v >> 8) & 0xff)
        buf[off + 2] = UInt8((v >> 16) & 0xff)
        buf[off + 3] = UInt8((v >> 24) & 0xff)
    }
    public static func leGet32(_ buf: [UInt8], _ off: Int) -> UInt32 {
        UInt32(buf[off]) | (UInt32(buf[off + 1]) << 8) |
        (UInt32(buf[off + 2]) << 16) | (UInt32(buf[off + 3]) << 24)
    }
    static func lePut64(_ buf: inout [UInt8], _ off: Int, _ v: UInt64) {
        for i in 0..<8 { buf[off + i] = UInt8((v >> (8 * i)) & 0xff) }
    }
    static func leGet64(_ buf: [UInt8], _ off: Int) -> UInt64 {
        var v: UInt64 = 0
        for i in stride(from: 7, through: 0, by: -1) { v = (v << 8) | UInt64(buf[off + i]) }
        return v
    }

    // MARK: Header (port of ds4_kvstore_fill_header / read_header)

    public struct Header: Sendable, Equatable {
        public var quantBits: UInt8
        public var reason: UInt8
        public var extFlags: UInt8
        public var modelId: UInt8
        public var tokens: UInt32
        public var hits: UInt32
        public var ctxSize: UInt32
        public var createdAt: UInt64
        public var lastUsed: UInt64
        public var payloadBytes: UInt64
    }

    public static func fillHeader(_ h: Header) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: fixedHeader)
        b[0] = magic0; b[1] = magic1; b[2] = magic2; b[3] = version
        b[4] = h.quantBits
        b[5] = h.reason
        b[6] = h.extFlags
        b[7] = h.modelId
        lePut32(&b, 8, h.tokens)
        lePut32(&b, 12, h.hits)
        lePut32(&b, 16, h.ctxSize)
        b[20] = payloadABI
        lePut64(&b, 24, h.createdAt)
        lePut64(&b, 32, h.lastUsed)
        lePut64(&b, 40, h.payloadBytes)
        return b
    }

    /// Parse a 48-byte header. Returns nil if magic/version/ABI are wrong or the
    /// validity check (tokens != 0 and quant in {2,4}) fails — matching read_header.
    public static func parseHeader(_ b: [UInt8]) -> Header? {
        guard b.count >= fixedHeader else { return nil }
        guard b[0] == magic0, b[1] == magic1, b[2] == magic2, b[3] == version else { return nil }
        guard b[20] == payloadABI else { return nil }
        let reasonByte = b[5] <= Reason.agentSession.rawValue ? b[5] : Reason.unknown.rawValue
        let h = Header(quantBits: b[4], reason: reasonByte, extFlags: b[6], modelId: b[7],
                       tokens: leGet32(b, 8), hits: leGet32(b, 12), ctxSize: leGet32(b, 16),
                       createdAt: leGet64(b, 24), lastUsed: leGet64(b, 32), payloadBytes: leGet64(b, 40))
        guard h.tokens != 0, h.quantBits == 2 || h.quantBits == 4 else { return nil }
        return h
    }

    // MARK: SHA1 naming (ds4_kvstore_sha1_bytes_hex / sha_hex_name)

    public static func sha1Hex(_ bytes: [UInt8]) -> String {
        Insecure.SHA1.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    /// Port of ds4_kvstore_sha_hex_name: "<40 hex>.kv" -> lowercased sha, else nil.
    public static func shaHexName(_ name: String) -> String? {
        guard name.count == 43, name.hasSuffix(".kv") else { return nil }
        let hex = name.prefix(40)
        var out = ""
        out.reserveCapacity(40)
        for c in hex {
            guard c.isHexDigit else { return nil }
            out.append(Character(c.lowercased()))
        }
        return out
    }

    // MARK: Eviction score (base path; incoming == nil)

    public struct Entry: Sendable {
        public var hits: UInt32
        public var tokens: UInt32
        public var fileSize: UInt64
        public var createdAt: UInt64
        public var lastUsed: UInt64
        public var reason: UInt8
        public init(hits: UInt32, tokens: UInt32, fileSize: UInt64,
                    createdAt: UInt64, lastUsed: UInt64, reason: UInt8) {
            self.hits = hits; self.tokens = tokens; self.fileSize = fileSize
            self.createdAt = createdAt; self.lastUsed = lastUsed; self.reason = reason
        }
    }

    static func reasonIsAnchor(_ reason: UInt8) -> Bool {
        reason == Reason.cold.rawValue || reason == Reason.evict.rawValue || reason == Reason.shutdown.rawValue
    }

    /// Port of ds4_kvstore_entry_eviction_score with incoming == NULL.
    public static func evictionScore(_ e: Entry, now: UInt64) -> Double {
        if e.fileSize == 0 { return 0.0 }
        var effectiveHits = Double(e.hits)
        let usedAt = e.lastUsed != 0 ? e.lastUsed : e.createdAt
        if usedAt == 0 {
            effectiveHits = 0.0
        } else if now > usedAt {
            let elapsed = Double(now - usedAt)
            effectiveHits *= exp2(-elapsed / hitHalfLifeSeconds)
            if effectiveHits < 0.01 { effectiveHits = 0.0 }
        }
        var score = (effectiveHits + 1.0) * Double(e.tokens) / Double(e.fileSize)
        if reasonIsAnchor(e.reason) { score *= 2.0 }
        return score
    }
}

/// The DSV4 session payload header (13 little-endian u32 fields), per ds4.h and
/// the documented disk-KV format. The tensor body that follows is produced by
/// the inference graph and is ported alongside the graph phases.
public struct DSV4PayloadHeader: Sendable, Equatable {
    public static let magic: UInt32 = 0x3456_5344  // "DSV4"
    public static let version: UInt32 = 2
    public static let u32Fields = 13

    public var savedContextSize: UInt32
    public var prefillChunk: UInt32
    public var rawKVCapacity: UInt32
    public var rawSlidingWindow: UInt32
    public var compressedKVCapacity: UInt32
    public var checkpointTokenCount: UInt32
    public var layerCount: UInt32
    public var rawHeadKVDim: UInt32
    public var indexerHeadDim: UInt32
    public var vocabSize: UInt32
    public var liveRawRows: UInt32

    /// Serialize the 13-field header (magic, version, then the fields above).
    public func serialize() -> [UInt8] {
        var b = [UInt8](repeating: 0, count: DSV4PayloadHeader.u32Fields * 4)
        let fields: [UInt32] = [
            DSV4PayloadHeader.magic, DSV4PayloadHeader.version,
            savedContextSize, prefillChunk, rawKVCapacity, rawSlidingWindow,
            compressedKVCapacity, checkpointTokenCount, layerCount,
            rawHeadKVDim, indexerHeadDim, vocabSize, liveRawRows,
        ]
        for (i, v) in fields.enumerated() { KVCFile.lePut32(&b, i * 4, v) }
        return b
    }

    public init(savedContextSize: UInt32, prefillChunk: UInt32, rawKVCapacity: UInt32,
                rawSlidingWindow: UInt32, compressedKVCapacity: UInt32,
                checkpointTokenCount: UInt32, layerCount: UInt32, rawHeadKVDim: UInt32,
                indexerHeadDim: UInt32, vocabSize: UInt32, liveRawRows: UInt32) {
        self.savedContextSize = savedContextSize; self.prefillChunk = prefillChunk
        self.rawKVCapacity = rawKVCapacity; self.rawSlidingWindow = rawSlidingWindow
        self.compressedKVCapacity = compressedKVCapacity
        self.checkpointTokenCount = checkpointTokenCount; self.layerCount = layerCount
        self.rawHeadKVDim = rawHeadKVDim; self.indexerHeadDim = indexerHeadDim
        self.vocabSize = vocabSize; self.liveRawRows = liveRawRows
    }

    /// Parse a header; returns nil if magic/version mismatch.
    public init?(_ b: [UInt8]) {
        guard b.count >= DSV4PayloadHeader.u32Fields * 4 else { return nil }
        guard KVCFile.leGet32(b, 0) == DSV4PayloadHeader.magic,
              KVCFile.leGet32(b, 4) == DSV4PayloadHeader.version else { return nil }
        savedContextSize = KVCFile.leGet32(b, 8)
        prefillChunk = KVCFile.leGet32(b, 12)
        rawKVCapacity = KVCFile.leGet32(b, 16)
        rawSlidingWindow = KVCFile.leGet32(b, 20)
        compressedKVCapacity = KVCFile.leGet32(b, 24)
        checkpointTokenCount = KVCFile.leGet32(b, 28)
        layerCount = KVCFile.leGet32(b, 32)
        rawHeadKVDim = KVCFile.leGet32(b, 36)
        indexerHeadDim = KVCFile.leGet32(b, 40)
        vocabSize = KVCFile.leGet32(b, 44)
        liveRawRows = KVCFile.leGet32(b, 48)
    }
}
