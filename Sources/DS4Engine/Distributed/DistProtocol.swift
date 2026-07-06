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
    /// v4: distributed KV continuity. ASSIGN carries the usage imatrix (slot
    /// cache pre-warm) and a disk-KV token budget; new KV control frames let
    /// the coordinator checkpoint/restore each worker's shard (kvQuery /
    /// kvLengths / kvRestore / kvSave / kvAck); WORK gained `turnStart` so a
    /// turn can begin mid-context (restored or reused prefix).
    /// v5: the coordinator DISTRIBUTES the files. Workers no longer need a
    /// local gguf: after HELLO the coordinator sends a FILE OFFER (name, size,
    /// sha256 for gguf + sidecar); the worker answers with what it is missing
    /// (hash-verified against its managed store and its local files) and the
    /// coordinator streams only those — the huge setup runs ONCE; later
    /// connects verify hashes from cached manifests in milliseconds.
    /// v6: derived caches travel too — the offer can include the Q4 dense
    /// requant cache (<gguf>.q4dense, ~1.4 GB beats minutes of re-requant on
    /// every worker) and ASSIGN carries the Q4 on/off decision.
    /// v7: WORK carries the chunk's token ids. The first n_hash_layer (3)
    /// layers route experts by TOKEN ID (ffn_gate_tid2eid), so a shard that
    /// covers them cannot route from the HC state alone.
    /// v8: RESUMABLE transfers. The offer carries a CHAINED-HASH checkpoint
    /// list per file (one SHA-256 every fileCheckpointBytes, each folded over
    /// the previous — chain[k] commits to the whole prefix); the worker keeps
    /// its `.part` across disconnects AND sessions, validates it block-by-block
    /// against the chain, truncates to the last good checkpoint and answers
    /// FILE NEED with a per-file RESUME OFFSET. The coordinator streams from
    /// there and retries a broken peer setup up to 3 times before failing.
    /// v9: ASSIGN carries the coordinator's PERFORMANCE KNOBS (DS4_* env,
    /// whitelisted). A worker with factory defaults ran the engine without
    /// dense streaming/mlock/pread — measured 0.37 tok/s against the same
    /// hardware's 2.7 local. The coordinator's measured configuration IS the
    /// job definition: the worker applies it before loading its engine.
    public static let protocolVersion: UInt32 = 9

    /// The env knobs an ASSIGN may carry and a worker will apply (v9). A
    /// WHITELIST on both sides: the wire must never gain the power to set
    /// arbitrary environment on a worker. All performance-only — none of
    /// these can change the numerics of the shard (DS4_DENSE_Q4, the one
    /// lossy knob, travels as a typed field and its cache as a file).
    public static let perfKnobKeys: [String] = [
        "DS4_DENSE_STREAM", "DS4_DENSE_AHEAD", "DS4_MLOCK",
        "DS4_EXPERT_PREAD", "DS4_PREAD_SPLIT", "DS4_WILLNEED_EXPERTS",
        "DS4_ASYNC_FFN", "DS4_EXPERT_LOOKAHEAD", "DS4_Q8_NSG",
        "DS4_LAZY_IDX", "DS4_RESIDENT_COMP", "DS4_FUSED_HC", "DS4_FUSED_MOE",
        "DS4_RAW_RING", "DS4_EXPERT_CACHE_UNIFORM", "DS4_POOL_INTERLEAVE",
        "DS4_PREFILL_UNION", "DS4_PREFILL_CHUNK", "DS4_PREFILL_FFN_BATCH",
        "DS4_PREFILL_ROUTE_BATCH", "DS4_PREFILL_MM",
    ]
    static let magic: UInt32 = 0x44_53_34_44   // "DS4D"
    /// Sanity cap on the route length in a WORK frame (a hostile frame could
    /// otherwise declare 4G entries and spin the decoder).
    static let maxRouteEntries = 256
    /// Cap on entries in a FILE OFFER / NEED (gguf + a handful of sidecars).
    static let maxFileEntries = 16
    /// File-transfer chunk payload (per fileChunk frame).
    public static let fileChunkBytes = 4 * 1024 * 1024
    /// Checkpoint granularity of the chained-hash list (v8 resumable
    /// transfers): one 32-byte digest every 256 MB — a 70 GB gguf carries
    /// ~280 digests (~9 KB) in the offer, and at most 256 MB are re-sent
    /// after an interruption. MUST be a multiple of fileChunkBytes.
    public static let fileCheckpointBytes: UInt64 = 256 * 1024 * 1024

    public enum MsgType: UInt32, Sendable {
        case hello     = 1    // worker → coordinator on connect (state + model identity)
        case work      = 3    // coordinator → worker: embed/HC input for a layer slice
        case result    = 4    // worker → coordinator: HC state or logits
        case error     = 2
        case assign    = 5    // coordinator → worker: gguf + settings + layer slice
        case ready     = 6    // worker → coordinator: assignment loaded (HELLO payload)
        case kvQuery   = 7    // coordinator → worker: which stored prefixes of these ids?
        case kvLengths = 8    // worker → coordinator: stored prefix lengths (may be empty)
        case kvRestore = 9    // coordinator → worker: restore EXACTLY these tokens' checkpoint
        case kvSave    = 10   // coordinator → worker: checkpoint your shard for these tokens
        case kvAck     = 11   // worker → coordinator: kvRestore/kvSave outcome
        case fileOffer = 12   // coordinator → worker: manifest (name+size+sha256 per file)
        case fileNeed  = 13   // worker → coordinator: offer indices it is missing
        case fileChunk = 14   // coordinator → worker: sequential slab of one file
        case fileDone  = 15   // coordinator → worker: file complete (worker verifies hash)
        case fileAck   = 16   // worker → coordinator: per-file receive outcome
        case progress  = 17   // worker → coordinator: load progress text (informational)
    }

    /// Flags on a WORK message.
    public struct WorkFlags: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        public static let resetSession = WorkFlags(rawValue: 1 << 0)  // pos==0: reset compressor/KV
        public static let outputLogits = WorkFlags(rawValue: 1 << 1)  // last slice: this worker also runs the head
        /// First chunk of a TURN (chat send or benchmark). Workers adopt the
        /// chunk's session id here — a turn may start mid-context (pos > 0)
        /// when the coordinator reuses or restores a KV prefix.
        public static let turnStart = WorkFlags(rawValue: 1 << 2)
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
/// the context size, the expert-cache budget, the disk-KV budget, the usage
/// imatrix to pre-warm the slot cache with, and the layer slice to own.
/// The worker loads (or reuses) its engine and replies READY (HELLO payload).
public struct DistAssign: Sendable {
    public var modelPath: String      // coordinator's path (verbatim, tried first)
    public var modelName: String      // gguf filename (fallback resolution key)
    public var contextSize: Int
    public var expertCacheSlots: Int  // 0 = no expert slot-cache on the worker
    public var diskKVBudgetTokens: Int // 0 = no disk-KV checkpoints on the worker
    public var useExpertBundle: Bool   // worker runs with the expert-bundle sidecar
    public var useDenseQ4: Bool        // worker uses the Q4 dense requant (cache offered)
    public var layerStart: Int
    public var layerEnd: Int          // inclusive
    public var hasOutput: Bool        // last slice: also runs the output head
    /// Usage-imatrix JSON (ExpertUsageStats.serialize) to seed the worker's
    /// slot-cache pre-warm; empty = none (the worker may still have its own).
    public var usageJSON: Data
    /// v9: the coordinator's PERFORMANCE env (whitelisted, Dist.perfKnobKeys).
    /// The worker applies these before loading its engine, so a shard runs
    /// with the coordinator's measured configuration instead of whatever the
    /// worker app's local defaults happen to be.
    public var envKnobs: [(key: String, value: String)]

    public init(modelPath: String, modelName: String, contextSize: Int, expertCacheSlots: Int,
                diskKVBudgetTokens: Int, useExpertBundle: Bool = false, useDenseQ4: Bool = false,
                layerStart: Int, layerEnd: Int, hasOutput: Bool,
                usageJSON: Data = Data(), envKnobs: [(key: String, value: String)] = []) {
        self.modelPath = modelPath; self.modelName = modelName
        self.contextSize = contextSize; self.expertCacheSlots = expertCacheSlots
        self.diskKVBudgetTokens = diskKVBudgetTokens
        self.useExpertBundle = useExpertBundle
        self.useDenseQ4 = useDenseQ4
        self.layerStart = layerStart; self.layerEnd = layerEnd; self.hasOutput = hasOutput
        self.usageJSON = usageJSON
        self.envKnobs = envKnobs
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(contextSize))
        d.appendLE(UInt32(expertCacheSlots))
        d.appendLE(UInt32(diskKVBudgetTokens))
        d.appendLE(UInt32(useExpertBundle ? 1 : 0))
        d.appendLE(UInt32(useDenseQ4 ? 1 : 0))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
        let path = Data(modelPath.utf8)
        d.appendLE(UInt32(path.count)); d.append(path)
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        d.appendLE(UInt32(usageJSON.count)); d.append(usageJSON)
        d.appendLE(UInt32(envKnobs.count))
        for (k, v) in envKnobs {
            let kd = Data(k.utf8), vd = Data(v.utf8)
            d.appendLE(UInt32(kd.count)); d.append(kd)
            d.appendLE(UInt32(vd.count)); d.append(vd)
        }
        return d
    }

    public static func decode(_ d: Data) -> DistAssign? {
        var o = d.startIndex
        guard d.count >= 48 else { return nil }
        let ctx = Int(d.readLE(&o) as UInt32)
        let slots = Int(d.readLE(&o) as UInt32)
        let kvBudget = Int(d.readLE(&o) as UInt32)
        let bundle = (d.readLE(&o) as UInt32) != 0
        let q4 = (d.readLE(&o) as UInt32) != 0
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let ho = (d.readLE(&o) as UInt32) != 0
        let pathLen = Int(d.readLE(&o) as UInt32)
        guard pathLen >= 0, o + pathLen + 4 <= d.endIndex else { return nil }
        let path = String(decoding: d[o..<o+pathLen], as: UTF8.self); o += pathLen
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen >= 0, o + nameLen + 4 <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self); o += nameLen
        let usageLen = Int(d.readLE(&o) as UInt32)
        guard usageLen >= 0, o + usageLen + 4 <= d.endIndex else { return nil }
        let usage = Data(d[o..<o+usageLen]); o += usageLen
        let knobCount = Int(d.readLE(&o) as UInt32)
        guard knobCount >= 0, knobCount <= 64 else { return nil }
        var knobs: [(key: String, value: String)] = []
        knobs.reserveCapacity(knobCount)
        for _ in 0..<knobCount {
            guard o + 4 <= d.endIndex else { return nil }
            let kLen = Int(d.readLE(&o) as UInt32)
            guard kLen > 0, kLen <= 256, o + kLen + 4 <= d.endIndex else { return nil }
            let k = String(decoding: d[o..<o+kLen], as: UTF8.self); o += kLen
            let vLen = Int(d.readLE(&o) as UInt32)
            guard vLen >= 0, vLen <= 256, o + vLen <= d.endIndex else { return nil }
            let v = String(decoding: d[o..<o+vLen], as: UTF8.self); o += vLen
            knobs.append((key: k, value: v))
        }
        return DistAssign(modelPath: path, modelName: name, contextSize: ctx,
                          expertCacheSlots: slots, diskKVBudgetTokens: kvBudget,
                          useExpertBundle: bundle, useDenseQ4: q4,
                          layerStart: ls, layerEnd: le, hasOutput: ho, usageJSON: usage,
                          envKnobs: knobs)
    }
}

// MARK: - File distribution payloads (fileOffer / fileNeed / fileChunk / fileDone)

/// One distributable file: the gguf itself or a derived sidecar. Identified by
/// NAME (filename only — the worker re-sanitizes, never trusts paths), exact
/// size, and full SHA-256 — the identity later connects verify instead of
/// re-transferring.
public struct DistFileEntry: Sendable, Equatable {
    public enum Kind: UInt32, Sendable { case gguf = 0, expertBundle = 1, q4Dense = 2 }
    public var kind: Kind
    public var name: String
    public var size: UInt64
    public var sha256: Data          // 32 bytes
    /// v8 chained checkpoint hashes, one per fileCheckpointBytes block, with
    /// b_k = SHA256(raw bytes of block k):
    /// chain[0] = SHA256(b_0); chain[k] = SHA256(chain[k-1] ‖ b_k).
    /// Each entry commits to the WHOLE prefix, so a `.part` file from a broken
    /// transfer is verifiable block-by-block and the resume point is the last
    /// matching checkpoint — a corrupt middle block can never be resumed over.
    public var chain: [Data]

    public init(kind: Kind, name: String, size: UInt64, sha256: Data, chain: [Data] = []) {
        self.kind = kind; self.name = name; self.size = size; self.sha256 = sha256
        self.chain = chain
    }

    func encode(into d: inout Data) {
        d.appendLE(kind.rawValue)
        let n = Data(name.utf8)
        d.appendLE(UInt32(n.count)); d.append(n)
        d.appendLE(size)
        d.append(sha256.prefix(32))
        d.appendLE(UInt32(chain.count))
        for c in chain { d.append(c.prefix(32)) }
    }

    static func decode(_ d: Data, _ o: inout Data.Index) -> DistFileEntry? {
        guard o + 8 <= d.endIndex, let kind = Kind(rawValue: d.readLE(&o)) else { return nil }
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen > 0, nameLen < 1024, o + nameLen + 8 + 32 + 4 <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self); o += nameLen
        let size = d.readLE(&o) as UInt64
        let sha = Data(d[o..<o+32]); o += 32
        let chainCount = Int(d.readLE(&o) as UInt32)
        // Sanity: la catena è ceil(size/checkpoint) — cap largo contro frame ostili.
        guard chainCount >= 0, chainCount <= 1 << 20, o + chainCount * 32 <= d.endIndex else { return nil }
        var chain: [Data] = []
        chain.reserveCapacity(chainCount)
        for _ in 0..<chainCount { chain.append(Data(d[o..<o+32])); o += 32 }
        return DistFileEntry(kind: kind, name: name, size: size, sha256: sha, chain: chain)
    }
}

/// FILE OFFER: everything the coordinator can distribute for this assignment.
public struct DistFileOffer: Sendable {
    public var entries: [DistFileEntry]
    public init(entries: [DistFileEntry]) { self.entries = entries }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(entries.count))
        for e in entries { e.encode(into: &d) }
        return d
    }

    public static func decode(_ d: Data) -> DistFileOffer? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let count = Int(d.readLE(&o) as UInt32)
        guard count >= 0, count <= Dist.maxFileEntries else { return nil }
        var entries: [DistFileEntry] = []
        for _ in 0..<count {
            guard let e = DistFileEntry.decode(d, &o) else { return nil }
            entries.append(e)
        }
        return DistFileOffer(entries: entries)
    }
}

/// FILE NEED: the offer indices the worker is missing (empty = has everything).
public struct DistFileNeed: Sendable {
    public var indices: [Int]
    /// v8: per-index RESUME offset (0 = from scratch). Always a multiple of
    /// Dist.fileCheckpointBytes: the worker truncated its `.part` to the last
    /// checkpoint whose chained hash matched the offer's.
    public var offsets: [UInt64]

    public init(indices: [Int], offsets: [UInt64]? = nil) {
        self.indices = indices
        self.offsets = offsets ?? [UInt64](repeating: 0, count: indices.count)
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(UInt32(indices.count))
        for (j, i) in indices.enumerated() {
            d.appendLE(UInt32(truncatingIfNeeded: i))
            d.appendLE(j < offsets.count ? offsets[j] : 0)
        }
        return d
    }

    public static func decode(_ d: Data) -> DistFileNeed? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let count = Int(d.readLE(&o) as UInt32)
        guard count >= 0, count <= Dist.maxFileEntries, o + count * 12 <= d.endIndex else { return nil }
        var indices: [Int] = []
        var offsets: [UInt64] = []
        for _ in 0..<count {
            indices.append(Int(d.readLE(&o) as UInt32))
            offsets.append(d.readLE(&o) as UInt64)
        }
        return DistFileNeed(indices: indices, offsets: offsets)
    }
}

/// FILE CHUNK: one sequential slab of the file at `index` in the offer. The
/// worker enforces `offset == bytes received so far` — no sparse writes.
public struct DistFileChunk: Sendable {
    public var index: Int
    public var offset: UInt64
    public var data: Data

    public init(index: Int, offset: UInt64, data: Data) {
        self.index = index; self.offset = offset; self.data = data
    }

    public func encoded() -> Data {
        var d = Data(capacity: 16 + data.count)
        d.appendLE(UInt32(index))
        d.appendLE(offset)
        d.appendLE(UInt32(data.count))
        d.append(data)
        return d
    }

    public static func decode(_ d: Data) -> DistFileChunk? {
        var o = d.startIndex
        guard d.count >= 16 else { return nil }
        let index = Int(d.readLE(&o) as UInt32)
        let offset = d.readLE(&o) as UInt64
        let len = Int(d.readLE(&o) as UInt32)
        guard len >= 0, len <= Dist.fileChunkBytes, o + len <= d.endIndex else { return nil }
        return DistFileChunk(index: index, offset: offset, data: Data(d[o..<o+len]))
    }
}

/// FILE DONE: the file at `index` is complete — the worker verifies size+hash.
public struct DistFileDone: Sendable {
    public var index: Int
    public init(index: Int) { self.index = index }
    public func encoded() -> Data {
        var d = Data(); d.appendLE(UInt32(index)); return d
    }
    public static func decode(_ d: Data) -> DistFileDone? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        return DistFileDone(index: Int(d.readLE(&o) as UInt32))
    }
}

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
