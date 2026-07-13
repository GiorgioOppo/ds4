import Foundation
import DS4Core

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

