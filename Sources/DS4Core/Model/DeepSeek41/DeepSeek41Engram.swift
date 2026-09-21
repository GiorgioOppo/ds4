import Foundation
import Darwin

/// SSD-backed original E4M3/E8M0 rows. The 189 GiB table is never copied or
/// registered as a Metal buffer. Only the rows selected by the n-gram hash
/// are read, with a per-call bounded deduplication cache.
public final class DeepSeek41Engram {
    public struct Layout: Sendable {
        public let tokenMap: [UInt32]
        public let compressedVocabulary: UInt32
        public let padID: UInt32
        public let rows: [UInt32]
        public let multipliers: [[UInt64]]
        public let primes: [[UInt32]]

        public init(tokenMap: [UInt32], compressedVocabulary: UInt32, padID: UInt32,
                    rows: [UInt32], multipliers: [[UInt64]], primes: [[UInt32]]) throws {
            guard !tokenMap.isEmpty, compressedVocabulary > 0,
                  compressedVocabulary <= Int32.max, padID < compressedVocabulary,
                  tokenMap.allSatisfy({ $0 < compressedVocabulary }), rows.count == 2,
                  multipliers.count == 2, primes.count == 2 else {
                throw DeepSeek41Error.engram("invalid hash layout")
            }
            for layer in 0..<2 {
                guard multipliers[layer].count == 4, primes[layer].count == 24,
                      multipliers[layer].allSatisfy({ $0 & 1 == 1 && $0 <= UInt64(Int64.max) / UInt64(compressedVocabulary) }),
                      primes[layer].allSatisfy({ $0 >= 2 }),
                      primes[layer].reduce(UInt64(0), { $0 + UInt64($1) }) == UInt64(rows[layer]) else {
                    throw DeepSeek41Error.engram("invalid hash primes or multipliers")
                }
            }
            self.tokenMap = tokenMap; self.compressedVocabulary = compressedVocabulary
            self.padID = padID; self.rows = rows; self.multipliers = multipliers; self.primes = primes
        }

        /// Returns [layer][token * 24 + column]. Masked positions break all
        /// crossing n-grams, as image spans do in the reference implementation.
        public func hash(tokens: [Int], mask: [Bool]? = nil, history: inout [Int32]) throws -> [[UInt32]] {
            guard history.count == 3,
                  history.allSatisfy({ $0 == -1 || ($0 >= 0 && UInt32($0) < compressedVocabulary) }),
                  tokens.allSatisfy({ tokenMap.indices.contains($0) }),
                  mask == nil || mask?.count == tokens.count,
                  tokens.count <= Int.max / 48 else { throw DeepSeek41Error.engram("invalid hash input") }
            var result = Array(repeating: [UInt32](), count: 2)
            result[0].reserveCapacity(tokens.count * 24); result[1].reserveCapacity(tokens.count * 24)
            for (position, token) in tokens.enumerated() {
                let current: Int32 = mask?[position] == false ? -1 : Int32(tokenMap[token])
                var blocked = false
                let ids = ([current] + history).map { id -> UInt64 in
                    blocked = blocked || id == -1
                    return UInt64(blocked ? padID : UInt32(id))
                }
                for layer in 0..<2 {
                    var value = ids[0] * multipliers[layer][0]
                    var offset: UInt32 = 0
                    for j in 1..<4 {
                        value ^= ids[j] * multipliers[layer][j]
                        for head in 0..<8 {
                            let prime = primes[layer][(j - 1) * 8 + head]
                            result[layer].append(UInt32(value % UInt64(prime)) + offset)
                            offset += prime
                        }
                    }
                }
                history = [current, history[0], history[1]]
            }
            return result
        }
    }

    public final class Table {
        private let fd: Int32
        private let offset: UInt64
        private let rowCount: UInt32

        public init(path: String, offset: UInt64, rows: UInt32) throws {
            let bytes = UInt64(rows) * 264
            guard rows > 0, offset <= UInt64(Int64.max), bytes <= UInt64(Int64.max) - offset else {
                throw DeepSeek41Error.engram("table extent overflow")
            }
            let descriptor = open(path, O_RDONLY | O_CLOEXEC)
            guard descriptor >= 0 else { throw DeepSeek41Error.engram("cannot open table: \(String(cString: strerror(errno)))") }
            var info = stat()
            guard fstat(descriptor, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
                  info.st_size >= 0, offset + bytes <= UInt64(info.st_size),
                  fcntl(descriptor, F_NOCACHE, 1) == 0, fcntl(descriptor, F_RDAHEAD, 0) == 0 else {
                close(descriptor)
                throw DeepSeek41Error.engram("invalid or truncated SSD table")
            }
            self.fd = descriptor; self.offset = offset; self.rowCount = rows
        }
        deinit { close(fd) }

        public static func decode(row: [UInt8]) throws -> [Float] {
            guard row.count == 264 else { throw DeepSeek41Error.engram("invalid row length") }
            return try (0..<256).map { i in
                let code = row[i], scale = row[256 + i / 32]
                guard code & 127 != 127, scale != 255 else { throw DeepSeek41Error.engram("non-finite row encoding") }
                let exponent = Int32((code >> 3) & 15), mantissa = Float(code & 7)
                var value = exponent == 0 ? ldexpf(mantissa, -9) : ldexpf(8 + mantissa, exponent - 10)
                if code & 128 != 0 { value = -value }
                value = ldexpf(value, Int32(scale) - 127)
                let bits = value.bitPattern
                value = Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
                guard value.isFinite else { throw DeepSeek41Error.engram("row overflow") }
                return value
            }
        }

        public func read(rows: [UInt32]) throws -> [Float] {
            guard rows.count <= Int.max / 256, rows.allSatisfy({ $0 < rowCount }) else {
                throw DeepSeek41Error.engram("row outside table")
            }
            var result = [Float](repeating: 0, count: rows.count * 256)
            // At most 256 unique decoded rows (256 KiB), regardless of batch.
            for start in stride(from: 0, to: rows.count, by: 256) {
                let end = min(rows.count, start + 256)
                var decoded: [UInt32: [Float]] = [:]
                for rowID in Set(rows[start..<end]).sorted() {
                    var raw = [UInt8](repeating: 0, count: 264)
                    try raw.withUnsafeMutableBytes { target in
                        var done = 0
                        while done < 264 {
                            let count = pread(fd, target.baseAddress!.advanced(by: done), 264 - done,
                                              off_t(offset + UInt64(rowID) * 264 + UInt64(done)))
                            if count < 0 && errno == EINTR { continue }
                            guard count > 0 else { throw DeepSeek41Error.engram("SSD read failed or truncated row") }
                            done += count
                        }
                    }
                    decoded[rowID] = try Self.decode(row: raw)
                }
                for index in start..<end {
                    result.replaceSubrange((index * 256)..<((index + 1) * 256), with: decoded[rows[index]]!)
                }
            }
            return result
        }
    }

    public let layout: Layout
    private let tables: [Table]
    private var history: [Int32] = [-1, -1, -1]

    public init(model: GGUFModel) throws {
        func array(_ suffix: String, count: Int, type: GGUFValueType) throws -> [Int64] {
            let key = "deepseek41.engram." + suffix
            guard model.array(key)?.type == type.rawValue,
                  let values = model.intArray(key), values.count == count,
                  values.allSatisfy({ $0 >= 0 }) else { throw DeepSeek41Error.invalidMetadata(key) }
            return values
        }
        let map = try array("token_map", count: 129280, type: .uint32).map(UInt32.init)
        let primes = try array("primes", count: 48, type: .uint32).map(UInt32.init)
        let multipliers = try array("multipliers", count: 8, type: .uint64).map(UInt64.init)
        guard let pad = model.u32("deepseek41.engram.pad_id") else {
            throw DeepSeek41Error.invalidMetadata("deepseek41.engram.pad_id")
        }
        let layout = try Layout(tokenMap: map, compressedVocabulary: 99092, padID: pad,
                            rows: DeepSeek41Configuration.engramRows.map(UInt32.init),
                            multipliers: [Array(multipliers[..<4]), Array(multipliers[4...])],
                            primes: [Array(primes[..<24]), Array(primes[24...])])
        self.layout = layout
        tables = try DeepSeek41Configuration.engramLayers.enumerated().map { index, layer in
            let name = "blk.\(layer).engram_embd.weight"
            guard let tensor = model.findTensor(name), tensor.type == 24,
                  tensor.dims == [264, UInt64(layout.rows[index])] else {
                throw DeepSeek41Error.invalidTensor(name)
            }
            return try Table(path: model.path, offset: tensor.absOffset, rows: layout.rows[index])
        }
    }
    public func reset() { history = [-1, -1, -1] }
    /// Both reads must succeed before hash state advances.
    public func read(tokens: [Int], mask: [Bool]? = nil) throws -> [[Float]] {
        var next = history
        let rows = try layout.hash(tokens: tokens, mask: mask, history: &next)
        let values = try (0..<2).map { try tables[$0].read(rows: rows[$0]) }
        history = next
        return values
    }
}
