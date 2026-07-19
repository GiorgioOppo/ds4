import Foundation

// Roadmap step 1 (docs/architectures/glm-5.2): connect the validated weight map
// to REAL file reads. This is the pread primitive under every future GLM cache
// or streaming fill: one descriptor → its exact payload bytes, one expert
// stream plan → one packed gate|up|down record per selected expert. It owns no
// GPU resource and never interprets the quantized bytes it moves, so it is
// fully testable against synthetic files without a Metal device.

public enum GLM52PayloadReaderError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case cannotOpen(path: String, code: Int32)
    case cannotStat(path: String, code: Int32)
    case rangeOutsideFile(name: String, end: UInt64, fileSize: UInt64)
    case byteCountTooLarge(name: String, bytes: UInt64)
    case destinationSizeMismatch(expected: UInt64, got: Int)
    case emptyPlan(layer: Int)
    case nonUniformPlan(layer: Int)
    case readFailed(name: String, offset: UInt64, bytes: UInt64)

    public var description: String {
        switch self {
        case .cannotOpen(let path, let code):
            return "GLM 5.2 payload: cannot open \(path) (errno \(code))"
        case .cannotStat(let path, let code):
            return "GLM 5.2 payload: cannot stat \(path) (errno \(code))"
        case .rangeOutsideFile(let name, let end, let fileSize):
            return "GLM 5.2 payload: \(name) ends at byte \(end) but the file has \(fileSize)"
        case .byteCountTooLarge(let name, let bytes):
            return "GLM 5.2 payload: \(name) byte count \(bytes) does not fit Int"
        case .destinationSizeMismatch(let expected, let got):
            return "GLM 5.2 payload: destination holds \(got) bytes, read needs exactly \(expected)"
        case .emptyPlan(let layer):
            return "GLM 5.2 payload: empty expert stream plan for layer \(layer)"
        case .nonUniformPlan(let layer):
            return "GLM 5.2 payload: layer \(layer) plan mixes per-expert byte sizes"
        case .readFailed(let name, let offset, let bytes):
            return "GLM 5.2 payload: pread failed on \(name) (\(bytes) bytes at \(offset))"
        }
    }
}

/// Byte layout of the buffer `read(plan:into:)` fills: the selected experts in
/// router rank order, each one as a contiguous gate|up|down record. The record
/// shape matches the interleaved slot layout the DeepSeek expert pool already
/// uses, so a future GLM slot cache can consume these buffers as-is.
public struct GLM52ExpertPackedRecordLayout: Sendable, Equatable {
    public let expertCount: Int
    public let gateBytes: Int
    public let upBytes: Int
    public let downBytes: Int

    public var gateOffset: Int { 0 }
    public var upOffset: Int { gateBytes }
    public var downOffset: Int { gateBytes + upBytes }
    public var recordBytes: Int { gateBytes + upBytes + downBytes }
    public var totalBytes: Int { recordBytes * expertCount }

    public func recordOffset(rank: Int) -> Int {
        precondition(rank >= 0 && rank < expertCount,
                     "rank \(rank) is outside 0..<\(expertCount)")
        return rank * recordBytes
    }
}

/// Bounded pread access to one GLM 5.2 GGUF payload.
///
/// Every read is validated twice before any byte moves: the planner/schema
/// already proved the range lies inside its tensor, and the reader re-checks it
/// against the real file size. A truncated download or a stale descriptor can
/// therefore never turn into a short read served as weights.
///
/// @unchecked Sendable: `fd`/`fileSize`/`path` are immutable after init and
/// pread takes explicit offsets (no shared cursor), so concurrent reads into
/// disjoint destinations are safe.
public final class GLM52PayloadReader: @unchecked Sendable {
    public let path: String
    public let fileSize: UInt64
    private let fd: Int32
    /// Base della vista FINESTRATA (0 = file intero): ogni pread trasla di
    /// questo offset e `fileSize` riporta la lunghezza della finestra, così
    /// tutta la validazione dei chiamanti resta relativa alla sezione. È il
    /// meccanismo con cui una sezione del pack unico si comporta esattamente
    /// come un file per-layer autonomo.
    private let windowBase: UInt64

    /// Vista finestrata su un sotto-intervallo di questo file (una sezione
    /// del pack). Il descrittore è DUPLICATO: la vista ha vita propria e il
    /// deinit chiude solo il proprio fd (F_NOCACHE segue la descrizione di
    /// file condivisa).
    public func windowed(offset: UInt64, length: UInt64) throws
        -> GLM52PayloadReader {
        guard length > 0, offset <= fileSize, length <= fileSize - offset
        else {
            throw GLM52PayloadReaderError.rangeOutsideFile(
                name: "window", end: offset + length, fileSize: fileSize)
        }
        let duplicated = dup(fd)
        guard duplicated >= 0 else {
            throw GLM52PayloadReaderError.cannotOpen(path: path, code: errno)
        }
        return GLM52PayloadReader(fd: duplicated, path: path,
                                  windowBase: windowBase + offset,
                                  fileSize: length)
    }

    private init(fd: Int32, path: String, windowBase: UInt64,
                 fileSize: UInt64) {
        self.fd = fd
        self.path = path
        self.windowBase = windowBase
        self.fileSize = fileSize
    }

    public init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw GLM52PayloadReaderError.cannotOpen(path: path, code: errno)
        }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 0 else {
            let code = errno
            close(fd)
            throw GLM52PayloadReaderError.cannotStat(path: path, code: code)
        }
        // F_NOCACHE opzionale (DS4_GLM_NOCACHE=1), MISURATO CONTROPRODUCENTE
        // come default su M1 Pro 16 GB (0.17→0.15 tok/s): il riuso della page
        // cache sui record esperti è basso ma non nullo, e il costo host del
        // decode viene dalla contesa MTLIO/commit, non dallo sfratto cache.
        // Lasciato come knob per macchine con più RAM o pattern diversi.
        if ProcessInfo.processInfo.environment["DS4_GLM_NOCACHE"] == "1" {
            _ = fcntl(fd, F_NOCACHE, 1)
        }
        self.fd = fd
        self.path = path
        self.windowBase = 0
        self.fileSize = UInt64(st.st_size)
    }

    /// Open the payload AND prove the whole validated directory fits the file.
    /// This is the constructor production code should use: it rejects a
    /// truncated or substituted GGUF at open time instead of at first touch.
    public convenience init(path: String, weightMap: GLM52WeightMap) throws {
        try self.init(path: path)
        if let farthest = weightMap.farthestDescriptor {
            let end = farthest.absOffset + farthest.bytes
            guard end <= fileSize else {
                throw GLM52PayloadReaderError.rangeOutsideFile(
                    name: farthest.name, end: end, fileSize: fileSize)
            }
        }
    }

    deinit { close(fd) }

    /// Read one validated descriptor's full payload. `destination` must hold
    /// exactly `descriptor.bytes` — a size mismatch is a caller bug surfaced as
    /// an error, never a partial fill.
    public func read(_ descriptor: GLM52WeightDescriptor,
                     into destination: UnsafeMutableRawBufferPointer) throws {
        let bytes = try checkedRange(name: descriptor.name,
                                     offset: descriptor.absOffset,
                                     byteCount: descriptor.bytes)
        guard destination.count == bytes else {
            throw GLM52PayloadReaderError.destinationSizeMismatch(
                expected: descriptor.bytes, got: destination.count)
        }
        guard let base = destination.baseAddress,
              GGUFWeights.preadFull(fd, into: base, bytes: bytes,
                                    offset: Int(windowBase
                                        + descriptor.absOffset)) else {
            throw GLM52PayloadReaderError.readFailed(
                name: descriptor.name,
                offset: descriptor.absOffset,
                bytes: descriptor.bytes)
        }
    }

    /// Convenience wrapper of `read(_:into:)` for small control tensors
    /// (norms, biases, router). Do not use it for multi-GiB payloads.
    public func bytes(of descriptor: GLM52WeightDescriptor) throws -> [UInt8] {
        let bytes = try checkedRange(name: descriptor.name,
                                     offset: descriptor.absOffset,
                                     byteCount: descriptor.bytes)
        var payload = [UInt8](repeating: 0, count: bytes)
        try payload.withUnsafeMutableBytes { try read(descriptor, into: $0) }
        return payload
    }

    /// A byte sub-range of one validated descriptor (the embedding-row read:
    /// one token's Q8_0 row out of a ~1 GiB tensor). The range is validated
    /// against the descriptor FIRST and against the file second, so a stale
    /// offset can never leave the tensor even on an oversized file.
    public func bytes(of descriptor: GLM52WeightDescriptor,
                      byteOffset: UInt64,
                      byteCount: UInt64) throws -> [UInt8] {
        var payload = [UInt8](repeating: 0, count: Int(byteCount))
        try payload.withUnsafeMutableBytes {
            try read(descriptor, byteOffset: byteOffset,
                     byteCount: byteCount, into: $0)
        }
        return payload
    }

    /// Zero-copy variant of the ranged read: the sub-range lands straight in
    /// `destination` (an MTLBuffer's contents on the expert staging path — no
    /// intermediate arrays). Same double validation, and safe for concurrent
    /// calls into disjoint destinations (stateless pread).
    public func read(_ descriptor: GLM52WeightDescriptor,
                     byteOffset: UInt64,
                     byteCount: UInt64,
                     into destination: UnsafeMutableRawBufferPointer) throws {
        guard byteCount > 0,
              byteOffset <= descriptor.bytes,
              byteCount <= descriptor.bytes - byteOffset else {
            throw GLM52PayloadReaderError.rangeOutsideFile(
                name: descriptor.name,
                end: descriptor.absOffset + byteOffset + byteCount,
                fileSize: fileSize)
        }
        let absolute = descriptor.absOffset + byteOffset
        let bytes = try checkedRange(name: descriptor.name,
                                     offset: absolute,
                                     byteCount: byteCount)
        guard destination.count == bytes else {
            throw GLM52PayloadReaderError.destinationSizeMismatch(
                expected: byteCount, got: destination.count)
        }
        guard let base = destination.baseAddress,
              GGUFWeights.preadFull(fd, into: base, bytes: bytes,
                                    offset: Int(windowBase + absolute)) else {
            throw GLM52PayloadReaderError.readFailed(
                name: descriptor.name, offset: absolute, bytes: byteCount)
        }
    }

    /// Execute one expert stream plan: the 24 planned ranges land as packed
    /// gate|up|down records in router rank order. The three preads of one
    /// expert stay adjacent in the destination even though they are scattered
    /// across three tensors in the file.
    ///
    /// `destination` must hold exactly `layout.totalBytes`. With `concurrent`
    /// the per-range reads run via `DispatchQueue.concurrentPerform` (disjoint
    /// destinations, explicit offsets); the serial path is byte-identical and
    /// exists for deterministic debugging.
    @discardableResult
    public func read(plan: GLM52ExpertStreamPlan,
                     into destination: UnsafeMutableRawBufferPointer,
                     concurrent: Bool = true) throws -> GLM52ExpertPackedRecordLayout {
        let layout = try packedLayout(of: plan)
        guard destination.count == layout.totalBytes else {
            throw GLM52PayloadReaderError.destinationSizeMismatch(
                expected: UInt64(layout.totalBytes), got: destination.count)
        }
        guard let base = destination.baseAddress else {
            throw GLM52PayloadReaderError.destinationSizeMismatch(
                expected: UInt64(layout.totalBytes), got: 0)
        }

        struct Job {
            let name: String
            let fileOffset: UInt64
            let destinationOffset: Int
            let bytes: Int
        }
        var jobs: [Job] = []
        jobs.reserveCapacity(plan.experts.count * 3)
        for (rank, expert) in plan.experts.enumerated() {
            let record = layout.recordOffset(rank: rank)
            for (range, offset) in [(expert.gate, layout.gateOffset),
                                    (expert.up, layout.upOffset),
                                    (expert.down, layout.downOffset)] {
                let bytes = try checkedRange(name: range.tensorName,
                                             offset: range.absoluteOffset,
                                             byteCount: range.byteCount)
                jobs.append(Job(name: range.tensorName,
                                fileOffset: range.absoluteOffset,
                                destinationOffset: record + offset,
                                bytes: bytes))
            }
        }

        if concurrent {
            // nonisolated(unsafe): ogni job scrive un range DISGIUNTO di base
            // (record e proiezione propri); il primo errore e' protetto dal
            // lock — stesso pattern di GGUFWeights.gatherExperts.
            nonisolated(unsafe) let destinationBase = base
            let jobList = jobs
            nonisolated(unsafe) var failure: GLM52PayloadReaderError?
            let lock = NSLock()
            let fd = self.fd
            let windowBase = self.windowBase
            DispatchQueue.concurrentPerform(iterations: jobList.count) { i in
                let job = jobList[i]
                let dst = destinationBase + job.destinationOffset
                if !GGUFWeights.preadFull(fd, into: dst, bytes: job.bytes,
                                          offset: Int(windowBase
                                              + job.fileOffset)) {
                    lock.lock()
                    if failure == nil {
                        failure = .readFailed(name: job.name,
                                              offset: job.fileOffset,
                                              bytes: UInt64(job.bytes))
                    }
                    lock.unlock()
                }
            }
            if let failure { throw failure }
        } else {
            for job in jobs {
                let dst = base + job.destinationOffset
                guard GGUFWeights.preadFull(fd, into: dst, bytes: job.bytes,
                                            offset: Int(windowBase
                                                + job.fileOffset)) else {
                    throw GLM52PayloadReaderError.readFailed(
                        name: job.name,
                        offset: job.fileOffset,
                        bytes: UInt64(job.bytes))
                }
            }
        }
        return layout
    }

    /// Derive the packed record layout of a plan and prove it uniform: every
    /// expert of one layer must plan identical gate/up/down byte sizes (the
    /// planner guarantees it — re-checked here so a hand-built plan cannot
    /// interleave mismatched records).
    public func packedLayout(of plan: GLM52ExpertStreamPlan) throws
        -> GLM52ExpertPackedRecordLayout {
        guard let first = plan.experts.first else {
            throw GLM52PayloadReaderError.emptyPlan(layer: plan.layer)
        }
        for expert in plan.experts.dropFirst() {
            guard expert.gate.byteCount == first.gate.byteCount,
                  expert.up.byteCount == first.up.byteCount,
                  expert.down.byteCount == first.down.byteCount else {
                throw GLM52PayloadReaderError.nonUniformPlan(layer: plan.layer)
            }
        }
        let gateBytes = try checkedByteCount(name: first.gate.tensorName,
                                             first.gate.byteCount)
        let upBytes = try checkedByteCount(name: first.up.tensorName,
                                           first.up.byteCount)
        let downBytes = try checkedByteCount(name: first.down.tensorName,
                                             first.down.byteCount)
        return GLM52ExpertPackedRecordLayout(
            expertCount: plan.experts.count,
            gateBytes: gateBytes,
            upBytes: upBytes,
            downBytes: downBytes)
    }

    // MARK: - Bounds

    private func checkedByteCount(name: String, _ byteCount: UInt64) throws -> Int {
        guard let bytes = Int(exactly: byteCount) else {
            throw GLM52PayloadReaderError.byteCountTooLarge(name: name, bytes: byteCount)
        }
        return bytes
    }

    /// One overflow-checked window proof: `offset + byteCount` must stay inside
    /// the real file. Returns the byte count as Int for the pread call.
    private func checkedRange(name: String,
                              offset: UInt64,
                              byteCount: UInt64) throws -> Int {
        let bytes = try checkedByteCount(name: name, byteCount)
        let (end, overflow) = offset.addingReportingOverflow(byteCount)
        guard !overflow, end <= fileSize, offset <= UInt64(Int.max) else {
            throw GLM52PayloadReaderError.rangeOutsideFile(
                name: name,
                end: overflow ? UInt64.max : end,
                fileSize: fileSize)
        }
        return bytes
    }
}
