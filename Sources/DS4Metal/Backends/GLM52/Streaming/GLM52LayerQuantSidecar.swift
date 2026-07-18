import DS4Core
import Foundation

// Q4_K layer sidecar — the GLM analog of the DeepSeek `.q4dense` cache: the
// streamed BIG tensors of a sparse layer (attention projections, indexer,
// shared expert — Q8_0 in the published GGUF) requantized ONCE to Q4_K on
// disk, roughly HALVING the dominant per-token SSD stream. keyB/valueB ride
// along VERBATIM (their kernels are Q8-hardwired), so a sidecar layer reads
// every big tensor from ONE file. The type-parametric MoE matvec kernels
// already dispatch Q4_K, so no Metal changes are involved.
//
// Like the expert bundles: one file per layer, bound to the source GGUF by
// an identity header (source offsets/bytes/file size), build is explicit,
// resumable and atomic, ANY subset of layers is useful, and a stale or
// foreign sidecar is refused — never silently served. The requantization
// is lossy on purpose (Q8_0 → Q4_K); it is an opt-in trade of quality
// margin for nearly 2× less layer I/O per token.

public enum GLM52LayerQuantSidecarError: Error, Sendable,
    CustomStringConvertible {
    case invalidHeader(path: String)
    case sourceMismatch(path: String)
    case shortFile(path: String)

    public var description: String {
        switch self {
        case .invalidHeader(let path):
            return "layer sidecar \(path): header non valido"
        case .sourceMismatch(let path):
            return "layer sidecar \(path): non appartiene a questo GGUF"
        case .shortFile(let path):
            return "layer sidecar \(path): troncato"
        }
    }
}

/// One validated, opened per-layer Q4_K sidecar.
public final class GLM52LayerQuantSidecar {
    public static let magic: UInt32 = 0x3451_4C47   // "GLQ4"
    public static let version: UInt32 = 1
    static let fixedHeaderBytes = 24
    static let entryBytes = 40

    public let layer: Int
    public let path: String
    /// Synthetic streamed-tensor view whose descriptors point INTO the
    /// sidecar file (requantized types included).
    public let tensors: GLM52StreamedLayerTensors
    /// Open pread reader over the sidecar file, handed to the streamer.
    public let reader: GLM52PayloadReader

    public static func fileName(layer: Int) -> String {
        "glm52-blk\(layer).layerq4"
    }

    public static func path(directory: String, layer: Int) -> String {
        (directory as NSString).appendingPathComponent(fileName(layer: layer))
    }

    private init(layer: Int, path: String,
                 tensors: GLM52StreamedLayerTensors,
                 reader: GLM52PayloadReader) {
        self.layer = layer
        self.path = path
        self.tensors = tensors
        self.reader = reader
    }

    /// The tensor order of the file — the single source of truth shared by
    /// build and open. `requant` marks the Q8_0→Q4_K candidates; keyB and
    /// valueB stay verbatim by contract (Q8-hardwired kernels).
    static func orderedEntries(of tensors: GLM52StreamedLayerTensors)
        -> [(descriptor: GLM52WeightDescriptor, requant: Bool)] {
        var entries: [(GLM52WeightDescriptor, Bool)] = [
            (tensors.qA, true), (tensors.qB, true), (tensors.kvA, true),
            (tensors.keyB, false), (tensors.valueB, false),
            (tensors.attnOutput, true),
        ]
        if let key = tensors.indexerKey { entries.append((key, true)) }
        if let queryB = tensors.indexerQueryB {
            entries.append((queryB, true))
        }
        entries.append((tensors.sharedGate, true))
        entries.append((tensors.sharedUp, true))
        entries.append((tensors.sharedDown, true))
        return entries
    }

    /// Whether a source tensor actually converts: validated Q8_0 input with
    /// a K-quant-aligned row width. Anything else is copied verbatim (its
    /// original type flows through the header to the kernels).
    static func converts(_ descriptor: GLM52WeightDescriptor,
                        requant: Bool) -> Bool {
        requant && descriptor.type == GLM52TensorSchema.q8_0
            && !descriptor.dims.isEmpty
            && descriptor.dims[0] % 256 == 0
    }

    /// Q4_K byte size of a converted tensor (row width dims[0], row count
    /// from the Q8_0 payload size).
    static func convertedBytes(_ descriptor: GLM52WeightDescriptor) -> UInt64 {
        let width = Int(descriptor.dims[0])
        let q8RowBytes = width / 32 * 34
        let rows = Int(descriptor.bytes) / q8RowBytes
        return UInt64(rows * (width / 256 * 144))
    }

    // MARK: - Open

    /// Open and validate a layer sidecar against the live GGUF tensors; nil
    /// if the file does not exist, throws if it exists but does not match.
    public static func open(directory: String, layer: Int,
                            source: GLM52StreamedLayerTensors,
                            sourceFileSize: UInt64) throws
        -> GLM52LayerQuantSidecar? {
        let sidecarPath = path(directory: directory, layer: layer)
        guard FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }
        let reader = try GLM52PayloadReader(path: sidecarPath)
        let entries = orderedEntries(of: source)
        let headerBytes = fixedHeaderBytes + entryBytes * entries.count
        guard reader.fileSize >= UInt64(headerBytes) else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: sidecarPath)
        }
        let headerDescriptor = GLM52WeightDescriptor(
            name: "sidecar.header", type: GLM52TensorSchema.q8_0,
            dims: [], absOffset: 0, bytes: UInt64(headerBytes))
        let header = try reader.bytes(of: headerDescriptor)
        func u32(_ offset: Int) -> UInt32 {
            header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
        }
        func u64(_ offset: Int) -> UInt64 {
            header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
            }
        }
        guard u32(0) == magic, u32(4) == version, u32(8) == UInt32(layer),
              u32(12) == UInt32(entries.count) else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: sidecarPath)
        }
        guard u64(16) == sourceFileSize else {
            throw GLM52LayerQuantSidecarError.sourceMismatch(path: sidecarPath)
        }

        var synthetic: [GLM52WeightDescriptor] = []
        for (rank, entry) in entries.enumerated() {
            let base = fixedHeaderBytes + entryBytes * rank
            let newType = u32(base)
            let newBytes = u64(base + 8)
            let newOffset = u64(base + 16)
            let sourceOffset = u64(base + 24)
            let sourceBytes = u64(base + 32)
            guard sourceOffset == entry.descriptor.absOffset,
                  sourceBytes == entry.descriptor.bytes else {
                throw GLM52LayerQuantSidecarError.sourceMismatch(
                    path: sidecarPath)
            }
            let expectConvert = converts(entry.descriptor,
                                         requant: entry.requant)
            let expectedType = expectConvert
                ? GLM52TensorSchema.q4_K : entry.descriptor.type
            let expectedBytes = expectConvert
                ? convertedBytes(entry.descriptor) : entry.descriptor.bytes
            guard newType == expectedType, newBytes == expectedBytes,
                  newOffset >= UInt64(headerBytes),
                  newOffset + newBytes <= reader.fileSize else {
                throw GLM52LayerQuantSidecarError.shortFile(path: sidecarPath)
            }
            synthetic.append(GLM52WeightDescriptor(
                name: entry.descriptor.name + ".layerq4",
                type: newType, dims: entry.descriptor.dims,
                absOffset: newOffset, bytes: newBytes))
        }

        let hasIndexer = source.indexerKey != nil
        let sharedStart = hasIndexer ? 8 : 6
        let tensors = GLM52StreamedLayerTensors(
            index: layer, fromSidecar: true,
            qA: synthetic[0], qB: synthetic[1], kvA: synthetic[2],
            keyB: synthetic[3], valueB: synthetic[4],
            attnOutput: synthetic[5],
            indexerKey: hasIndexer ? synthetic[6] : nil,
            indexerQueryB: hasIndexer ? synthetic[7] : nil,
            sharedGate: synthetic[sharedStart],
            sharedUp: synthetic[sharedStart + 1],
            sharedDown: synthetic[sharedStart + 2])
        return GLM52LayerQuantSidecar(
            layer: layer, path: sidecarPath, tensors: tensors,
            reader: reader)
    }

    // MARK: - Build

    /// Build one layer's sidecar (skip if a valid one exists). Atomic:
    /// `.part`, fsync, rename. Returns whether a build happened.
    @discardableResult
    public static func build(directory: String, layer: Int,
                             source: GLM52StreamedLayerTensors,
                             reader: GLM52PayloadReader) throws -> Bool {
        let existing = (try? open(directory: directory, layer: layer,
                                  source: source,
                                  sourceFileSize: reader.fileSize)) ?? nil
        if existing != nil { return false }
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        let finalPath = path(directory: directory, layer: layer)
        let partPath = finalPath + ".part"
        FileManager.default.createFile(atPath: partPath, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: partPath) else {
            throw GLM52PayloadReaderError.cannotOpen(path: partPath, code: 0)
        }

        let entries = orderedEntries(of: source)
        let headerBytes = fixedHeaderBytes + entryBytes * entries.count
        var header = [UInt8]()
        header.reserveCapacity(headerBytes)
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) {
                header.append(contentsOf: $0)
            }
        }
        func u64(_ value: UInt64) {
            withUnsafeBytes(of: value.littleEndian) {
                header.append(contentsOf: $0)
            }
        }
        u32(magic); u32(version)
        u32(UInt32(layer)); u32(UInt32(entries.count))
        u64(reader.fileSize)

        // Lay the payload out 16-byte aligned after the header.
        var cursor = UInt64(headerBytes)
        var plans: [(entry: (descriptor: GLM52WeightDescriptor,
                             requant: Bool),
                     offset: UInt64, bytes: UInt64, type: UInt32)] = []
        for entry in entries {
            cursor = (cursor + 15) & ~15
            let convert = converts(entry.descriptor, requant: entry.requant)
            let bytes = convert
                ? convertedBytes(entry.descriptor) : entry.descriptor.bytes
            let type = convert
                ? GLM52TensorSchema.q4_K : entry.descriptor.type
            u32(type); u32(0)
            u64(bytes); u64(cursor)
            u64(entry.descriptor.absOffset); u64(entry.descriptor.bytes)
            plans.append((entry, cursor, bytes, type))
            cursor += bytes
        }
        try handle.write(contentsOf: Data(header))

        for plan in plans {
            try handle.seek(toOffset: plan.offset)
            let descriptor = plan.entry.descriptor
            let sourceBytes = try reader.bytes(of: descriptor)
            if plan.type == GLM52TensorSchema.q4_K,
               descriptor.type == GLM52TensorSchema.q8_0 {
                let width = Int(descriptor.dims[0])
                let q8RowBytes = width / 32 * 34
                let q4RowBytes = width / 256 * 144
                let rows = sourceBytes.count / q8RowBytes
                var output = [UInt8](repeating: 0,
                                     count: rows * q4RowBytes)
                // Chunked rows: bounded F32 scratch (~8 MiB) regardless of
                // tensor size; quantizeQ4_K parallelizes internally.
                let chunkRows = max(1, (8 << 20) / (width * 4))
                var f32 = [Float](repeating: 0, count: chunkRows * width)
                var row = 0
                while row < rows {
                    let n = min(chunkRows, rows - row)
                    sourceBytes.withUnsafeBytes { sourceBuffer in
                        f32.withUnsafeMutableBufferPointer { floatBuffer in
                            Quantize.dequantQ8_0(
                                sourceBuffer.baseAddress! + row * q8RowBytes,
                                count: n * width,
                                into: floatBuffer.baseAddress!)
                        }
                    }
                    f32.withUnsafeBufferPointer { floatBuffer in
                        output.withUnsafeMutableBytes { outBuffer in
                            Quantize.quantizeQ4_K(
                                floatBuffer.baseAddress!, count: n * width,
                                into: outBuffer.baseAddress!
                                    + row * q4RowBytes)
                        }
                    }
                    row += n
                }
                try handle.write(contentsOf: Data(output))
            } else {
                try handle.write(contentsOf: Data(sourceBytes))
            }
        }
        try handle.synchronize()
        try handle.close()
        _ = try FileManager.default.replaceItemAt(
            URL(fileURLWithPath: finalPath),
            withItemAt: URL(fileURLWithPath: partPath))
        return true
    }

    public struct BuildSummary: Sendable {
        public let created: Int
        public let alreadyValid: Int
        public let remaining: Int
        public let stoppedBecause: String?
    }

    /// Bulk build over the sparse layers with the SAME semantics as the
    /// expert bundles: `maxSidecars` is a TOTAL cap (existing valid ones
    /// count toward it — rerunning with the same cap builds nothing more),
    /// the walk stops gracefully under `minFreeBytes`, any prefix of layers
    /// is useful, and a partial `.part` is removed on error.
    public static func buildAvailable(directory: String,
                                      weightMap: GLM52WeightMap,
                                      reader: GLM52PayloadReader,
                                      maxSidecars: Int? = nil,
                                      minFreeBytes: UInt64 = 8 << 30,
                                      layerProgress: ((Int, Bool) -> Void)?
                                          = nil) throws -> BuildSummary {
        let shape = weightMap.configuration.shape
        let sparse = Int(shape.nLeadingDense)..<Int(shape.inferenceLayerCount)
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        var created = 0
        var valid = 0
        var stopped: String?
        for layer in sparse {
            if let cap = maxSidecars, created + valid >= cap {
                stopped = "raggiunto il tetto di \(cap) sidecar totali"
                break
            }
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(
                layer, shape: shape)
            let source = try GLM52StreamedLayerTensors(
                index: layer, map: weightMap, fullIndexer: isFull)
            let perLayer = orderedEntries(of: source).reduce(UInt64(0)) {
                $0 + (converts($1.descriptor, requant: $1.requant)
                    ? convertedBytes($1.descriptor) : $1.descriptor.bytes)
            }
            let free = ((try? FileManager.default.attributesOfFileSystem(
                forPath: directory))?[.systemFreeSize] as? UInt64) ?? 0
            let existing = (try? open(
                directory: directory, layer: layer, source: source,
                sourceFileSize: reader.fileSize)) ?? nil
            if existing == nil, free < perLayer + minFreeBytes {
                stopped = "spazio disco insufficiente (liberi "
                    + "\(free >> 30) GiB, servono "
                    + "\((perLayer + minFreeBytes) >> 30) GiB)"
                break
            }
            do {
                let built = try build(directory: directory, layer: layer,
                                      source: source, reader: reader)
                if built { created += 1 } else { valid += 1 }
                layerProgress?(layer, built)
            } catch {
                try? FileManager.default.removeItem(
                    atPath: path(directory: directory, layer: layer)
                        + ".part")
                stopped = "errore su blk\(layer): \(error)"
                break
            }
        }
        return BuildSummary(
            created: created, alreadyValid: valid,
            remaining: sparse.count - created - valid,
            stoppedBecause: stopped)
    }
}
