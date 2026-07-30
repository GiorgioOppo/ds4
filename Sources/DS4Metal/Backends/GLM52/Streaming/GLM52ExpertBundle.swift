import DS4Core
import Foundation

// Per-layer expert bundles — the GLM analog of the DeepSeek ExpertBundle:
// every routed expert's gate|up|down record repacked CONTIGUOUSLY on disk,
// so a selected expert costs ONE sequential pread instead of three scattered
// ones (and gives the OS readahead something to work with). Bundles are
// bound to their source GGUF by an identity header (file size plus the three
// tensor offsets and types); a stale or foreign bundle is refused, never
// silently served. Build is one-time, per layer, resumable (existing valid
// bundles are skipped) and atomic. I record vivono nel PACK UNICO
// glm52-experts.glmsidecar (stesso contenitore a sezioni del sidecar Q4);
// i file per-layer legacy restano leggibili e vengono migrati nel pack
// alla prima build successiva.
//
// Disk cost is honest and deliberate: the routed payload is duplicated
// (~most of the GGUF). This is the same trade the DeepSeek bundle makes —
// an explicit user opt-in via DS4_BUNDLE_DIR.

public enum GLM52ExpertBundleError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidHeader(path: String)
    case sourceMismatch(path: String)
    case shortFile(path: String)

    public var description: String {
        switch self {
        case .invalidHeader(let path):
            return "bundle \(path): header non valido"
        case .sourceMismatch(let path):
            return "bundle \(path): non appartiene a questo GGUF"
        case .shortFile(let path):
            return "bundle \(path): troncato"
        }
    }
}

/// One validated, opened per-layer bundle.
public final class GLM52ExpertBundle {
    public static let magic: UInt32 = 0x424D_4C47   // "GLMB"
    public static let version: UInt32 = 1
    public static let headerBytes = 80

    public let layer: Int
    public let expertCount: Int
    public let gateBytes: Int
    public let upBytes: Int
    public let downBytes: Int
    public let gateUpType: UInt32
    public let downType: UInt32
    public var recordBytes: Int { gateBytes + upBytes + downBytes }

    private let reader: GLM52PayloadReader
    private let payload: GLM52WeightDescriptor
    /// True quando il record vive nel pack unico; false = file per-layer
    /// legacy (candidato alla migrazione) o vista embedded nel sidecar.
    public let fromPack: Bool

    public static func fileName(layer: Int) -> String {
        "glm52-blk\(layer).experts"
    }

    public static func path(directory: String, layer: Int) -> String {
        (directory as NSString).appendingPathComponent(fileName(layer: layer))
    }

    /// The 80-byte identity header binding a bundle to its source GGUF.
    static func header(layer: Int, weights: GLM52RoutedExpertWeights,
                       expertCount: Int,
                       sourceFileSize: UInt64) -> [UInt8] {
        var bytes = [UInt8]()
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) {
                bytes.append(contentsOf: $0)
            }
        }
        func u64(_ value: UInt64) {
            withUnsafeBytes(of: value.littleEndian) {
                bytes.append(contentsOf: $0)
            }
        }
        u32(magic); u32(version)
        u32(UInt32(layer)); u32(UInt32(expertCount))
        u32(weights.gate.type); u32(weights.down.type)
        u64(weights.gate.bytes / UInt64(expertCount))
        u64(weights.up.bytes / UInt64(expertCount))
        u64(weights.down.bytes / UInt64(expertCount))
        u64(sourceFileSize)
        u64(weights.gate.absOffset)
        u64(weights.up.absOffset)
        u64(weights.down.absOffset)
        return bytes
    }

    /// Open and validate a layer bundle against the live weight map; nil
    /// quando assente, throws se esiste ma non corrisponde. Ordine di
    /// ricerca: prima la sezione nel PACK UNICO, poi il file per-layer
    /// legacy (leggibile finché non migrato nel pack alla prossima build).
    public static func open(directory: String, layer: Int,
                            weights: GLM52RoutedExpertWeights,
                            expertCount: Int,
                            sourcePath: String,
                            sourceFileSize: UInt64) throws
        -> GLM52ExpertBundle? {
        if let index = try GLM52SidecarPack.scan(
               directory: directory,
               fileName: GLM52SidecarPack.expertsFileName(
                   sourcePath: sourcePath),
               legacyFileName: GLM52SidecarPack.legacyExpertsFileName,
               sourceFileSize: sourceFileSize),
           let section = index.sections[layer] {
            let pack = try GLM52PayloadReader(path: index.path)
            return try open(
                reader: try pack.windowed(offset: section.offset,
                                          length: section.length),
                at: index.path + "#blk\(layer)", fromPack: true,
                layer: layer, weights: weights, expertCount: expertCount,
                sourceFileSize: sourceFileSize)
        }
        let bundlePath = path(directory: directory, layer: layer)
        guard FileManager.default.fileExists(atPath: bundlePath) else {
            return nil
        }
        let reader = try GLM52PayloadReader(path: bundlePath)
        return try open(reader: reader, at: bundlePath, fromPack: false,
                        layer: layer, weights: weights,
                        expertCount: expertCount,
                        sourceFileSize: sourceFileSize)
    }

    /// Validazione comune: file per-layer o vista finestrata sulla sezione
    /// del pack — stessi offset interni per costruzione.
    private static func open(reader: GLM52PayloadReader, at bundlePath: String,
                             fromPack: Bool, layer: Int,
                             weights: GLM52RoutedExpertWeights,
                             expertCount: Int,
                             sourceFileSize: UInt64) throws
        -> GLM52ExpertBundle? {
        guard reader.fileSize >= UInt64(headerBytes) else {
            throw GLM52ExpertBundleError.invalidHeader(path: bundlePath)
        }
        let headerDescriptor = GLM52WeightDescriptor(
            name: "bundle.header", type: GLM52TensorSchema.q8_0,
            dims: [], absOffset: 0, bytes: UInt64(headerBytes))
        let stored = try reader.bytes(of: headerDescriptor)
        let expected = header(layer: layer, weights: weights,
                              expertCount: expertCount,
                              sourceFileSize: sourceFileSize)
        guard stored == expected else {
            throw GLM52ExpertBundleError.sourceMismatch(path: bundlePath)
        }
        let perGate = Int(weights.gate.bytes) / expertCount
        let perUp = Int(weights.up.bytes) / expertCount
        let perDown = Int(weights.down.bytes) / expertCount
        let recordBytes = perGate + perUp + perDown
        let payloadBytes = UInt64(recordBytes * expertCount)
        guard reader.fileSize >= UInt64(headerBytes) + payloadBytes else {
            throw GLM52ExpertBundleError.shortFile(path: bundlePath)
        }
        return GLM52ExpertBundle(
            layer: layer, expertCount: expertCount,
            gateBytes: perGate, upBytes: perUp, downBytes: perDown,
            gateUpType: weights.gate.type, downType: weights.down.type,
            reader: reader, fromPack: fromPack,
            payload: GLM52WeightDescriptor(
                name: "bundle.blk\(layer).payload",
                type: weights.gate.type, dims: [],
                absOffset: UInt64(headerBytes), bytes: payloadBytes))
    }

    /// View over an expert payload EMBEDDED in a unified layer sidecar:
    /// the caller has already validated the identity header; this binds the
    /// shared reader to the section's offset. Reads behave exactly like a
    /// standalone bundle (one bounded pread per record).
    static func view(layer: Int, expertCount: Int, gateBytes: Int,
                     upBytes: Int, downBytes: Int, gateUpType: UInt32,
                     downType: UInt32, reader: GLM52PayloadReader,
                     payloadOffset: UInt64) -> GLM52ExpertBundle {
        let recordBytes = gateBytes + upBytes + downBytes
        return GLM52ExpertBundle(
            layer: layer, expertCount: expertCount,
            gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes,
            gateUpType: gateUpType, downType: downType,
            reader: reader, fromPack: false,
            payload: GLM52WeightDescriptor(
                name: "pack.blk\(layer).experts",
                type: gateUpType, dims: [],
                absOffset: payloadOffset,
                bytes: UInt64(recordBytes * expertCount)))
    }

    private init(layer: Int, expertCount: Int, gateBytes: Int, upBytes: Int,
                 downBytes: Int, gateUpType: UInt32, downType: UInt32,
                 reader: GLM52PayloadReader, fromPack: Bool,
                 payload: GLM52WeightDescriptor) {
        self.layer = layer
        self.expertCount = expertCount
        self.gateBytes = gateBytes
        self.upBytes = upBytes
        self.downBytes = downBytes
        self.gateUpType = gateUpType
        self.downType = downType
        self.reader = reader
        self.fromPack = fromPack
        self.payload = payload
    }

    /// One expert's record read straight into `destination` (the zero-copy
    /// staging path: destination is an MTLBuffer slice). Single bounded
    /// contiguous pread; safe concurrently for disjoint destinations.
    public func read(_ id: UInt32,
                     into destination: UnsafeMutableRawBufferPointer) throws {
        try reader.read(payload,
                        byteOffset: UInt64(Int(id) * recordBytes),
                        byteCount: UInt64(recordBytes),
                        into: destination)
    }

    /// One expert's record in a single bounded contiguous read.
    public func expert(_ id: UInt32) throws -> GLM52QuantizedExpert {
        let record = try reader.bytes(
            of: payload,
            byteOffset: UInt64(Int(id) * recordBytes),
            byteCount: UInt64(recordBytes))
        return GLM52QuantizedExpert(
            gateUpType: gateUpType, downType: downType,
            gate: Array(record[0..<gateBytes]),
            up: Array(record[gateBytes..<gateBytes + upBytes]),
            down: Array(record[(gateBytes + upBytes)...]))
    }

    // MARK: - Build

    /// Build one layer's bundle from the GGUF via the validated weight map.
    @discardableResult
    public static func build(directory: String, layer: Int,
                             weightMap: GLM52WeightMap,
                             reader: GLM52PayloadReader,
                             progress: ((Int, Int) -> Void)? = nil) throws
        -> Bool {
        try build(directory: directory, layer: layer,
                  weights: weightMap.routedExperts(layer: layer),
                  expertCount: Int(weightMap.configuration.shape.nExpert),
                  reader: reader, progress: progress)
    }

    public struct BuildSummary: Sendable {
        public let created: Int
        public let alreadyValid: Int
        public let remaining: Int
        /// Non-nil when the run stopped early (disk space / layer cap).
        public let stoppedBecause: String?
    }

    /// PARTIAL-friendly bulk build: bundles are per layer and the provider
    /// falls back to the GGUF wherever one is missing, so ANY prefix of
    /// bundles is useful (each covers 1/75 of the per-token expert seeks).
    /// This walks the sparse layers and stops GRACEFULLY — never mid-file —
    /// when the free space would drop under `minFreeBytes` (default 8 GiB)
    /// or when `maxBundles` TOTAL bundles exist (created now or already
    /// valid): rerunning with the same cap builds nothing more. A partially
    /// written .part is removed on error; rerunning resumes at the first
    /// missing layer.
    public static func buildAvailable(directory: String,
                                      weightMap: GLM52WeightMap,
                                      reader: GLM52PayloadReader,
                                      maxBundles: Int? = nil,
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
            if let cap = maxBundles, created + valid >= cap {
                stopped = "raggiunto il tetto di \(cap) bundle totali"
                break
            }
            let weights = try weightMap.routedExperts(layer: layer)
            let perLayer = weights.gate.bytes + weights.up.bytes
                + weights.down.bytes
            let free = ((try? FileManager.default.attributesOfFileSystem(
                forPath: directory))?[.systemFreeSize] as? UInt64) ?? 0
            let existing = (try? open(
                directory: directory, layer: layer, weights: weights,
                expertCount: Int(shape.nExpert),
                sourcePath: reader.path,
                sourceFileSize: reader.fileSize)) ?? nil
            if existing == nil, free < perLayer + minFreeBytes {
                stopped = "spazio disco insufficiente (liberi "
                    + "\(free >> 30) GiB, servono "
                    + "\((perLayer + minFreeBytes) >> 30) GiB)"
                break
            }
            do {
                let built = try build(
                    directory: directory, layer: layer,
                    weights: weights, expertCount: Int(shape.nExpert),
                    reader: reader)
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

    /// Build one layer's bundle (skip if a valid one exists). Atomic:
    /// written to `.part`, fsynced, then renamed. Returns whether a build
    /// happened (false = valid bundle already present).
    @discardableResult
    public static func build(directory: String, layer: Int,
                             weights: GLM52RoutedExpertWeights,
                             expertCount: Int,
                             reader: GLM52PayloadReader,
                             progress: ((Int, Int) -> Void)? = nil) throws
        -> Bool {
        let packName = GLM52SidecarPack.expertsFileName(
            sourcePath: reader.path)
        let existing = (try? open(directory: directory, layer: layer,
                                  weights: weights,
                                  expertCount: expertCount,
                                  sourcePath: reader.path,
                                  sourceFileSize: reader.fileSize)) ?? nil
        if let existing {
            if existing.fromPack { return false }
            // Bundle valido ma ancora file per-layer: MIGRAZIONE nel pack
            // unico — copia grezza e rimozione del file solo dopo.
            try GLM52SidecarPack.append(
                directory: directory, fileName: packName,
                legacyFileName: GLM52SidecarPack.legacyExpertsFileName,
                layer: layer,
                contentsOf: path(directory: directory, layer: layer),
                sourceFileSize: reader.fileSize)
            try? FileManager.default.removeItem(
                atPath: path(directory: directory, layer: layer))
            return true
        }
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        let finalPath = path(directory: directory, layer: layer)
        let partPath = finalPath + ".part"
        FileManager.default.createFile(atPath: partPath, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: partPath) else {
            throw GLM52PayloadReaderError.cannotOpen(path: partPath, code: 0)
        }

        let planner = try GLM52ExpertStreamPlanner(
            layer: layer, weights: weights)
        try handle.write(contentsOf: Data(header(
            layer: layer, weights: weights, expertCount: expertCount,
            sourceFileSize: reader.fileSize)))
        for expert in 0..<expertCount {
            let plan = try planner.plan(selectedExperts: [UInt32(expert)])
            let layout = try reader.packedLayout(of: plan)
            var record = [UInt8](repeating: 0, count: layout.totalBytes)
            _ = try record.withUnsafeMutableBytes {
                try reader.read(plan: plan, into: $0)
            }
            try handle.write(contentsOf: Data(record))
            progress?(expert + 1, expertCount)
        }
        try handle.synchronize()
        try handle.close()
        // Nel PACK UNICO (stesso contenitore a sezioni del sidecar Q4): la
        // sezione è l'immagine del .part; un eventuale file per-layer
        // stantio viene reclamato solo DOPO che il pack è durabile.
        try GLM52SidecarPack.append(
            directory: directory, fileName: packName,
            legacyFileName: GLM52SidecarPack.legacyExpertsFileName,
            layer: layer,
            contentsOf: partPath, sourceFileSize: reader.fileSize)
        try? FileManager.default.removeItem(atPath: partPath)
        try? FileManager.default.removeItem(atPath: finalPath)
        return true
    }
}
