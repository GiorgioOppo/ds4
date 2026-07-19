import DS4Core
import Foundation

// UNIFIED per-layer sidecar (version 2) — one file per sparse layer holding
// BOTH acceleration payloads:
//
// 1. The streamed big tensors requantized Q8_0 → Q4_K (attention
//    projections, indexer, shared expert; keyB/valueB verbatim — their
//    kernels are Q8-hardwired): halves the dominant per-token SSD stream.
// 2. The routed expert records repacked CONTIGUOUSLY (gate|up|down per
//    expert, the exact legacy `.experts` bundle layout): one bounded pread
//    per selected expert.
//
// One file, one identity header, one reader per layer. Version 1 files
// (tensors only) remain readable; a build MIGRATES automatically: tensor
// payloads are copied from a valid v1 (no re-requantization), expert
// records from a valid legacy bundle (no GGUF gather), the v1 is replaced
// in place by the rename and the legacy `.experts` file is deleted after
// the new pack is validated — net disk usage stays put.
//
// Same contract as always: build is explicit, resumable, atomic; ANY
// subset of layers is useful; a stale or foreign sidecar is refused loudly.

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

/// PACK UNICO — un solo file per modello, come il .expbundle DeepSeek, con
/// la semantica a checkpoint del q4cache DeepSeek: dentro ci sono le stesse
/// sezioni per-layer del formato v2, appese una alla volta. Ogni sezione è
/// l'immagine ESATTA di un file .layerq4 (il reader la serve tramite una
/// vista finestrata, quindi tutta la validazione v2 si riusa invariata).
/// La scansione si ferma alla prima sezione invalida: una coda strappata da
/// una build interrotta viene troncata e riscritta dalla build successiva.
/// QUALSIASI sottoinsieme di layer resta utile; identità = dimensione GGUF.
public enum GLM52SidecarPack {
    public static let magic: UInt32 = 0x314B_5047        // "GPK1"
    static let sectionMagic: UInt32 = 0x4345_5347        // "GSEC"
    static let headerBytes: UInt64 = 16
    static let sectionHeaderBytes: UInt64 = 16
    /// Nomi standard, parità DeepSeek: il sidecar Q4 unificato è
    /// "<gguf>.q4dense" e il bundle esperti "<gguf>.expbundle", nella
    /// directory risolta. I pack coi nomi storici glm52-*.glmsidecar
    /// vengono RINOMINATI al nome nuovo alla prima scansione (stesso
    /// volume: rename atomico, le sezioni già costruite si conservano).
    public static func q4FileName(sourcePath: String) -> String {
        (sourcePath as NSString).lastPathComponent + ".q4dense"
    }
    public static func expertsFileName(sourcePath: String) -> String {
        (sourcePath as NSString).lastPathComponent + ".expbundle"
    }
    static let legacyQ4FileName = "glm52-pack.glmsidecar"
    static let legacyExpertsFileName = "glm52-experts.glmsidecar"

    public static func path(directory: String, fileName: String) -> String {
        (directory as NSString).appendingPathComponent(fileName)
    }

    struct Index {
        let path: String
        let sections: [Int: (offset: UInt64, length: UInt64)]
        /// Fine dell'ultima sezione valida: punto di troncamento/append.
        let validEnd: UInt64
    }

    /// Scansiona il pack: nil se il file non esiste; throws se esiste ma
    /// appartiene a un ALTRO GGUF (rifiuto a voce alta, mai silenzioso).
    /// `legacyFileName`: nome storico da rinominare al nome nuovo quando
    /// il file nuovo manca (migrazione one-shot, atomica).
    static func scan(directory: String,
                     fileName: String,
                     legacyFileName: String? = nil,
                     sourceFileSize: UInt64) throws
        -> Index? {
        let packPath = path(directory: directory, fileName: fileName)
        let fm = FileManager.default
        if !fm.fileExists(atPath: packPath), let legacyFileName {
            let legacyPath = path(directory: directory,
                                  fileName: legacyFileName)
            if fm.fileExists(atPath: legacyPath) {
                try fm.moveItem(atPath: legacyPath, toPath: packPath)
            }
        }
        guard fm.fileExists(atPath: packPath) else {
            return nil
        }
        let reader = try GLM52PayloadReader(path: packPath)
        guard reader.fileSize >= headerBytes else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: packPath)
        }
        func bytes(_ offset: UInt64, _ count: UInt64) throws -> [UInt8] {
            try reader.bytes(of: GLM52WeightDescriptor(
                name: "pack.header", type: GLM52TensorSchema.q8_0,
                dims: [], absOffset: offset, bytes: count))
        }
        let head = try bytes(0, headerBytes)
        func u32(_ o: Int, of b: [UInt8]) -> UInt32 {
            b.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)
            }
        }
        func u64(_ o: Int, of b: [UInt8]) -> UInt64 {
            b.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: o, as: UInt64.self)
            }
        }
        guard u32(0, of: head) == magic, u32(4, of: head) == 1 else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: packPath)
        }
        guard u64(8, of: head) == sourceFileSize else {
            throw GLM52LayerQuantSidecarError.sourceMismatch(path: packPath)
        }
        var sections: [Int: (offset: UInt64, length: UInt64)] = [:]
        var cursor = headerBytes
        while cursor + sectionHeaderBytes <= reader.fileSize {
            let header = try bytes(cursor, sectionHeaderBytes)
            let layer = u32(4, of: header)
            let length = u64(8, of: header)
            guard u32(0, of: header) == sectionMagic, layer < 4096,
                  length > 0,
                  length <= reader.fileSize
                      - cursor - sectionHeaderBytes else { break }
            // Ultima vince: un upgrade v1→v2 appende una sezione nuova per
            // lo stesso layer (lo spazio della vecchia resta: raro, accettato).
            sections[Int(layer)] = (cursor + sectionHeaderBytes, length)
            cursor = (cursor + sectionHeaderBytes + length + 15) & ~15
        }
        return Index(path: packPath, sections: sections, validEnd: cursor)
    }

    /// Appende una sezione (l'immagine di un file .layerq4) al pack: crea il
    /// pack se manca, TRONCA una coda strappata, scrive header + contenuto +
    /// padding e sincronizza. Il chiamante cancella la sorgente solo dopo.
    static func append(directory: String,
                       fileName: String,
                       legacyFileName: String? = nil,
                       layer: Int,
                       contentsOf source: String,
                       sourceFileSize: UInt64) throws {
        let fm = FileManager.default
        let packPath = path(directory: directory, fileName: fileName)
        var validEnd = headerBytes
        if let index = try scan(directory: directory, fileName: fileName,
                                legacyFileName: legacyFileName,
                                sourceFileSize: sourceFileSize) {
            validEnd = index.validEnd
        } else {
            try fm.createDirectory(atPath: directory,
                                   withIntermediateDirectories: true)
            fm.createFile(atPath: packPath, contents: nil)
            guard let handle = FileHandle(forWritingAtPath: packPath) else {
                throw GLM52PayloadReaderError.cannotOpen(path: packPath,
                                                         code: errno)
            }
            var head = Data()
            withUnsafeBytes(of: magic.littleEndian) { head.append(contentsOf: $0) }
            withUnsafeBytes(of: UInt32(1).littleEndian) { head.append(contentsOf: $0) }
            withUnsafeBytes(of: sourceFileSize.littleEndian) { head.append(contentsOf: $0) }
            try handle.write(contentsOf: head)
            try handle.synchronize()
            try handle.close()
        }
        guard let sourceSize = (try? fm.attributesOfItem(atPath: source))?[
            .size] as? UInt64, sourceSize > 0 else {
            throw GLM52LayerQuantSidecarError.shortFile(path: source)
        }
        guard let handle = FileHandle(forWritingAtPath: packPath),
              let input = FileHandle(forReadingAtPath: source) else {
            throw GLM52PayloadReaderError.cannotOpen(path: packPath,
                                                     code: errno)
        }
        defer { try? handle.close(); try? input.close() }
        try handle.truncate(atOffset: validEnd)
        try handle.seek(toOffset: validEnd)
        var header = Data()
        withUnsafeBytes(of: sectionMagic.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(layer).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: sourceSize.littleEndian) { header.append(contentsOf: $0) }
        try handle.write(contentsOf: header)
        while let chunk = try input.read(upToCount: 64 << 20),
              !chunk.isEmpty {
            try handle.write(contentsOf: chunk)
        }
        let end = validEnd + sectionHeaderBytes + sourceSize
        let padded = (end + 15) & ~15
        if padded > end {
            try handle.write(contentsOf: Data(count: Int(padded - end)))
        }
        try handle.synchronize()
    }
}

/// One validated, opened per-layer sidecar (v1: tensors only; v2: tensors
/// plus the embedded expert payload).
public final class GLM52LayerQuantSidecar {
    public static let magic: UInt32 = 0x3451_4C47   // "GLQ4"
    public static let version: UInt32 = 2
    static let fixedHeaderBytes = 24
    static let expertBlockBytes = 72
    static let entryBytes = 40

    public let layer: Int
    public let path: String
    /// True quando la sezione vive nel pack unico; false = file per-layer
    /// legacy (candidato alla migrazione nel pack alla prossima build).
    public let fromPack: Bool
    /// Synthetic streamed-tensor view whose descriptors point INTO the
    /// sidecar file (requantized types included).
    public let tensors: GLM52StreamedLayerTensors
    /// Bundle-compatible view over the embedded expert payload (nil on a
    /// version-1 file).
    public let expertView: GLM52ExpertBundle?
    /// Open pread reader over the sidecar file (o la vista finestrata sulla
    /// sezione del pack), handed to the streamer.
    public let reader: GLM52PayloadReader

    public static func fileName(layer: Int) -> String {
        "glm52-blk\(layer).layerq4"
    }

    public static func path(directory: String, layer: Int) -> String {
        (directory as NSString).appendingPathComponent(fileName(layer: layer))
    }

    private init(layer: Int, path: String, fromPack: Bool,
                 tensors: GLM52StreamedLayerTensors,
                 expertView: GLM52ExpertBundle?,
                 reader: GLM52PayloadReader) {
        self.layer = layer
        self.path = path
        self.fromPack = fromPack
        self.tensors = tensors
        self.expertView = expertView
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

    /// Open and validate a layer sidecar against the live GGUF; nil when
    /// assente, throws se esiste ma non corrisponde. Ordine di ricerca:
    /// prima la sezione nel PACK UNICO, poi il file per-layer legacy (che
    /// resta leggibile e viene migrato nel pack alla prossima build).
    /// Accepts version 1 (tensors only — `expertView` nil) and version 2.
    public static func open(directory: String, layer: Int,
                            source: GLM52StreamedLayerTensors,
                            routed: GLM52RoutedExpertWeights,
                            expertCount: Int,
                            sourcePath: String,
                            sourceFileSize: UInt64) throws
        -> GLM52LayerQuantSidecar? {
        if let index = try GLM52SidecarPack.scan(
               directory: directory,
               fileName: GLM52SidecarPack.q4FileName(sourcePath: sourcePath),
               legacyFileName: GLM52SidecarPack.legacyQ4FileName,
               sourceFileSize: sourceFileSize),
           let section = index.sections[layer] {
            let pack = try GLM52PayloadReader(path: index.path)
            return try open(
                reader: try pack.windowed(offset: section.offset,
                                          length: section.length),
                at: index.path + "#blk\(layer)", fromPack: true,
                layer: layer, source: source, routed: routed,
                expertCount: expertCount, sourceFileSize: sourceFileSize)
        }
        let sidecarPath = path(directory: directory, layer: layer)
        guard FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }
        let reader = try GLM52PayloadReader(path: sidecarPath)
        return try open(reader: reader, at: sidecarPath, fromPack: false,
                        layer: layer, source: source, routed: routed,
                        expertCount: expertCount,
                        sourceFileSize: sourceFileSize)
    }

    /// Validazione comune: il reader è il file per-layer O la vista
    /// finestrata sulla sezione del pack — gli offset interni sono identici
    /// per costruzione (la sezione è l'immagine esatta del file).
    private static func open(reader: GLM52PayloadReader, at sidecarPath: String,
                             fromPack: Bool, layer: Int,
                             source: GLM52StreamedLayerTensors,
                             routed: GLM52RoutedExpertWeights,
                             expertCount: Int,
                             sourceFileSize: UInt64) throws
        -> GLM52LayerQuantSidecar? {
        let entries = orderedEntries(of: source)
        guard reader.fileSize >= UInt64(fixedHeaderBytes) else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: sidecarPath)
        }
        func headerBytes(version: UInt32) -> Int {
            fixedHeaderBytes
                + (version >= 2 ? expertBlockBytes : 0)
                + entryBytes * entries.count
        }
        let probe = try reader.bytes(of: GLM52WeightDescriptor(
            name: "sidecar.fixed", type: GLM52TensorSchema.q8_0,
            dims: [], absOffset: 0, bytes: UInt64(fixedHeaderBytes)))
        func probeU32(_ offset: Int) -> UInt32 {
            probe.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
        }
        let fileVersion = probeU32(4)
        guard probeU32(0) == magic, fileVersion >= 1,
              fileVersion <= version,
              probeU32(8) == UInt32(layer),
              probeU32(12) == UInt32(entries.count),
              reader.fileSize >= UInt64(headerBytes(version: fileVersion))
        else {
            throw GLM52LayerQuantSidecarError.invalidHeader(path: sidecarPath)
        }
        let header = try reader.bytes(of: GLM52WeightDescriptor(
            name: "sidecar.header", type: GLM52TensorSchema.q8_0,
            dims: [], absOffset: 0,
            bytes: UInt64(headerBytes(version: fileVersion))))
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
        guard u64(16) == sourceFileSize else {
            throw GLM52LayerQuantSidecarError.sourceMismatch(path: sidecarPath)
        }

        // Expert block (v2 only), validated against the live routed weights.
        var expertView: GLM52ExpertBundle?
        if fileVersion >= 2 {
            let block = fixedHeaderBytes
            let perGate = routed.gate.bytes / UInt64(expertCount)
            let perUp = routed.up.bytes / UInt64(expertCount)
            let perDown = routed.down.bytes / UInt64(expertCount)
            guard u32(block) == UInt32(expertCount),
                  u32(block + 4) == routed.gate.type,
                  u32(block + 8) == routed.down.type,
                  u64(block + 16) == perGate,
                  u64(block + 24) == perUp,
                  u64(block + 32) == perDown,
                  u64(block + 48) == routed.gate.absOffset,
                  u64(block + 56) == routed.up.absOffset,
                  u64(block + 64) == routed.down.absOffset else {
                throw GLM52LayerQuantSidecarError.sourceMismatch(
                    path: sidecarPath)
            }
            let expertOffset = u64(block + 40)
            let recordBytes = perGate + perUp + perDown
            let payloadBytes = recordBytes * UInt64(expertCount)
            guard expertOffset + payloadBytes <= reader.fileSize else {
                throw GLM52LayerQuantSidecarError.shortFile(path: sidecarPath)
            }
            expertView = GLM52ExpertBundle.view(
                layer: layer, expertCount: expertCount,
                gateBytes: Int(perGate), upBytes: Int(perUp),
                downBytes: Int(perDown),
                gateUpType: routed.gate.type, downType: routed.down.type,
                reader: reader, payloadOffset: expertOffset)
        }

        let entriesBase = fixedHeaderBytes
            + (fileVersion >= 2 ? expertBlockBytes : 0)
        var synthetic: [GLM52WeightDescriptor] = []
        for (rank, entry) in entries.enumerated() {
            let base = entriesBase + entryBytes * rank
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
                  newOffset >= UInt64(entriesBase
                      + entryBytes * entries.count),
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
            layer: layer, path: sidecarPath, fromPack: fromPack,
            tensors: tensors, expertView: expertView, reader: reader)
    }

    // MARK: - Build

    /// Build one layer's UNIFIED sidecar (skip if a valid v2 exists; a v1
    /// gets upgraded in place by the rename). Atomic: `.part`, fsync,
    /// rename. Fast paths: tensor payloads copied from a valid v1, expert
    /// records copied from a valid legacy `.experts` bundle — which is
    /// DELETED after the new pack is in place. Returns whether a build
    /// happened.
    @discardableResult
    public static func build(directory: String, layer: Int,
                             source: GLM52StreamedLayerTensors,
                             routed: GLM52RoutedExpertWeights,
                             expertCount: Int,
                             reader: GLM52PayloadReader,
                             legacyBundleDirectory: String? = nil) throws
        -> Bool {
        let packName = GLM52SidecarPack.q4FileName(sourcePath: reader.path)
        let existing = (try? open(directory: directory, layer: layer,
                                  source: source, routed: routed,
                                  expertCount: expertCount,
                                  sourcePath: reader.path,
                                  sourceFileSize: reader.fileSize)) ?? nil
        if let existing, existing.expertView != nil {
            if existing.fromPack { return false }
            // V2 valido ma ancora file per-layer: MIGRAZIONE nel pack unico
            // — copia grezza della sezione e rimozione del file solo dopo
            // (uso disco netto invariato, transiente = un layer).
            try GLM52SidecarPack.append(
                directory: directory, fileName: packName,
                legacyFileName: GLM52SidecarPack.legacyQ4FileName,
                layer: layer, contentsOf: existing.path,
                sourceFileSize: reader.fileSize)
            try? FileManager.default.removeItem(atPath: existing.path)
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
        let legacyBundle = try legacyBundleDirectory.flatMap {
            try GLM52ExpertBundle.open(
                directory: $0, layer: layer, weights: routed,
                expertCount: expertCount, sourcePath: reader.path,
                sourceFileSize: reader.fileSize)
        }

        let entries = orderedEntries(of: source)
        let headerBytes = fixedHeaderBytes + expertBlockBytes
            + entryBytes * entries.count
        let perGate = routed.gate.bytes / UInt64(expertCount)
        let perUp = routed.up.bytes / UInt64(expertCount)
        let perDown = routed.down.bytes / UInt64(expertCount)
        let recordBytes = Int(perGate + perUp + perDown)

        // Payload layout: tensors first (16-aligned), expert records last.
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
            plans.append((entry, cursor, bytes, type))
            cursor += bytes
        }
        cursor = (cursor + 15) & ~15
        let expertOffset = cursor

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
        u32(UInt32(expertCount)); u32(routed.gate.type)
        u32(routed.down.type); u32(0)
        u64(perGate); u64(perUp); u64(perDown)
        u64(expertOffset)
        u64(routed.gate.absOffset); u64(routed.up.absOffset)
        u64(routed.down.absOffset)
        for plan in plans {
            u32(plan.type); u32(0)
            u64(plan.bytes); u64(plan.offset)
            u64(plan.entry.descriptor.absOffset)
            u64(plan.entry.descriptor.bytes)
        }
        try handle.write(contentsOf: Data(header))

        // Tensor payloads: copy from a valid v1 sidecar (already Q4) or
        // requantize/copy from the GGUF.
        for (rank, plan) in plans.enumerated() {
            try handle.seek(toOffset: plan.offset)
            if let existing {
                let previous = orderedTensor(existing.tensors, rank: rank)
                if previous.type == plan.type,
                   previous.bytes == plan.bytes {
                    try handle.write(contentsOf: Data(
                        try existing.reader.bytes(of: previous)))
                    continue
                }
            }
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

        // Expert records: copy from the legacy bundle when present (already
        // packed in this exact layout), otherwise gather from the GGUF.
        try handle.seek(toOffset: expertOffset)
        var record = [UInt8](repeating: 0, count: recordBytes)
        let planner = try GLM52ExpertStreamPlanner(
            layer: layer, weights: routed)
        for expert in 0..<expertCount {
            try record.withUnsafeMutableBytes { destination in
                if let legacyBundle {
                    try legacyBundle.read(UInt32(expert), into: destination)
                } else {
                    let plan = try planner.plan(
                        selectedExperts: [UInt32(expert)])
                    _ = try reader.read(plan: plan, into: destination,
                                        concurrent: true)
                }
            }
            try handle.write(contentsOf: Data(record))
        }
        try handle.synchronize()
        try handle.close()
        // Nel PACK UNICO: la sezione è l'immagine del .part appena scritto.
        try GLM52SidecarPack.append(directory: directory, fileName: packName,
                                    legacyFileName:
                                        GLM52SidecarPack.legacyQ4FileName,
                                    layer: layer, contentsOf: partPath,
                                    sourceFileSize: reader.fileSize)
        try? FileManager.default.removeItem(atPath: partPath)
        // Migration: un eventuale file per-layer (v1 appena assorbito o v2
        // stantio) e il bundle legacy sono ora ridondanti — reclamati solo
        // DOPO che il pack è durabile su disco. Un bundle nel pack esperti
        // non è reclamabile per-sezione (no-op qui): si libera cancellando
        // l'intero glm52-experts.glmsidecar quando tutti i layer sono
        // unificati.
        try? FileManager.default.removeItem(atPath: finalPath)
        if let legacyBundleDirectory, legacyBundle != nil,
           !legacyBundle!.fromPack {
            try? FileManager.default.removeItem(
                atPath: GLM52ExpertBundle.path(
                    directory: legacyBundleDirectory, layer: layer))
        }
        return true
    }

    /// The rank-th tensor of a sidecar view, in `orderedEntries` order.
    private static func orderedTensor(_ tensors: GLM52StreamedLayerTensors,
                                      rank: Int) -> GLM52WeightDescriptor {
        var list = [tensors.qA, tensors.qB, tensors.kvA, tensors.keyB,
                    tensors.valueB, tensors.attnOutput]
        if let key = tensors.indexerKey { list.append(key) }
        if let queryB = tensors.indexerQueryB { list.append(queryB) }
        list.append(tensors.sharedGate)
        list.append(tensors.sharedUp)
        list.append(tensors.sharedDown)
        return list[rank]
    }

    public struct BuildSummary: Sendable {
        public let created: Int
        public let alreadyValid: Int
        public let remaining: Int
        public let stoppedBecause: String?
    }

    /// Bulk build over the sparse layers with the usual semantics:
    /// `maxSidecars` is a TOTAL cap (existing valid v2 count toward it),
    /// the walk stops gracefully under `minFreeBytes`, any prefix is
    /// useful, a partial `.part` is removed on error. Legacy per-layer
    /// files are migrated into the pack (and the `.experts` deleted) as
    /// the walk proceeds.
    public static func buildAvailable(directory: String,
                                      weightMap: GLM52WeightMap,
                                      reader: GLM52PayloadReader,
                                      legacyBundleDirectory: String? = nil,
                                      maxSidecars: Int? = nil,
                                      minFreeBytes: UInt64 = 8 << 30,
                                      layerProgress: ((Int, Bool) -> Void)?
                                          = nil) throws -> BuildSummary {
        let shape = weightMap.configuration.shape
        let sparse = Int(shape.nLeadingDense)..<Int(shape.inferenceLayerCount)
        let expertCount = Int(shape.nExpert)
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
            let routed = try weightMap.routedExperts(layer: layer)
            let tensorBytes = orderedEntries(of: source).reduce(UInt64(0)) {
                $0 + (converts($1.descriptor, requant: $1.requant)
                    ? convertedBytes($1.descriptor) : $1.descriptor.bytes)
            }
            let perLayer = tensorBytes + routed.gate.bytes
                + routed.up.bytes + routed.down.bytes
            let free = ((try? FileManager.default.attributesOfFileSystem(
                forPath: directory))?[.systemFreeSize] as? UInt64) ?? 0
            let existing = (try? open(
                directory: directory, layer: layer, source: source,
                routed: routed, expertCount: expertCount,
                sourcePath: reader.path,
                sourceFileSize: reader.fileSize)) ?? nil
            if existing?.expertView == nil, free < perLayer + minFreeBytes {
                stopped = "spazio disco insufficiente (liberi "
                    + "\(free >> 30) GiB, servono "
                    + "\((perLayer + minFreeBytes) >> 30) GiB)"
                break
            }
            do {
                let built = try build(
                    directory: directory, layer: layer, source: source,
                    routed: routed, expertCount: expertCount,
                    reader: reader,
                    legacyBundleDirectory: legacyBundleDirectory)
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
