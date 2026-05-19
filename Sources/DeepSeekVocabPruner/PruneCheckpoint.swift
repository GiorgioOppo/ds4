import Foundation
import CryptoKit

/// Stato persistente del job di vocab pruning, salvato in
/// `<outputDir>/checkpoint/vocab_pruner.json` per permettere il
/// ripristino dopo un'interruzione (kill, crash, cancel utente).
/// Una eventuale versione legacy in `<outputDir>/.vocab_pruner_checkpoint.json`
/// viene migrata automaticamente al primo load (vedi `load(from:)`).
///
/// Validità: lo `specHash` viene confrontato all'inizio del run
/// successivo. Se non corrisponde (es. l'utente ha cambiato corpus
/// o coverage), il checkpoint viene rifiutato e il job riparte
/// da zero.
public struct PruneCheckpoint: Codable, Sendable {

    public enum Phase: String, Codable, Sendable {
        case analyzer    // ancora in Fase 1
        case rewriter    // Fase 1 completa, in Fase 2
        case done        // tutto completo (il file viene cancellato a fine job)
    }

    public var phase: Phase
    public var specHash: String
    public var savedAt: Date

    /// Stato della Fase 1. `nil` quando la fase è già passata.
    public var analyzer: AnalyzerState?

    /// Decisione consolidata, salvata appena Fase 1 finisce.
    /// `nil` finché siamo in Fase 1.
    public var decision: KeepDecision?

    /// Stato della Fase 2. `nil` finché siamo in Fase 1.
    public var rewriter: RewriterState?

    public init(phase: Phase, specHash: String,
                analyzer: AnalyzerState? = nil,
                decision: KeepDecision? = nil,
                rewriter: RewriterState? = nil,
                savedAt: Date = Date()) {
        self.phase = phase
        self.specHash = specHash
        self.analyzer = analyzer
        self.decision = decision
        self.rewriter = rewriter
        self.savedAt = savedAt
    }

    public struct AnalyzerState: Codable, Sendable {
        /// Path assoluti dei file già processati. Usati come filtro
        /// in `VocabAnalyzer.analyze(alreadyProcessed:)`.
        public var processedFiles: [String]
        /// Count map parziale (id → count). Serializzato come due
        /// array paralleli perché `[Int: Int]` Codable di default
        /// usa String keys.
        public var partialCounts: [Int: Int]
        public var linesScanned: Int
        public var tokensScanned: Int

        /// File attualmente in elaborazione e progressi intra-file
        /// (solo per `concurrency == 1`). Save periodico ogni
        /// ~`tokenBatchThreshold` token. Al resume, l'analyzer
        /// riprende dalla `lineOffset+1` di questo file con i
        /// `partialCountsForFile` già accumulati. `nil` quando
        /// l'analyzer non sta processando un file specifico
        /// (es. fra un file e l'altro, o appena prima del primo).
        public var inFlightFile: InFlightFile?

        /// File in elaborazione nel branch single-file chunked
        /// (`concurrency > 1` con un solo file nel corpus). Tiene
        /// traccia dei chunk completati per permettere il resume
        /// chunk-per-chunk (i chunk già completati vengono saltati
        /// al rilancio; solo i restanti vengono processati).
        /// `inFlightFile` e `chunkedFile` sono mutually exclusive
        /// — al massimo uno dei due è settato per un dato file.
        public var chunkedFile: ChunkedFileState?

        public init(processedFiles: [String] = [],
                    partialCounts: [Int: Int] = [:],
                    linesScanned: Int = 0,
                    tokensScanned: Int = 0,
                    inFlightFile: InFlightFile? = nil,
                    chunkedFile: ChunkedFileState? = nil) {
            self.processedFiles = processedFiles
            self.partialCounts = partialCounts
            self.linesScanned = linesScanned
            self.tokensScanned = tokensScanned
            self.inFlightFile = inFlightFile
            self.chunkedFile = chunkedFile
        }

        private enum CodingKeys: String, CodingKey {
            case processedFiles, countsKeys, countsValues,
                 linesScanned, tokensScanned, inFlightFile, chunkedFile
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.processedFiles = try c.decode([String].self,
                                                forKey: .processedFiles)
            let ks = try c.decode([Int].self, forKey: .countsKeys)
            let vs = try c.decode([Int].self, forKey: .countsValues)
            var map: [Int: Int] = [:]
            for (k, v) in zip(ks, vs) { map[k] = v }
            self.partialCounts = map
            self.linesScanned = try c.decode(Int.self, forKey: .linesScanned)
            self.tokensScanned = try c.decode(Int.self, forKey: .tokensScanned)
            self.inFlightFile = try? c.decode(InFlightFile.self,
                                                forKey: .inFlightFile)
            self.chunkedFile = try? c.decode(ChunkedFileState.self,
                                              forKey: .chunkedFile)
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(processedFiles, forKey: .processedFiles)
            let pairs = partialCounts.sorted { $0.key < $1.key }
            try c.encode(pairs.map { $0.key }, forKey: .countsKeys)
            try c.encode(pairs.map { $0.value }, forKey: .countsValues)
            try c.encode(linesScanned, forKey: .linesScanned)
            try c.encode(tokensScanned, forKey: .tokensScanned)
            try c.encodeIfPresent(inFlightFile, forKey: .inFlightFile)
            try c.encodeIfPresent(chunkedFile, forKey: .chunkedFile)
        }
    }

    /// Stato per il save intra-file in modalità single-file chunked
    /// (`concurrency > 1` con un solo file).
    ///
    /// Il file viene diviso in N chunk di byte, allineati al newline.
    /// Ogni chunk è processato in parallel da un thread; quando un
    /// thread completa il proprio chunk, fa save al checkpoint
    /// aggiornando `completedChunks` + `partialCountsForFile`.
    ///
    /// Al resume:
    /// - L'analyzer ricalcola i boundary dato `concurrency` corrente.
    /// - Se `fileSize` e `boundaries` corrispondono ai salvati,
    ///   skippa i chunk in `completedChunks` (i loro counts sono
    ///   già in `partialCountsForFile`) e processa i restanti.
    /// - Se non corrispondono (file modificato o concurrency
    ///   cambiata), invalida il checkpoint e riparte da capo.
    public struct ChunkedFileState: Codable, Sendable {
        public var path: String
        public var fileSize: UInt64
        public var boundaries: [Int]
        public var completedChunks: [Int]
        public var partialCountsForFile: [Int: Int]
        public var tokensInFile: Int
        public var linesInFile: Int

        public init(path: String,
                    fileSize: UInt64,
                    boundaries: [Int],
                    completedChunks: [Int] = [],
                    partialCountsForFile: [Int: Int] = [:],
                    tokensInFile: Int = 0,
                    linesInFile: Int = 0) {
            self.path = path
            self.fileSize = fileSize
            self.boundaries = boundaries
            self.completedChunks = completedChunks
            self.partialCountsForFile = partialCountsForFile
            self.tokensInFile = tokensInFile
            self.linesInFile = linesInFile
        }

        private enum CodingKeys: String, CodingKey {
            case path, fileSize, boundaries, completedChunks,
                 countsKeys, countsValues, tokensInFile, linesInFile
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.path = try c.decode(String.self, forKey: .path)
            self.fileSize = try c.decode(UInt64.self, forKey: .fileSize)
            self.boundaries = try c.decode([Int].self, forKey: .boundaries)
            self.completedChunks = try c.decode([Int].self, forKey: .completedChunks)
            let ks = try c.decode([Int].self, forKey: .countsKeys)
            let vs = try c.decode([Int].self, forKey: .countsValues)
            var map: [Int: Int] = [:]
            for (k, v) in zip(ks, vs) { map[k] = v }
            self.partialCountsForFile = map
            self.tokensInFile = try c.decode(Int.self, forKey: .tokensInFile)
            self.linesInFile = try c.decode(Int.self, forKey: .linesInFile)
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(path, forKey: .path)
            try c.encode(fileSize, forKey: .fileSize)
            try c.encode(boundaries, forKey: .boundaries)
            try c.encode(completedChunks, forKey: .completedChunks)
            let pairs = partialCountsForFile.sorted { $0.key < $1.key }
            try c.encode(pairs.map { $0.key }, forKey: .countsKeys)
            try c.encode(pairs.map { $0.value }, forKey: .countsValues)
            try c.encode(tokensInFile, forKey: .tokensInFile)
            try c.encode(linesInFile, forKey: .linesInFile)
        }
    }

    /// Progresso intra-file per un singolo file (Fase 1
    /// sequential). `lineOffset` indica fino a quale linea
    /// (esclusiva) i `partialCountsForFile` coprono.
    public struct InFlightFile: Codable, Sendable {
        public var path: String
        public var lineOffset: Int
        public var tokensInFile: Int
        public var partialCountsForFile: [Int: Int]

        public init(path: String,
                    lineOffset: Int = 0,
                    tokensInFile: Int = 0,
                    partialCountsForFile: [Int: Int] = [:]) {
            self.path = path
            self.lineOffset = lineOffset
            self.tokensInFile = tokensInFile
            self.partialCountsForFile = partialCountsForFile
        }

        private enum CodingKeys: String, CodingKey {
            case path, lineOffset, tokensInFile, countsKeys, countsValues
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.path = try c.decode(String.self, forKey: .path)
            self.lineOffset = try c.decode(Int.self, forKey: .lineOffset)
            self.tokensInFile = try c.decode(Int.self, forKey: .tokensInFile)
            let ks = try c.decode([Int].self, forKey: .countsKeys)
            let vs = try c.decode([Int].self, forKey: .countsValues)
            var map: [Int: Int] = [:]
            for (k, v) in zip(ks, vs) { map[k] = v }
            self.partialCountsForFile = map
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(path, forKey: .path)
            try c.encode(lineOffset, forKey: .lineOffset)
            try c.encode(tokensInFile, forKey: .tokensInFile)
            let pairs = partialCountsForFile.sorted { $0.key < $1.key }
            try c.encode(pairs.map { $0.key }, forKey: .countsKeys)
            try c.encode(pairs.map { $0.value }, forKey: .countsValues)
        }
    }

    public struct RewriterState: Codable, Sendable {
        /// Nomi dei file shard già scritti correttamente in output.
        /// Salta-listing per il resume della Fase 2.
        public var completedShards: [String]

        public init(completedShards: [String] = []) {
            self.completedShards = completedShards
        }
    }

    // MARK: - File I/O

    /// Nome del file di checkpoint. Vive dentro la sottocartella
    /// `checkpoint/` della destinazione (vedi `fileURL(in:)`).
    public static let filename = "vocab_pruner.json"

    /// Nome della sottocartella che contiene il checkpoint (e in
    /// futuro eventuali file ausiliari del resume).
    public static let subdir = "checkpoint"

    /// Path legacy (pre-introduzione della sottocartella
    /// `checkpoint/`): file hidden direttamente in `outputDir`.
    /// Mantenuto solo per la migration al load — niente di nuovo
    /// viene scritto qui.
    public static let legacyFilename = ".vocab_pruner_checkpoint.json"

    /// URL del checkpoint corrente: `<outputDir>/checkpoint/vocab_pruner.json`.
    public static func fileURL(in outputDir: URL) -> URL {
        outputDir
            .appendingPathComponent(subdir, isDirectory: true)
            .appendingPathComponent(filename)
    }

    /// URL del checkpoint legacy (pre-migration).
    public static func legacyFileURL(in outputDir: URL) -> URL {
        outputDir.appendingPathComponent(legacyFilename)
    }

    /// Carica un checkpoint dal disco. Restituisce nil se il file
    /// non esiste o se il parse fallisce (corruzione → ripartiamo
    /// da zero, è più sicuro che provare a ripristinare).
    ///
    /// Migration: se il checkpoint non esiste al nuovo path
    /// (`checkpoint/vocab_pruner.json`) ma esiste al legacy path
    /// (`.vocab_pruner_checkpoint.json`), lo MIGRA atomicamente al
    /// nuovo path prima di ritornare. Questo significa che la
    /// prima volta che apri lo sheet su una destinazione con un
    /// checkpoint precedente, vedi il resume info come prima.
    public static func load(from outputDir: URL) -> PruneCheckpoint? {
        let url = fileURL(in: outputDir)
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            // Tenta migration dal legacy path.
            let legacy = legacyFileURL(in: outputDir)
            if fm.fileExists(atPath: legacy.path) {
                do {
                    try fm.createDirectory(
                        at: url.deletingLastPathComponent(),
                        withIntermediateDirectories: true)
                    try fm.moveItem(at: legacy, to: url)
                } catch {
                    // Migration fallita → leggi dal legacy senza
                    // spostare. Il prossimo save scriverà nel nuovo
                    // path e lascerà orfano il legacy (lo lasciamo
                    // all'utente per evitare data loss).
                    guard let data = try? Data(contentsOf: legacy) else {
                        return nil
                    }
                    let decoder = JSONDecoder()
                    decoder.dateDecodingStrategy = .iso8601
                    return try? decoder.decode(PruneCheckpoint.self, from: data)
                }
            }
        }
        guard fm.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(PruneCheckpoint.self, from: data)
    }

    /// Persiste atomicamente. Crea `outputDir/checkpoint/` se non
    /// esiste.
    public func save(to outputDir: URL) throws {
        let url = Self.fileURL(in: outputDir)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// Cancella il checkpoint. Rimuove sia il file nuovo che il
    /// legacy se presente, e prova a rimuovere la sottocartella
    /// `checkpoint/` se diventa vuota (per non lasciare directory
    /// orfana nella destinazione).
    public static func delete(from outputDir: URL) {
        let fm = FileManager.default
        try? fm.removeItem(at: fileURL(in: outputDir))
        try? fm.removeItem(at: legacyFileURL(in: outputDir))
        let subdirURL = outputDir.appendingPathComponent(subdir,
                                                          isDirectory: true)
        if let contents = try? fm.contentsOfDirectory(atPath: subdirURL.path),
           contents.isEmpty
        {
            try? fm.removeItem(at: subdirURL)
        }
    }

    // MARK: - Hash dello spec

    /// SHA-256 dei campi "che invalidano il checkpoint quando
    /// cambiano": inputDir, corpus, coverage. Output dir è dove
    /// sta il checkpoint stesso, quindi escluso. Hash troncato ai
    /// primi 16 hex char per leggibilità.
    public static func computeSpecHash(inputDir: URL,
                                        corpus: URL?,
                                        coverage: Double) -> String {
        let key = "\(inputDir.standardizedFileURL.path)|" +
                  "\(corpus?.standardizedFileURL.path ?? "<none>")|" +
                  String(format: "%.6f", coverage)
        let digest = SHA256.hash(data: Data(key.utf8))
        let hex = digest.compactMap { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }
}
