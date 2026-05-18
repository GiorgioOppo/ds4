import Foundation
import CryptoKit

/// Stato persistente del job di vocab pruning, salvato in
/// `<outputDir>/.vocab_pruner_checkpoint.json` per permettere il
/// ripristino dopo un'interruzione (kill, crash, cancel utente).
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

        public init(processedFiles: [String] = [],
                    partialCounts: [Int: Int] = [:],
                    linesScanned: Int = 0,
                    tokensScanned: Int = 0,
                    inFlightFile: InFlightFile? = nil) {
            self.processedFiles = processedFiles
            self.partialCounts = partialCounts
            self.linesScanned = linesScanned
            self.tokensScanned = tokensScanned
            self.inFlightFile = inFlightFile
        }

        private enum CodingKeys: String, CodingKey {
            case processedFiles, countsKeys, countsValues,
                 linesScanned, tokensScanned, inFlightFile
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

    public static let filename = ".vocab_pruner_checkpoint.json"

    public static func fileURL(in outputDir: URL) -> URL {
        outputDir.appendingPathComponent(filename)
    }

    /// Carica un checkpoint dal disco. Restituisce nil se il file
    /// non esiste o se il parse fallisce (corruzione → ripartiamo
    /// da zero, è più sicuro che provare a ripristinare).
    public static func load(from outputDir: URL) -> PruneCheckpoint? {
        let url = fileURL(in: outputDir)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(PruneCheckpoint.self, from: data)
    }

    /// Persiste atomicamente. Crea `outputDir` se non esiste.
    public func save(to outputDir: URL) throws {
        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: Self.fileURL(in: outputDir), options: .atomic)
    }

    public static func delete(from outputDir: URL) {
        try? FileManager.default.removeItem(at: fileURL(in: outputDir))
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
