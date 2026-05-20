import Foundation
import CryptoKit

/// Persistent state of an expert-prune job, saved to
/// `<outputDir>/checkpoint/expert_pruner.json`. Sibling of
/// `vocab_pruner.json` — the two phases own separate files so a
/// pipeline run that survives only the vocab phase still drops
/// the (vocab-) checkpoint at the right time and the expert
/// phase starts with a clean slate. Same patterns and conventions
/// as `PruneCheckpoint`.
public struct ExpertPruneCheckpoint: Codable, Sendable {

    public enum Phase: String, Codable, Sendable {
        case analyzer    // still walking the calibration corpus
        case rewriter    // analyzer done, writing the new shards
        case done        // everything complete (file is deleted at the end)
    }

    public var phase: Phase
    public var specHash: String
    public var savedAt: Date

    /// Phase 1 state. nil after the analyzer commits its decision.
    public var analyzer: AnalyzerState?

    /// Consolidated decision saved as soon as Phase 1 finishes.
    /// nil while still in Phase 1.
    public var decision: ExpertKeepDecision?

    /// Phase 2 state. nil while still in Phase 1.
    public var rewriter: RewriterState?

    public init(phase: Phase,
                specHash: String,
                analyzer: AnalyzerState? = nil,
                decision: ExpertKeepDecision? = nil,
                rewriter: RewriterState? = nil,
                savedAt: Date = Date())
    {
        self.phase = phase
        self.specHash = specHash
        self.analyzer = analyzer
        self.decision = decision
        self.rewriter = rewriter
        self.savedAt = savedAt
    }

    public struct AnalyzerState: Codable, Sendable {
        /// Files fully processed so far. Used by the analyzer to skip
        /// re-tokenizing + re-running forward on these on resume.
        public var processedFiles: [String]
        /// Total tokens observed during calibration so far. Used
        /// for the `--max-calibration-tokens` cap and for progress
        /// reporting.
        public var tokensProcessed: Int
        /// Partial usage grid accumulated so far. Serialized as
        /// flat `[ExpertUsageRow]` because Codable's default
        /// dict-with-Int-keys handling stringifies the keys.
        public var partialUsage: [ExpertUsageRow]

        public init(processedFiles: [String] = [],
                    tokensProcessed: Int = 0,
                    partialUsage: [ExpertUsageRow] = [])
        {
            self.processedFiles = processedFiles
            self.tokensProcessed = tokensProcessed
            self.partialUsage = partialUsage
        }
    }

    public struct RewriterState: Codable, Sendable {
        /// Shard filenames already written successfully on a prior
        /// run. Skipped on resume (existing output kept, no rewrite).
        public var completedShards: [String]

        public init(completedShards: [String] = []) {
            self.completedShards = completedShards
        }
    }

    // MARK: - File I/O

    public static let filename = "expert_pruner.json"
    public static let subdir = "checkpoint"

    public static func fileURL(in outputDir: URL) -> URL {
        outputDir
            .appendingPathComponent(subdir, isDirectory: true)
            .appendingPathComponent(filename)
    }

    public static func load(from outputDir: URL) -> ExpertPruneCheckpoint? {
        let url = fileURL(in: outputDir)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ExpertPruneCheckpoint.self, from: data)
    }

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

    public static func delete(from outputDir: URL) {
        let fm = FileManager.default
        try? fm.removeItem(at: fileURL(in: outputDir))
        let subdirURL = outputDir
            .appendingPathComponent(subdir, isDirectory: true)
        if let contents = try? fm.contentsOfDirectory(atPath: subdirURL.path),
           contents.isEmpty
        {
            try? fm.removeItem(at: subdirURL)
        }
    }

    // MARK: - Spec hash

    /// SHA-256 of fields that invalidate the checkpoint when they
    /// change: inputDir, calibCorpus, coverage, minKeptFloor.
    /// outputDir is where the checkpoint lives, so excluded.
    /// expertStatsFile is also excluded — if the user re-runs with
    /// a precomputed stats file, they explicitly skip Phase 1, so
    /// the analyzer checkpoint isn't relevant.
    public static func computeSpecHash(inputDir: URL,
                                        calibCorpus: URL?,
                                        coverage: Double,
                                        minKeptFloor: Int) -> String
    {
        let key = "\(inputDir.standardizedFileURL.path)|" +
                  "\(calibCorpus?.standardizedFileURL.path ?? "<none>")|" +
                  String(format: "%.6f", coverage) + "|" +
                  String(minKeptFloor)
        let digest = SHA256.hash(data: Data(key.utf8))
        let hex = digest.compactMap { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }
}
