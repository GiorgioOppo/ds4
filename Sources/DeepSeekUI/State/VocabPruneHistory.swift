import Foundation

/// Record persistente di un job di vocab pruning completato.
/// Codable per essere serializzato in JSON nella directory di
/// Application Support.
struct VocabPruneRecord: Codable, Identifiable, Equatable {
    var id: UUID
    let timestamp: Date
    let inputDir: String
    let outputDir: String
    let corpus: String?
    let coverage: Double
    let oldVocabSize: Int
    let newVocabSize: Int
    let bytesIn: UInt64
    let bytesOut: UInt64
    let dryRun: Bool

    init(id: UUID = UUID(),
         timestamp: Date = Date(),
         inputDir: String,
         outputDir: String,
         corpus: String?,
         coverage: Double,
         oldVocabSize: Int,
         newVocabSize: Int,
         bytesIn: UInt64,
         bytesOut: UInt64,
         dryRun: Bool) {
        self.id = id
        self.timestamp = timestamp
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.corpus = corpus
        self.coverage = coverage
        self.oldVocabSize = oldVocabSize
        self.newVocabSize = newVocabSize
        self.bytesIn = bytesIn
        self.bytesOut = bytesOut
        self.dryRun = dryRun
    }
}

/// Singleton-like history dei job di vocab pruning. Persiste in
/// `~/Library/Application Support/DeepSeek-V4/vocab_pruner_history.json`.
/// @MainActor perché letto/scritto dalla UI; il file I/O e' fatto
/// dal main thread (file è piccolo, ms al massimo).
@MainActor
final class VocabPruneHistory: ObservableObject {
    @Published private(set) var records: [VocabPruneRecord] = []

    private static let maxRecords = 50

    init() {
        load()
    }

    func add(_ record: VocabPruneRecord) {
        records.insert(record, at: 0)
        if records.count > Self.maxRecords {
            records = Array(records.prefix(Self.maxRecords))
        }
        save()
    }

    func remove(_ id: UUID) {
        records.removeAll { $0.id == id }
        save()
    }

    func clear() {
        records.removeAll()
        save()
    }

    // MARK: - persistence

    private static func fileURL() -> URL? {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
        else { return nil }
        let dir = appSupport.appendingPathComponent("DeepSeek-V4")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("vocab_pruner_history.json")
    }

    private func load() {
        guard let url = Self.fileURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let arr = try? decoder.decode([VocabPruneRecord].self, from: data) {
            self.records = arr
        }
    }

    private func save() {
        guard let url = Self.fileURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(records) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
