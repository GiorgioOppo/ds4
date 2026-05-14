import Foundation
import SwiftUI

/// One entry in the global vectorized-documents library.
///
/// Step 1 only materialises the `.tokens` payload (raw int32 token ids
/// produced by the model's BPE tokenizer). Step 3 will additionally
/// produce a `.vec` file holding the KV cache obtained by running a
/// prefill forward over those tokens; until then `hasPrecomputedCache`
/// stays false and the chat-side loader (Step 2/3) falls back to
/// re-tokenising the prompt as a system message.
struct VectorizedDocument: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String              // user-visible label, defaults to filename
    var sourceFilename: String    // original on-disk file name
    var byteCount: Int            // size of the source text in bytes
    var tokenCount: Int           // length of the int32 token sequence
    var createdAt: Date
    /// Fingerprint of the model the tokens were produced against. The
    /// BPE vocab is part of the checkpoint, so a token sequence is only
    /// safe to reuse with the same model. Step 3 widens this to also
    /// cover the KV cache, which depends on every weight tensor.
    var modelFingerprint: String
    /// True once Step 3 has dumped the KV cache into the `.vec` file.
    var hasPrecomputedCache: Bool

    init(id: UUID = UUID(),
         name: String,
         sourceFilename: String,
         byteCount: Int,
         tokenCount: Int,
         createdAt: Date = .now,
         modelFingerprint: String,
         hasPrecomputedCache: Bool = false) {
        self.id = id
        self.name = name
        self.sourceFilename = sourceFilename
        self.byteCount = byteCount
        self.tokenCount = tokenCount
        self.createdAt = createdAt
        self.modelFingerprint = modelFingerprint
        self.hasPrecomputedCache = hasPrecomputedCache
    }
}

/// Global library of vectorized documents. Lives for the app's
/// lifetime; persisted as an `index.json` + one binary file per
/// document in `Application Support/.../documents/`. Conversations
/// reference entries here by `id` (added in Step 2).
@MainActor
final class DocumentLibrary: ObservableObject {
    @Published private(set) var documents: [VectorizedDocument] = []

    init() {
        load()
    }

    /// Append a freshly-imported document and persist the tokens to
    /// disk. Returns the assigned id so callers can immediately attach
    /// it to a conversation.
    func add(name: String,
              sourceFilename: String,
              byteCount: Int,
              tokens: [Int32],
              modelFingerprint: String) throws -> VectorizedDocument {
        let doc = VectorizedDocument(
            name: name,
            sourceFilename: sourceFilename,
            byteCount: byteCount,
            tokenCount: tokens.count,
            modelFingerprint: modelFingerprint)
        let url = try PersistencePaths.documentTokensURL(id: doc.id)
        try tokens.withUnsafeBufferPointer { ptr in
            let data = Data(buffer: ptr)
            try data.write(to: url, options: .atomic)
        }
        documents.insert(doc, at: 0)
        saveIndex()
        return doc
    }

    /// Remove a document and wipe its on-disk payloads. Best-effort:
    /// missing files don't surface as errors.
    func delete(_ id: UUID) {
        guard let idx = documents.firstIndex(where: { $0.id == id }) else { return }
        documents.remove(at: idx)
        let fm = FileManager.default
        if let u = try? PersistencePaths.documentTokensURL(id: id) {
            try? fm.removeItem(at: u)
        }
        if let u = try? PersistencePaths.documentVecURL(id: id) {
            try? fm.removeItem(at: u)
        }
        saveIndex()
    }

    /// Reload tokens for a document. Used by Step 2's chat loader.
    func tokens(of id: UUID) throws -> [Int32] {
        let url = try PersistencePaths.documentTokensURL(id: id)
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { raw in
            let buf = raw.bindMemory(to: Int32.self)
            return Array(buf)
        }
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.documentsIndexURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let entries = try? decoder.decode([VectorizedDocument].self, from: data) {
            documents = entries.sorted { $0.createdAt > $1.createdAt }
        }
    }

    private func saveIndex() {
        guard let url = try? PersistencePaths.documentsIndexURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(documents) {
            try? data.write(to: url, options: .atomic)
        }
    }
}

/// Lightweight fingerprint for a model checkpoint. The BPE vocab and
/// weights are tied to the on-disk directory; hashing its absolute
/// path is enough to detect "different model selected" between
/// vectorize-time and chat-time. Step 3 swaps this for a content hash
/// (digest of `config.json`) once we start persisting KV bytes too.
enum ModelFingerprint {
    static func of(modelDirPath: String) -> String {
        // FNV-1a 64-bit on UTF-8 bytes. Stable across processes; no
        // dependency on CryptoKit (not all SDKs surface it cleanly in
        // a CLI build).
        var h: UInt64 = 0xcbf29ce484222325
        for byte in modelDirPath.utf8 {
            h ^= UInt64(byte)
            h = h &* 0x100000001b3
        }
        return String(format: "%016llx", h)
    }
}
