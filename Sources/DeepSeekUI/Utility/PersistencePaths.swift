import Foundation

/// Where the app writes its on-disk state. One JSON file per
/// conversation under
/// `~/Library/Application Support/DeepSeek-V4-Pro-MacOS/conversations/`.
enum PersistencePaths {
    static let appName = "DeepSeek-V4-Pro-MacOS"

    /// Creates the directory tree on first use; failures bubble up.
    static func conversationsDir() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
            .appendingPathComponent("conversations", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func conversationURL(id: UUID) throws -> URL {
        try conversationsDir()
            .appendingPathComponent("\(id.uuidString).json")
    }

    /// Path to the persistent KV cache file for a conversation. Lives
    /// next to the conversation's `.json` so deleting a conversation
    /// can wipe both atomically. Step 2 wires this into the layer
    /// allocators; Step 3 uses the header inside to enable cross-turn
    /// prefill reuse.
    static func kvCacheURL(id: UUID) throws -> URL {
        try conversationsDir()
            .appendingPathComponent("\(id.uuidString).kvcache")
    }

    /// Root for the global "vectorized documents" library — the
    /// per-document files (.tokens / .vec) plus an index.json that
    /// `DocumentLibrary` reads at app launch.
    static func documentsDir() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
            .appendingPathComponent("documents", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// Index file: a JSON array of `VectorizedDocument` entries.
    static func documentsIndexURL() throws -> URL {
        try documentsDir().appendingPathComponent("index.json")
    }

    /// Raw token-id payload for a vectorized document (int32 little-endian,
    /// one per token). Written by Step 1's import flow.
    static func documentTokensURL(id: UUID) throws -> URL {
        try documentsDir().appendingPathComponent("\(id.uuidString).tokens")
    }

    /// Pre-computed KV cache for a vectorized document. Created by
    /// Step 3's import pipeline; absent until then.
    static func documentVecURL(id: UUID) throws -> URL {
        try documentsDir().appendingPathComponent("\(id.uuidString).vec")
    }
}
