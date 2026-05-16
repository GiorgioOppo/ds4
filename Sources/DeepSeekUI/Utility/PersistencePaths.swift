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

    /// Root for the projects library. One JSON index next to it lists
    /// every project; the per-project token files reuse
    /// `documentTokensURL(id:)` since project documents live in the
    /// same `VectorizedDocument` table, tagged with `projectID`.
    static func projectsDir() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
            .appendingPathComponent("projects", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func projectsIndexURL() throws -> URL {
        try projectsDir().appendingPathComponent("index.json")
    }

    /// Cached snapshot of OpenRouter's `/models` response. Lives
    /// alongside `models.json` so the "Add OpenRouter model"
    /// picker has instant autocompletion at app launch instead of
    /// waiting on a network round-trip. Refreshed by the catalog
    /// loader after 24 h, or on demand via the picker's reload
    /// button.
    static func openRouterCatalogURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("openrouter-catalog.json")
    }

    /// Persisted list of model endpoints the user has loaded —
    /// powers the in-chat model picker's "Recent" submenu. One
    /// JSON file under Application Support, decoded into
    /// `[ConfiguredModelEntry]` at app launch.
    static func modelsConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("models.json")
    }

    /// User-defined agent registry. Each agent is a preset of
    /// (system prompt + tool subset + sampling defaults + thinking
    /// mode). One JSON file under Application Support, decoded into
    /// `[AgentConfig]` at app launch.
    static func agentsConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("agents.json")
    }

    /// User-defined MCP (Model Context Protocol) server registry.
    /// Layout matches the simple-list shape we use for documents and
    /// projects: one JSON file under Application Support, decoded
    /// into `[MCPServerConfig]` at app launch. The actual JSON-RPC
    /// clients are spawned lazily by `MCPClientPool` (Step M2).
    static func mcpConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("mcp.json")
    }
}
