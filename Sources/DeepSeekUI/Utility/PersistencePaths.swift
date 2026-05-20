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

    // MARK: - v2 lazy-loading layout
    //
    // New chats live in a per-chat folder so deletion is a single
    // `removeItem(at:)` and the streaming hot path can write to a
    // small `pending.json` instead of re-encoding the full
    // transcript. Legacy chats keep `kvCacheURL(id:)` /
    // `conversationURL(id:)` above unchanged.

    /// Folder root for one v2 chat. Created on demand; throws only on
    /// FS errors. The caller is responsible for not racing against
    /// `delete(id:)` — same contract as `conversationURL`.
    static func chatDir(id: UUID) throws -> URL {
        let dir = try conversationsDir()
            .appendingPathComponent(id.uuidString, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// `chat.json` — the `ChatManifest` for a v2 chat. Tiny (~kB)
    /// regardless of conversation length, read at app launch to
    /// paint the sidebar.
    static func chatManifestURL(id: UUID) throws -> URL {
        try chatDir(id: id).appendingPathComponent("chat.json")
    }

    /// `chat.kvcache` — moved inside the chat folder so a folder
    /// wipe takes the KV cache with it.
    static func chatKVCacheURL(id: UUID) throws -> URL {
        try chatDir(id: id).appendingPathComponent("chat.kvcache")
    }

    /// `chat.tokens` — binary `[Int32]` payload of the canonical
    /// tokenised prompt prefix (v1 `Conversation.encodedTokens`).
    /// Header: 4-byte magic ('DSTK') + 4-byte little-endian count.
    /// Loaded only on `send()` fast path; absent for remote-only
    /// chats.
    static func chatTokensURL(id: UUID) throws -> URL {
        try chatDir(id: id).appendingPathComponent("chat.tokens")
    }

    /// `pending.json` — `PendingSnapshot` written hot during
    /// streaming so a crash mid-turn loses at most ~200 ms of
    /// sampling. Cleared at `.done`.
    static func chatPendingURL(id: UUID) throws -> URL {
        try chatDir(id: id).appendingPathComponent("pending.json")
    }

    /// `turns/` — folder holding one `{turnID}.json` per turn
    /// summary, and one `{turnID}/rounds/` subfolder per turn for
    /// the heavy per-round payloads.
    static func chatTurnsDir(chatID: UUID) throws -> URL {
        let dir = try chatDir(id: chatID)
            .appendingPathComponent("turns", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// `turns/{turnID}.json` — the medium-weight `TurnSummary` file.
    static func turnSummaryURL(chatID: UUID, turnID: UUID) throws -> URL {
        try chatTurnsDir(chatID: chatID)
            .appendingPathComponent("\(turnID.uuidString).json")
    }

    /// `turns/{turnID}/rounds/` — folder holding one
    /// `{roundID}.json` per assistant generate pass within this
    /// turn.
    static func roundsDir(chatID: UUID, turnID: UUID) throws -> URL {
        let dir = try chatDir(id: chatID)
            .appendingPathComponent("turns", isDirectory: true)
            .appendingPathComponent(turnID.uuidString, isDirectory: true)
            .appendingPathComponent("rounds", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// `turns/{turnID}/rounds/{roundID}.json` — the full
    /// per-round payload. Loaded only on demand (disclosure open,
    /// truncated-preview fetch, or send-time history materialisation).
    static func roundURL(chatID: UUID, turnID: UUID, roundID: UUID) throws -> URL {
        try roundsDir(chatID: chatID, turnID: turnID)
            .appendingPathComponent("\(roundID.uuidString).json")
    }

    /// True when a per-chat folder exists at the v2 layout location.
    /// Used by `ChatStore.loadFromDisk()` to decide whether a chat
    /// id should be read through the v2 manifest pipeline or the
    /// legacy single-file pipeline.
    static func isV2Chat(id: UUID) -> Bool {
        guard let dir = try? chatDir(id: id) else { return false }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: dir.path, isDirectory: &isDir) else { return false }
        return isDir.boolValue
            && FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("chat.json").path)
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

    /// Per-project symlink-farm root. `ProjectRootBuilder` materializes
    /// each `Project.sourcePaths` entry here as a tree of real
    /// directories with file symlinks, so tools running with
    /// `ToolContext.rootDirectory` set to this URL see the project
    /// layout without us copying any bytes.
    static func projectRootDir(id: UUID) throws -> URL {
        let dir = try projectsDir()
            .appendingPathComponent(id.uuidString, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
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

    /// Persistent permission defaults for native tools: a JSON map
    /// `<tool>:<category> → ask|alwaysAllow|alwaysDeny`. Loaded by
    /// `PermissionStore` at app launch.
    static func permissionsConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("permissions.json")
    }

    /// User-defined skills (system prompt addenda + tool subsets).
    static func skillsConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("skills.json")
    }

    /// User overrides for theme / keybindings. Two separate files so
    /// the user can wipe one without losing the other.
    static func themeConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("themes.json")
    }

    static func keybindingsConfigURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir.appendingPathComponent("keybindings.json")
    }

    /// Directory for custom slash command JSON files (one per command).
    static func slashCommandsDir() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
            .appendingPathComponent("commands", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }
}
