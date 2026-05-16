import Foundation
import SwiftUI

/// One MCP (Model Context Protocol) server the user has registered.
/// Each entry describes how to *spawn* the server — the actual
/// child process + JSON-RPC connection is owned by `MCPClientPool`
/// (Step M2) and is keyed by `id`.
///
/// Field layout mirrors the Claude Desktop config format so an
/// import-from-clipboard / import-from-file flow can ingest existing
/// configurations without translation. Example:
///
///     {
///       "name": "filesystem",
///       "command": "npx",
///       "args": ["@modelcontextprotocol/server-filesystem",
///                 "/Users/me/Documents"],
///       "env": { "DEBUG": "1" }
///     }
///
/// `enabled` is a quick toggle: a disabled server keeps the config
/// row but the pool won't spawn it, won't expose its tools to the
/// model, and won't surface in the chat-side picker (Step M3).
struct MCPServerConfig: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String              // human-readable, also JSON-RPC client id
    var command: String           // binary in PATH, or absolute path
    var args: [String]            // argv after `command`
    var env: [String: String]     // extra env vars (API keys, …)
    var enabled: Bool
    var createdAt: Date

    init(id: UUID = UUID(),
         name: String,
         command: String,
         args: [String] = [],
         env: [String: String] = [:],
         enabled: Bool = true,
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.enabled = enabled
        self.createdAt = createdAt
    }
}

/// Persisted registry of MCP servers. ObservableObject so the
/// Settings UI re-renders on add/remove/toggle, but the actual
/// runtime client state (connection status, tool list) lives on
/// `MCPClientPool` and is read out separately (Step M2).
@MainActor
final class MCPServerLibrary: ObservableObject {
    @Published private(set) var servers: [MCPServerConfig] = []

    init() {
        load()
    }

    func add(_ server: MCPServerConfig) {
        servers.append(server)
        save()
    }

    func update(_ server: MCPServerConfig) {
        guard let idx = servers.firstIndex(where: { $0.id == server.id }) else { return }
        servers[idx] = server
        save()
    }

    func delete(_ id: UUID) {
        servers.removeAll { $0.id == id }
        save()
    }

    func setEnabled(_ id: UUID, _ enabled: Bool) {
        guard let idx = servers.firstIndex(where: { $0.id == id }) else { return }
        guard servers[idx].enabled != enabled else { return }
        servers[idx].enabled = enabled
        save()
    }

    /// One-shot import from a Claude Desktop-shaped config blob.
    /// Accepted shapes:
    ///   { "mcpServers": { "name": { command, args, env } } }   (Claude Desktop)
    ///   { "name":         { command, args, env } }              (older variant)
    ///
    /// Duplicate names are skipped silently — the user can rename and
    /// retry. Returns the number of servers actually imported.
    @discardableResult
    func importClaudeDesktopJSON(_ data: Data) throws -> Int {
        guard let any = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return 0 }
        let dict: [String: Any]
        if let nested = any["mcpServers"] as? [String: Any] {
            dict = nested
        } else {
            dict = any
        }
        var imported = 0
        let existingNames = Set(servers.map(\.name))
        for (name, raw) in dict {
            guard let obj = raw as? [String: Any] else { continue }
            if existingNames.contains(name) { continue }
            let command = (obj["command"] as? String) ?? ""
            guard !command.isEmpty else { continue }
            let args = (obj["args"] as? [String]) ?? []
            let env = (obj["env"] as? [String: String]) ?? [:]
            servers.append(MCPServerConfig(
                name: name, command: command, args: args, env: env))
            imported += 1
        }
        if imported > 0 { save() }
        return imported
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.mcpConfigURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let entries = try? decoder.decode([MCPServerConfig].self, from: data) {
            servers = entries.sorted { $0.createdAt < $1.createdAt }
        }
    }

    private func save() {
        guard let url = try? PersistencePaths.mcpConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(servers) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
