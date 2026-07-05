import Foundation
import DS4Core

// Process-wide registry of MCP servers (mirroring AgentRegistry's pattern):
// the app pushes the configured servers in, the manager owns the client
// connections, and every tool consumer — chat, agents, distributed mode —
// reads the same synchronous snapshot of specs/statuses. Connections and tool
// calls are async; snapshots are lock-protected so the UI and the tool loop
// can query them without awaiting. Change handlers fire on every state
// transition so consumers re-declare tools when a server (dis)connects
// instead of polling.
//
// Namespacing: a server tool `read_file` on server `fs` is exposed to the
// model as `mcp_fs_read_file` (name collisions get a numeric suffix). The
// mapping back to (server, tool) is kept in an explicit index — names are
// never parsed apart.

/// Connection state of one configured server.
public enum MCPServerState: Sendable, Equatable {
    case disabled                 // configured but switched off
    case connecting
    case connected
    case failed(String)
}

/// UI-facing snapshot of one server: config summary + live state + the tools
/// it exposes (as the namespaced specs the model actually sees).
public struct MCPServerStatus: Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var transportSummary: String
    public var enabled: Bool
    public var state: MCPServerState
    public var serverInfo: String       // "name version" from initialize
    public var tools: [ToolSpec]
}

public final class MCPManager: @unchecked Sendable {
    public static let shared = MCPManager()
    public static let toolNamePrefix = "mcp_"

    /// Everything the manager knows about one configured server. Kept in a
    /// single struct so every state transition is one entry update.
    private struct Entry {
        var config: MCPServerConfig
        var state: MCPServerState
        var serverInfo = ""
        var tools: [MCPToolInfo] = []
        var client: MCPClient?
        /// Namespaced specs exposed to the model (rebuilt with the index).
        var specs: [ToolSpec] = []
    }

    private let lock = NSLock()
    private var order: [String] = []            // config order (drives collision priority)
    private var entries: [String: Entry] = [:]
    /// Exposed spec name → (server id, original tool name).
    private var toolIndex: [String: (server: String, tool: String)] = [:]
    /// All connected servers' specs, in config order (cached; O(1) reads).
    private var cachedSpecs: [ToolSpec] = []
    /// Bumped on every state change; cheap for the UI to compare.
    private var stateVersion = 0
    /// Fired (off-lock) after every state change. Consumers hop to their own
    /// isolation (the app re-syncs declared tools, the panel refreshes).
    private var changeHandlers: [@Sendable () -> Void] = []

    // MARK: Configuration

    /// Register a change callback (called after every connect/disconnect/
    /// config change). There is no removal: handlers are registered once by
    /// long-lived owners and must capture weakly.
    public func addChangeHandler(_ handler: @escaping @Sendable () -> Void) {
        lock.lock(); changeHandlers.append(handler); lock.unlock()
    }

    /// Replace the configured server set. Servers whose config changed (or that
    /// disappeared) are disconnected; enabled servers (re)connect.
    public func setConfigs(_ newConfigs: [MCPServerConfig]) {
        var toDisconnect: [MCPClient] = []
        var toConnect: [MCPServerConfig] = []
        mutate {
            var newEntries: [String: Entry] = [:]
            var newOrder: [String] = []
            for config in newConfigs {
                guard newEntries[config.id] == nil else { continue }   // duplicate ids: first wins
                newOrder.append(config.id)
                if let old = entries[config.id], old.config.transport == config.transport,
                   old.config.enabled == config.enabled {
                    var kept = old
                    kept.config = config                // name may have changed
                    newEntries[config.id] = kept
                    continue
                }
                // New or changed server: drop any old connection and start over.
                if let client = entries[config.id]?.client { toDisconnect.append(client) }
                newEntries[config.id] = Entry(config: config,
                                              state: config.enabled ? .connecting : .disabled)
                if config.enabled { toConnect.append(config) }
            }
            for (id, entry) in entries where newEntries[id] == nil {
                if let client = entry.client { toDisconnect.append(client) }
            }
            entries = newEntries
            order = newOrder
        }
        for client in toDisconnect { Task { await client.shutdown() } }
        for config in toConnect { connect(config) }
    }

    /// Drop and re-open one server's connection (the UI's "Reconnect" button).
    public func reconnect(id: String) {
        var oldClient: MCPClient?
        var config: MCPServerConfig?
        mutate {
            guard var entry = entries[id] else { return }
            oldClient = entry.client
            config = entry.config
            entry.client = nil
            entry.tools = []
            entry.serverInfo = ""
            entry.state = entry.config.enabled ? .connecting : .disabled
            entries[id] = entry
        }
        if let oldClient { Task { await oldClient.shutdown() } }
        if let config, config.enabled { connect(config) }
    }

    private func connect(_ config: MCPServerConfig) {
        let id = config.id
        let client = MCPClient(config: config, onDisconnect: { [weak self] client, reason in
            self?.markDisconnected(id: id, client: client, reason: reason)
        })
        mutate {
            guard var entry = entries[id] else { return }
            entry.state = .connecting
            entry.client = client
            entries[id] = entry
        }
        Task { [weak self] in
            do {
                let (serverInfo, serverTools) = try await client.connect()
                self?.update(id: id, ifClient: client) { entry in
                    entry.state = .connected
                    entry.serverInfo = serverInfo
                    entry.tools = serverTools
                }
            } catch {
                // Attach the stdio stderr tail when there is one: "npx: command
                // not found" beats a bare "server process exited".
                var message = "\(error)"
                let tail = await client.stderrTail.trimmingCharacters(in: .whitespacesAndNewlines)
                if !tail.isEmpty, !message.contains(tail.suffix(40)) {
                    message += " — stderr: \(String(tail.suffix(300)))"
                }
                await client.shutdown()
                self?.update(id: id, ifClient: client) { entry in
                    entry.client = nil
                    entry.state = .failed(message)
                    entry.tools = []
                }
            }
        }
    }

    /// A transport died on its own. The identity check matters: a STALE
    /// client's exit (e.g. the old process dying right after a Reconnect)
    /// must not tear down the replacement connection.
    private func markDisconnected(id: String, client: MCPClient, reason: String) {
        update(id: id, ifClient: client) { entry in
            entry.client = nil
            entry.state = .failed(reason.isEmpty ? "disconnected" : reason)
            entry.tools = []
        }
    }

    /// Apply `change` to the entry — but only if it still belongs to `client`
    /// (a reconnect may have superseded it; pass nil to skip that check).
    private func update(id: String, ifClient client: MCPClient?,
                        _ change: (inout Entry) -> Void) {
        mutate {
            guard var entry = entries[id] else { return }
            if let client, entry.client !== client { return }
            change(&entry)
            entries[id] = entry
        }
    }

    /// Run a state mutation under the lock, rebuild the tool index/spec cache,
    /// bump the version, then fire the change handlers OFF the lock.
    private func mutate(_ body: () -> Void) {
        lock.lock()
        body()
        rebuildLocked()
        stateVersion += 1
        let handlers = changeHandlers
        lock.unlock()
        for h in handlers { h() }
    }

    /// One pass over the connected servers, in config order: assign each tool
    /// its exposed name (numeric suffix on collision), fill the reverse index,
    /// and cache the per-server + global spec arrays.
    private func rebuildLocked() {
        toolIndex.removeAll()
        cachedSpecs.removeAll()
        for id in order {
            guard var entry = entries[id] else { continue }
            entry.specs = []
            if entry.state == .connected {
                for tool in entry.tools {
                    var name = Self.specName(server: id, tool: tool.name)
                    var n = 2
                    while toolIndex[name] != nil { name = Self.specName(server: id, tool: tool.name) + "_\(n)"; n += 1 }
                    toolIndex[name] = (id, tool.name)
                    let desc = tool.description.isEmpty ? "MCP tool from server \(entry.config.name)."
                                                        : String(tool.description.prefix(500))
                    let spec = ToolSpec(name: name, description: "[\(entry.config.name)] \(desc)",
                                        parametersJSON: tool.inputSchemaJSON)
                    entry.specs.append(spec)
                    cachedSpecs.append(spec)
                }
            }
            entries[id] = entry
        }
    }

    /// `mcp_<server>_<tool>` with the tool name reduced to the same DSML-safe
    /// charset as server ids (before collision suffixing).
    public static func specName(server: String, tool: String) -> String {
        "\(toolNamePrefix)\(server)_\(MCPServerConfig.sanitizeID(tool))"
    }

    // MARK: Snapshots (synchronous, for UI + tool declaration)

    /// Monotonic change counter — compare it to know when to re-read statuses.
    public var version: Int { lock.lock(); defer { lock.unlock() }; return stateVersion }

    public func statuses() -> [MCPServerStatus] {
        lock.lock(); defer { lock.unlock() }
        return order.compactMap { id in
            guard let e = entries[id] else { return nil }
            return MCPServerStatus(id: id, name: e.config.name,
                                   transportSummary: e.config.transport.summary,
                                   enabled: e.config.enabled, state: e.state,
                                   serverInfo: e.serverInfo, tools: e.specs)
        }
    }

    /// All tools of all CONNECTED servers as namespaced ToolSpecs, ready to sit
    /// next to the built-ins in pickers and agent tool lists.
    public func toolSpecs() -> [ToolSpec] {
        lock.lock(); defer { lock.unlock() }
        return cachedSpecs
    }

    /// The namespaced subset in `names` (same contract as ToolRegistry.specs).
    public func specs(enabled names: Set<String>) -> [ToolSpec] {
        lock.lock(); defer { lock.unlock() }
        return cachedSpecs.filter { names.contains($0.name) }
    }

    /// True if `name` belongs to a KNOWN (connected) MCP tool.
    public func isMCPTool(named name: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return toolIndex[name] != nil
    }

    // MARK: Execution

    /// Synchronous locked lookup: NSLock is not usable directly inside an async
    /// function (the compiler forbids it — a suspension while holding the lock
    /// would deadlock), so `execute` resolves the target through this helper.
    private func target(forToolNamed name: String)
        -> (client: MCPClient?, serverName: String, tool: String)? {
        lock.lock(); defer { lock.unlock() }
        guard let target = toolIndex[name] else { return nil }
        return (entries[target.server]?.client,
                entries[target.server]?.config.name ?? target.server,
                target.tool)
    }

    /// Execute a model-emitted call if it targets an MCP tool; nil if the name
    /// is not an MCP tool (callers then fall back to manual entry). Transport /
    /// server errors come back as an error ToolOutput so the model can react.
    public func execute(_ call: ToolCall) async -> ToolOutput? {
        guard let target = target(forToolNamed: call.name) else { return nil }
        let client = target.client
        let serverName = target.serverName

        guard let client else {
            let msg = WebClient.jsonEscape("MCP server '\(serverName)' is not connected")
            return ToolOutput(callId: call.id, name: call.name, content: #"{"error":\#(msg)}"#)
        }
        do {
            let text = try await client.callTool(name: target.tool, argumentsJSON: call.argumentsJSON)
            return ToolOutput(callId: call.id, name: call.name, content: text)
        } catch {
            let msg = WebClient.jsonEscape("MCP call failed (\(serverName)/\(target.tool)): \(error)")
            return ToolOutput(callId: call.id, name: call.name, content: #"{"error":\#(msg)}"#)
        }
    }
}
