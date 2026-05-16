import Foundation
import SwiftUI

/// One MCP tool surfaced by a connected server. `serverID` lets the
/// chat flow (Step M3) route a `tools/call` back to the right
/// client when the model picks this tool by name.
///
/// `qualifiedName` is "<server>__<tool>" so two servers can expose
/// tools with colliding short names without confusing the model —
/// the system-prompt block built from these is keyed by qualified
/// names and the chat flow strips the prefix before invoking.
struct MCPToolSchema: Hashable {
    let serverID: UUID
    let serverName: String
    let toolName: String
    let description: String
    /// Raw inputSchema as the server returned it. Already a valid
    /// JSON object (the wire decode validated that), persisted as
    /// `[String: Any]` so we can re-serialize as part of the model's
    /// tools system block without round-tripping through a typed
    /// Swift representation.
    let inputSchema: [String: Any]

    var qualifiedName: String { "\(serverName)__\(toolName)" }

    // Manual Hashable: the raw `[String: Any]` doesn't conform, so
    // we hash on the parts that uniquely identify the tool.
    static func == (lhs: MCPToolSchema, rhs: MCPToolSchema) -> Bool {
        lhs.serverID == rhs.serverID && lhs.toolName == rhs.toolName
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(serverID)
        hasher.combine(toolName)
    }
}

/// Connection state surfaced to the Settings UI. M2 only ever
/// transitions between these four — richer telemetry (last error
/// timestamp, tools-list snapshots over time) would live elsewhere.
enum MCPConnectionStatus: Equatable {
    case idle
    case connecting
    case connected(toolCount: Int)
    case error(String)
}

/// Errors surfaced through the public async API. The JSON-RPC
/// `error` payloads are flattened into `.rpc(code:message:)` so
/// callers don't need to know the wire shape.
enum MCPError: Error, LocalizedError {
    case spawnFailed(String)
    case timeout(method: String)
    case rpc(code: Int, message: String)
    case protocolError(String)
    case notConnected

    var errorDescription: String? {
        switch self {
        case .spawnFailed(let msg):          return "Failed to spawn server: \(msg)"
        case .timeout(let method):           return "MCP \(method) timed out"
        case .rpc(let code, let message):    return "MCP error \(code): \(message)"
        case .protocolError(let msg):        return "MCP protocol error: \(msg)"
        case .notConnected:                  return "MCP server is not connected"
        }
    }
}

// MARK: - Client

/// One live MCP server connection. Owns the child `Process`, the
/// stdin/stdout pipes, the request-id map, and the published
/// status. Lifecycle is `connect()` (idempotent on success) →
/// `callTool` zero or more times → `disconnect()`.
///
/// Concurrency: `MCPClient` is `@MainActor` for its `@Published`
/// surface, but the raw byte plumbing (readabilityHandler, stdin
/// writes) runs off the main actor. Cross-actor mutations go
/// through `MainActor.assumeIsolated`-style hops; the pending-
/// request map is guarded by an `NSLock` because it's accessed
/// from both sides.
@MainActor
final class MCPClient: ObservableObject {
    let config: MCPServerConfig

    @Published private(set) var status: MCPConnectionStatus = .idle
    @Published private(set) var tools: [MCPToolSchema] = []

    // Handles + Process need to be touched from both the main
    // actor (spawn / terminate) and the GCD readabilityHandler /
    // stdin write callbacks running on Foundation's internal
    // queues. Marked nonisolated(unsafe) and guarded by the
    // discipline that mutations only happen in `spawn` / `terminate`
    // (both MainActor-isolated) — the stdin write itself is
    // threadsafe inside Foundation's FileHandle.
    nonisolated(unsafe) private var process: Process?
    nonisolated(unsafe) private var stdinHandle: FileHandle?
    nonisolated(unsafe) private var stdoutHandle: FileHandle?
    nonisolated(unsafe) private var stderrHandle: FileHandle?

    /// Accumulator for inbound stdout bytes. The reader callback
    /// appends here on a background queue and slices off complete
    /// `\n`-terminated frames. NSLock-guarded; the lock itself is
    /// Sendable so it doesn't need a `nonisolated(unsafe)` tag.
    nonisolated(unsafe) private var inboundBuffer = Data()
    private let inboundLock = NSLock()

    /// Request id → continuation. The reader matches responses
    /// against this map. NSLock-guarded because reads happen on the
    /// stdout queue and writes happen on the main actor / call site.
    nonisolated(unsafe) private var pending:
        [Int: CheckedContinuation<[String: Any], Error>] = [:]
    private let pendingLock = NSLock()
    nonisolated(unsafe) private var nextRequestID: Int = 1

    /// JSON-RPC request timeout. `npx`/`uvx` first-run installs can
    /// genuinely take ~20 s while node modules download; the
    /// initialize handshake therefore allows a generous window.
    /// Subsequent `tools/call` rounds get a tighter budget so a
    /// stuck server doesn't freeze the chat indefinitely.
    private let handshakeTimeoutSeconds: Double = 30
    private let callTimeoutSeconds: Double = 60

    init(config: MCPServerConfig) {
        self.config = config
    }

    // MARK: - lifecycle

    /// Spawn the server, run the initialize handshake, then fetch
    /// the tools list. Idempotent in the sense that calling it on
    /// an already-connected client is a no-op; calling on a `.error`
    /// client tries again.
    func connect() async {
        if case .connected = status { return }
        status = .connecting
        do {
            try spawn()
            _ = try await sendRequest(
                method: "initialize",
                params: [
                    "protocolVersion": "2024-11-05",
                    "capabilities": [:],
                    "clientInfo": [
                        "name": "DeepSeek-V4-Pro-MacOS",
                        "version": "1.0"
                    ]
                ],
                timeout: handshakeTimeoutSeconds)
            // Notify the server we're done initializing. This is a
            // notification (no id) per JSON-RPC, so no response is
            // expected — fire-and-forget.
            try sendNotification(method: "notifications/initialized",
                                  params: [:])
            let toolsResp = try await sendRequest(
                method: "tools/list",
                params: [:],
                timeout: handshakeTimeoutSeconds)
            self.tools = Self.parseTools(toolsResp,
                                          serverID: config.id,
                                          serverName: config.name)
            self.status = .connected(toolCount: tools.count)
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            self.status = .error(msg)
            terminate()
        }
    }

    /// Send SIGTERM to the child and tear down the pipes. Any
    /// in-flight requests resume with `.notConnected`. Safe to call
    /// from any state (including never-connected).
    func disconnect() {
        terminate()
        status = .idle
        tools = []
    }

    // MARK: - tool invocation

    /// Invoke `tools/call` and flatten the response into a single
    /// string suitable for splicing into the chat as a tool output.
    /// The MCP `content` array can carry text, images, and resource
    /// references; for now we collect every `text` chunk and
    /// describe everything else with a placeholder so the model
    /// gets *something* readable even if the server returned media.
    func callTool(_ toolName: String, arguments: [String: Any]) async throws -> String {
        guard case .connected = status else { throw MCPError.notConnected }
        let resp = try await sendRequest(
            method: "tools/call",
            params: [
                "name": toolName,
                "arguments": arguments
            ],
            timeout: callTimeoutSeconds)
        return Self.flattenToolContent(resp)
    }

    // MARK: - JSON-RPC plumbing

    private func sendRequest(method: String,
                              params: [String: Any],
                              timeout: Double) async throws -> [String: Any] {
        let id = Self.nextID(in: self)
        var payload: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method
        ]
        if !params.isEmpty { payload["params"] = params }
        let data = try JSONSerialization.data(withJSONObject: payload)
        // Race the response against a timeout so a wedged server
        // surfaces as an error instead of hanging the UI forever.
        return try await withThrowingTaskGroup(of: [String: Any].self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation { cont in
                    self.pendingLock.withLock { self.pending[id] = cont }
                    do {
                        try self.writeFrame(data)
                    } catch {
                        let stillPending = self.pendingLock.withLock {
                            self.pending.removeValue(forKey: id)
                        }
                        stillPending?.resume(throwing: error)
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                let stillPending = self.pendingLock.withLock {
                    self.pending.removeValue(forKey: id)
                }
                stillPending?.resume(throwing: MCPError.timeout(method: method))
                throw MCPError.timeout(method: method)
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    private func sendNotification(method: String, params: [String: Any]) throws {
        var payload: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method
        ]
        if !params.isEmpty { payload["params"] = params }
        let data = try JSONSerialization.data(withJSONObject: payload)
        try writeFrame(data)
    }

    nonisolated private func writeFrame(_ data: Data) throws {
        guard let stdin = stdinHandle else { throw MCPError.notConnected }
        var bytes = data
        bytes.append(0x0A) // newline-terminated framing
        try stdin.write(contentsOf: bytes)
    }

    nonisolated private static func nextID(in client: MCPClient) -> Int {
        client.pendingLock.withLock {
            let id = client.nextRequestID
            client.nextRequestID += 1
            return id
        }
    }

    /// Process one complete JSON frame received on stdout. Routes
    /// responses (have `id` + `result`/`error`) back to the matching
    /// continuation; everything else (notifications, server→client
    /// requests) is logged and dropped for now.
    nonisolated private func handleFrame(_ data: Data) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            NSLog("[MCP %@] unparseable frame: %@",
                  config.name,
                  String(data: data, encoding: .utf8) ?? "<binary>")
            return
        }
        if let id = obj["id"] as? Int {
            let cont = pendingLock.withLock { pending.removeValue(forKey: id) }
            guard let cont else {
                // Server-to-client request — currently unsupported
                // (would need sampling/createMessage etc.). Drop.
                return
            }
            if let err = obj["error"] as? [String: Any] {
                let code = (err["code"] as? Int) ?? -32000
                let message = (err["message"] as? String) ?? "unknown error"
                cont.resume(throwing: MCPError.rpc(code: code, message: message))
            } else if let result = obj["result"] as? [String: Any] {
                cont.resume(returning: result)
            } else {
                cont.resume(throwing: MCPError.protocolError(
                    "response missing result and error"))
            }
        } else {
            // Notification — log only.
            if let method = obj["method"] as? String {
                NSLog("[MCP %@] notification: %@", config.name, method)
            }
        }
    }

    // MARK: - process

    private func spawn() throws {
        let process = Process()
        // Run through /usr/bin/env so PATH lookup finds commands
        // like `npx`/`uvx`/`python` that live in the user's shell
        // PATH (Homebrew, fnm, pyenv, …) instead of the Mac app
        // bundle's minimal default. We also prepend the common
        // bin dirs into PATH so /usr/bin/env can find them even
        // when launchd handed us a stripped environment.
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [config.command] + config.args

        var env = ProcessInfo.processInfo.environment
        let extraPath = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        env["PATH"] = (env["PATH"]?.appending(":\(extraPath)")) ?? extraPath
        for (k, v) in config.env { env[k] = v }
        process.environment = env

        let stdin  = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardInput  = stdin
        process.standardOutput = stdout
        process.standardError  = stderr

        // stdout reader: append every chunk to the inbound buffer
        // and slice off any complete newline-terminated frames.
        // Runs on a background queue; handleFrame is nonisolated
        // and shuffles responses back through pendingLock to the
        // awaiting tasks.
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let chunk = handle.availableData
            if chunk.isEmpty {
                self.onChildExited()
                return
            }
            // Drain every complete `\n`-terminated frame out of the
            // shared inbound buffer under the lock, then dispatch
            // them outside the critical section — `handleFrame`
            // takes pendingLock and holding both at once would
            // risk a lock-ordering bug.
            let frames = self.inboundLock.withLock { () -> [Data] in
                self.inboundBuffer.append(chunk)
                var out: [Data] = []
                while let nl = self.inboundBuffer.firstIndex(of: 0x0A) {
                    let frame = self.inboundBuffer.subdata(
                        in: self.inboundBuffer.startIndex..<nl)
                    self.inboundBuffer.removeSubrange(
                        self.inboundBuffer.startIndex...nl)
                    if !frame.isEmpty { out.append(frame) }
                }
                return out
            }
            for frame in frames {
                self.handleFrame(frame)
            }
        }

        // stderr reader: just log to stderr for now so users can
        // see what's wrong via Console.app or by running the host
        // app from a terminal. Stream is otherwise discarded.
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            let chunk = handle.availableData
            if chunk.isEmpty { return }
            if let s = String(data: chunk, encoding: .utf8) {
                NSLog("[MCP %@ stderr] %@", self.config.name, s)
            }
        }

        do {
            try process.run()
        } catch {
            throw MCPError.spawnFailed(error.localizedDescription)
        }

        self.process = process
        self.stdinHandle = stdin.fileHandleForWriting
        self.stdoutHandle = stdout.fileHandleForReading
        self.stderrHandle = stderr.fileHandleForReading
    }

    private func terminate() {
        // Detach pipe readers first so we don't try to handle the
        // EOF as a "child exited unexpectedly" event.
        stdoutHandle?.readabilityHandler = nil
        stderrHandle?.readabilityHandler = nil
        if let p = process, p.isRunning {
            p.terminate()
        }
        process = nil
        stdinHandle = nil
        stdoutHandle = nil
        stderrHandle = nil
        // Resume every pending continuation so callers don't hang
        // forever waiting for a response that will never arrive.
        let drained = pendingLock.withLock { () -> [Int: CheckedContinuation<[String: Any], Error>] in
            let snap = pending
            pending = [:]
            return snap
        }
        for (_, cont) in drained {
            cont.resume(throwing: MCPError.notConnected)
        }
    }

    nonisolated private func onChildExited() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            // If status was already .error, leave it (the original
            // cause is more informative than "child exited").
            if case .error = self.status { return }
            self.status = .error("Server process exited unexpectedly")
            self.terminate()
            self.tools = []
        }
    }

    // MARK: - response parsers

    private static func parseTools(_ resp: [String: Any],
                                    serverID: UUID,
                                    serverName: String) -> [MCPToolSchema] {
        guard let arr = resp["tools"] as? [[String: Any]] else { return [] }
        return arr.compactMap { t in
            guard let name = t["name"] as? String else { return nil }
            let desc = (t["description"] as? String) ?? ""
            let schema = (t["inputSchema"] as? [String: Any]) ?? [:]
            return MCPToolSchema(serverID: serverID,
                                  serverName: serverName,
                                  toolName: name,
                                  description: desc,
                                  inputSchema: schema)
        }
    }

    private static func flattenToolContent(_ resp: [String: Any]) -> String {
        guard let content = resp["content"] as? [[String: Any]] else {
            if let isError = resp["isError"] as? Bool, isError {
                return "[tool returned isError without content]"
            }
            return ""
        }
        var out: [String] = []
        for item in content {
            let type = (item["type"] as? String) ?? "unknown"
            switch type {
            case "text":
                if let s = item["text"] as? String { out.append(s) }
            case "image":
                let mime = (item["mimeType"] as? String) ?? "image/*"
                out.append("[image: \(mime)]")
            case "resource":
                let uri = ((item["resource"] as? [String: Any])?["uri"] as? String)
                    ?? "<unknown>"
                out.append("[resource: \(uri)]")
            default:
                out.append("[\(type)]")
            }
        }
        return out.joined(separator: "\n\n")
    }
}

// MARK: - Pool

/// Coordinator over every `MCPClient` derived from the user's
/// configured server registry. Owned by the App scene and
/// re-synced whenever the `MCPServerLibrary` mutates so a newly-
/// enabled server connects automatically and a disabled one
/// disconnects.
@MainActor
final class MCPClientPool: ObservableObject {
    @Published private(set) var clients: [UUID: MCPClient] = [:]

    /// Observe the library and keep our client set in lock-step.
    /// Called once from `DeepSeekUIApp` after both objects exist.
    func attach(to library: MCPServerLibrary) {
        sync(with: library.servers)
        // Re-sync on every change to the registry. NotificationCenter
        // would be overkill here — the library's @Published is
        // already broadcast to any SwiftUI subscriber, and the pool
        // gets that via the App scene's view re-render flow
        // (`reconcile` is also called from the picker / settings
        // edit paths via `librarySynced(_:)`).
    }

    /// Explicitly re-sync after the library mutates. Called by
    /// `MCPServersView` on every add/edit/delete/toggle, plus by
    /// the App scene on launch.
    func librarySynced(_ library: MCPServerLibrary) {
        sync(with: library.servers)
    }

    func client(forServer id: UUID) -> MCPClient? {
        clients[id]
    }

    /// Flatten every connected server's tool list into one array
    /// suitable for the chat-flow tool schemas block (Step M3).
    func allTools() -> [MCPToolSchema] {
        clients.values.flatMap { $0.tools }
    }

    /// Build the JSON-encoded tools array that the chat template's
    /// `EncodingDSV4.toolsBlock` expects in the system message.
    /// Tool names are emitted in qualified form ("<server>__<tool>")
    /// so the model picks an unambiguous identifier; on execution
    /// (M3b) we'll split that back into (serverID, toolName).
    ///
    /// Returns nil when no enabled server has reported any tool —
    /// signals to the caller "don't bother injecting the tools
    /// section at all".
    func toolSchemasJSON() -> String? {
        let tools = allTools()
        if tools.isEmpty { return nil }
        var payload: [[String: Any]] = []
        for t in tools {
            payload.append([
                "name": t.qualifiedName,
                "description": t.description,
                "inputSchema": t.inputSchema
            ])
        }
        guard let data = try? JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys])
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func disconnectAll() {
        for client in clients.values { client.disconnect() }
        clients = [:]
    }

    /// Route a model-emitted tool call to the right MCPClient and
    /// return its flattened textual output. The name argument is
    /// the qualified form produced by `toolSchemasJSON` —
    /// "<server>__<tool>" — and the implementation strips the
    /// server prefix to find both the connection and the bare tool
    /// name to send over the wire.
    ///
    /// Args come in as the model's JSON string (taken from
    /// `ToolCall.args`); we parse it into a dictionary here so the
    /// caller doesn't have to. Empty / unparseable args become an
    /// empty `arguments: {}` payload.
    ///
    /// Errors are flattened to a human-readable string (no throws)
    /// so the chat loop can always emit *some* tool_output and
    /// keep going — letting the model self-correct on a bad call
    /// is more useful than aborting the whole turn.
    func invokeQualified(_ qualifiedName: String,
                          argsJSON: String) async -> String {
        guard let sep = qualifiedName.range(of: "__") else {
            return "[error: tool name not qualified as <server>__<tool>: \(qualifiedName)]"
        }
        let serverName = String(qualifiedName[..<sep.lowerBound])
        let toolName = String(qualifiedName[sep.upperBound...])
        guard let client = clients.values.first(where: {
            $0.config.name == serverName
        }) else {
            return "[error: no MCP server named \(serverName)]"
        }
        // Parse arguments. The DSML decoder produces a JSON object
        // string (matching what `encodeArguments` round-trips). If
        // parsing fails — model emitted a malformed payload — we
        // still call with an empty argument set so the server can
        // either succeed or return a structured error of its own.
        var argsDict: [String: Any] = [:]
        if !argsJSON.isEmpty,
           let data = argsJSON.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            argsDict = parsed
        }
        do {
            return try await client.callTool(toolName, arguments: argsDict)
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            return "[error: \(msg)]"
        }
    }

    // MARK: - private

    private func sync(with servers: [MCPServerConfig]) {
        let enabled = servers.filter(\.enabled)
        let wanted = Set(enabled.map(\.id))

        // Drop clients whose server was removed or disabled.
        for (id, client) in clients where !wanted.contains(id) {
            client.disconnect()
            clients[id] = nil
        }

        // Add clients for newly-enabled servers.
        for server in enabled where clients[server.id] == nil {
            let client = MCPClient(config: server)
            clients[server.id] = client
            Task { await client.connect() }
        }

        // For existing clients whose config changed (command/args/env),
        // bounce the connection so the new spawn picks up the changes.
        for server in enabled {
            guard let existing = clients[server.id],
                  existing.config != server else { continue }
            existing.disconnect()
            let replacement = MCPClient(config: server)
            clients[server.id] = replacement
            Task { await replacement.connect() }
        }
    }
}
