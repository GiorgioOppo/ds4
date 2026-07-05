import Foundation

// Configuration model for MCP (Model Context Protocol) servers. A server is an
// external tool provider the app connects to as an MCP CLIENT: at connect time
// the server declares its tools (`tools/list`) and the app exposes them to the
// model next to the built-ins; calls are forwarded with `tools/call`.
//
// Two transports are supported, mirroring the MCP spec:
//  - stdio: the app SPAWNS the server as a child process and speaks
//    newline-delimited JSON-RPC over its stdin/stdout. Note that in a sandboxed
//    (App Store) build the child inherits the app sandbox, so servers that need
//    the network or arbitrary file access may fail there; dev builds
//    (`swift run`, `make app`) run unsandboxed.
//  - http: Streamable-HTTP — JSON-RPC POSTed to the server's URL (remote MCP
//    servers; also the way to reach a local server the sandbox cannot spawn).

/// How to reach an MCP server.
public enum MCPTransportConfig: Sendable, Equatable, Codable {
    /// Spawn `command args...` and speak JSON-RPC over stdin/stdout.
    /// `environment` is merged over the inherited environment.
    case stdio(command: String, arguments: [String], environment: [String: String])
    /// POST JSON-RPC to `url` (Streamable HTTP). `headers` are added verbatim
    /// (e.g. an Authorization bearer token).
    case http(url: String, headers: [String: String])

    private enum CodingKeys: String, CodingKey {
        case kind, command, arguments, environment, url, headers
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .kind) {
        case "stdio":
            self = .stdio(command: try c.decode(String.self, forKey: .command),
                          arguments: try c.decodeIfPresent([String].self, forKey: .arguments) ?? [],
                          environment: try c.decodeIfPresent([String: String].self, forKey: .environment) ?? [:])
        case "http":
            self = .http(url: try c.decode(String.self, forKey: .url),
                         headers: try c.decodeIfPresent([String: String].self, forKey: .headers) ?? [:])
        default:
            throw DecodingError.dataCorruptedError(forKey: .kind, in: c,
                                                   debugDescription: "unknown transport kind")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .stdio(let command, let arguments, let environment):
            try c.encode("stdio", forKey: .kind)
            try c.encode(command, forKey: .command)
            try c.encode(arguments, forKey: .arguments)
            try c.encode(environment, forKey: .environment)
        case .http(let url, let headers):
            try c.encode("http", forKey: .kind)
            try c.encode(url, forKey: .url)
            try c.encode(headers, forKey: .headers)
        }
    }

    /// One-line human summary for lists ("npx -y @modelcontextprotocol/…" / URL).
    public var summary: String {
        switch self {
        case .stdio(let command, let arguments, _):
            return ([command] + arguments).joined(separator: " ")
        case .http(let url, _):
            return url
        }
    }
}

/// A configured MCP server. `id` is a stable sanitized key derived from the
/// name; it prefixes the exposed tool names (`mcp_<id>_<tool>`), so it must be
/// DSML-safe: lowercase alphanumerics and underscores only.
public struct MCPServerConfig: Sendable, Equatable, Codable, Identifiable {
    public var id: String
    public var name: String
    public var transport: MCPTransportConfig
    public var enabled: Bool

    public init(name: String, transport: MCPTransportConfig, enabled: Bool = true) {
        self.name = name
        self.id = MCPServerConfig.sanitizeID(name)
        self.transport = transport
        self.enabled = enabled
    }

    /// Sanitize a display name into a DSML-safe tool-name prefix: lowercase,
    /// alphanumerics kept, every other run of characters collapsed to "_".
    public static func sanitizeID(_ name: String) -> String {
        var out = ""
        var lastWasSep = true          // also strips a leading separator
        for ch in name.lowercased() {
            if ch.isLetter || ch.isNumber {
                out.append(ch); lastWasSep = false
            } else if !lastWasSep {
                out.append("_"); lastWasSep = true
            }
        }
        while out.hasSuffix("_") { out.removeLast() }
        return out.isEmpty ? "server" : out
    }
}

// MARK: - `mcpServers` JSON interchange (Claude Desktop / Cursor / VS Code format)

extension MCPServerConfig {
    /// Parse the de-facto standard config JSON:
    /// `{"mcpServers": {"<name>": {"command": "npx", "args": […], "env": {…}}
    ///                 | {"url": "https://…", "headers": {…}}}}`.
    /// A bare `{"<name>": {…}}` object without the wrapper is accepted too.
    /// Unrecognized entries are skipped; throws only when nothing parses.
    public static func importJSON(_ json: String) throws -> [MCPServerConfig] {
        enum ImportError: Error, CustomStringConvertible {
            case notAnObject, noServers
            var description: String {
                switch self {
                case .notAnObject: return "not a JSON object"
                case .noServers: return "no valid server entries found (expected \"mcpServers\": {…})"
                }
            }
        }
        guard let data = json.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ImportError.notAnObject
        }
        let servers = (root["mcpServers"] as? [String: Any]) ?? root
        var out: [MCPServerConfig] = []
        for (name, value) in servers.sorted(by: { $0.key < $1.key }) {
            guard let entry = value as? [String: Any] else { continue }
            if let command = entry["command"] as? String, !command.isEmpty {
                let args = (entry["args"] as? [Any])?.compactMap { $0 as? String } ?? []
                let env = (entry["env"] as? [String: Any])?
                    .compactMapValues { $0 as? String } ?? [:]
                out.append(MCPServerConfig(name: name,
                                           transport: .stdio(command: command, arguments: args,
                                                             environment: env)))
            } else if let url = entry["url"] as? String, !url.isEmpty {
                let headers = (entry["headers"] as? [String: Any])?
                    .compactMapValues { $0 as? String } ?? [:]
                out.append(MCPServerConfig(name: name, transport: .http(url: url, headers: headers)))
            }
        }
        guard !out.isEmpty else { throw ImportError.noServers }
        return out
    }

    /// Render configs back to the `mcpServers` interchange format.
    public static func exportJSON(_ configs: [MCPServerConfig]) -> String {
        var servers: [String: Any] = [:]
        for c in configs {
            switch c.transport {
            case .stdio(let command, let arguments, let environment):
                var e: [String: Any] = ["command": command, "args": arguments]
                if !environment.isEmpty { e["env"] = environment }
                servers[c.name] = e
            case .http(let url, let headers):
                var e: [String: Any] = ["url": url]
                if !headers.isEmpty { e["headers"] = headers }
                servers[c.name] = e
            }
        }
        let root: [String: Any] = ["mcpServers": servers]
        guard let data = try? JSONSerialization.data(withJSONObject: root,
                                                     options: [.prettyPrinted, .sortedKeys]),
              let s = String(data: data, encoding: .utf8) else { return "{\"mcpServers\":{}}" }
        return s
    }
}
