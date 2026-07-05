import Foundation
import DS4Core

// Pure JSON-RPC 2.0 / MCP message helpers, kept free of transport and actor
// state so they are directly unit-testable: build request/notification frames,
// classify incoming frames, and translate MCP results (tools/list, tools/call)
// into the engine's ToolSpec / plain-text tool output.

/// The MCP protocol revision this client speaks. Servers negotiating an older
/// revision still work: the subset used here (initialize, tools/list,
/// tools/call, ping) is stable across revisions.
public enum MCP {
    public static let protocolVersion = "2025-06-18"
    public static let clientName = "DwarfStar"
    public static let clientVersion = "1.0"
}

/// A tool as declared by an MCP server (before namespacing).
public struct MCPToolInfo: Sendable, Equatable, Identifiable {
    public var name: String
    public var description: String
    public var inputSchemaJSON: String   // JSON Schema object, verbatim
    public var id: String { name }
    public init(name: String, description: String, inputSchemaJSON: String) {
        self.name = name; self.description = description; self.inputSchemaJSON = inputSchemaJSON
    }
}

enum MCPProtocol {
    // MARK: Outgoing frames

    /// `{"jsonrpc":"2.0","id":N,"method":…,"params":{…}}` as one line of data.
    /// `params` must be JSON-encodable (String/number/dict/array trees only).
    static func request(id: Int, method: String, params: [String: Any]?) -> Data {
        var obj: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
        if let params { obj["params"] = params }
        return encode(obj)
    }

    static func notification(method: String, params: [String: Any]? = nil) -> Data {
        var obj: [String: Any] = ["jsonrpc": "2.0", "method": method]
        if let params { obj["params"] = params }
        return encode(obj)
    }

    /// Empty-result response, used to answer server→client `ping` requests.
    static func emptyResponse(id: Any) -> Data {
        encode(["jsonrpc": "2.0", "id": id, "result": [String: Any]()])
    }

    /// "Method not found" error response for any other server→client request.
    static func methodNotFound(id: Any, method: String) -> Data {
        encode(["jsonrpc": "2.0", "id": id,
                "error": ["code": -32601, "message": "method not supported by this client: \(method)"]])
    }

    static func initializeParams() -> [String: Any] {
        ["protocolVersion": MCP.protocolVersion,
         "capabilities": [String: Any](),
         "clientInfo": ["name": MCP.clientName, "version": MCP.clientVersion]]
    }

    private static func encode(_ obj: [String: Any]) -> Data {
        (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
    }

    // MARK: Incoming frames

    /// What one parsed incoming JSON-RPC frame means to the client.
    enum Incoming {
        /// A response to one of our requests: id + result payload (or error text).
        case response(id: Int, result: [String: Any]?, errorMessage: String?)
        /// A server→client REQUEST that needs an answer (ping, sampling, roots…).
        case serverRequest(id: Any, method: String)
        /// A notification (logging, list_changed…) — safe to ignore.
        case notification(method: String)
        /// Anything unparseable.
        case invalid
    }

    static func classify(_ data: Data) -> Incoming {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .invalid
        }
        let method = obj["method"] as? String
        let id = obj["id"]
        if let method {
            guard let id else { return .notification(method: method) }
            return .serverRequest(id: id, method: method)
        }
        // A response: id may arrive as a number or (some servers) a numeric string.
        let numericID: Int? = (id as? NSNumber)?.intValue ?? (id as? String).flatMap(Int.init)
        guard let numericID else { return .invalid }
        if let err = obj["error"] as? [String: Any] {
            let msg = (err["message"] as? String) ?? "unknown error"
            let code = (err["code"] as? NSNumber)?.intValue
            return .response(id: numericID, result: nil,
                             errorMessage: code.map { "\(msg) (code \($0))" } ?? msg)
        }
        return .response(id: numericID, result: obj["result"] as? [String: Any] ?? [:],
                         errorMessage: nil)
    }

    // MARK: Result translation

    /// Parse a `tools/list` result page: the tools plus the pagination cursor.
    static func parseToolsList(_ result: [String: Any]) -> (tools: [MCPToolInfo], nextCursor: String?) {
        var out: [MCPToolInfo] = []
        for entry in result["tools"] as? [[String: Any]] ?? [] {
            guard let name = entry["name"] as? String, !name.isEmpty else { continue }
            let desc = (entry["description"] as? String) ?? ""
            var schema = #"{"type":"object","properties":{}}"#
            if let s = entry["inputSchema"] as? [String: Any],
               let d = try? JSONSerialization.data(withJSONObject: s, options: [.sortedKeys]),
               let str = String(data: d, encoding: .utf8) {
                schema = str
            }
            out.append(MCPToolInfo(name: name, description: desc, inputSchemaJSON: schema))
        }
        return (out, result["nextCursor"] as? String)
    }

    /// Flatten a `tools/call` result into the plain text fed back to the model:
    /// text content concatenated; non-text content noted; `structuredContent`
    /// serialized when there is no text; `isError` surfaced as a prefix so the
    /// model knows the call failed.
    static func flattenCallResult(_ result: [String: Any]) -> String {
        var parts: [String] = []
        for item in result["content"] as? [[String: Any]] ?? [] {
            switch item["type"] as? String {
            case "text":
                if let t = item["text"] as? String { parts.append(t) }
            case "image":
                parts.append("[image content: \((item["mimeType"] as? String) ?? "unknown type")]")
            case "audio":
                parts.append("[audio content: \((item["mimeType"] as? String) ?? "unknown type")]")
            case "resource":
                let res = item["resource"] as? [String: Any]
                if let t = res?["text"] as? String { parts.append(t) }
                else { parts.append("[resource: \((res?["uri"] as? String) ?? "?")]") }
            case "resource_link":
                parts.append("[resource link: \((item["uri"] as? String) ?? "?")]")
            default:
                break
            }
        }
        if parts.isEmpty, let structured = result["structuredContent"],
           JSONSerialization.isValidJSONObject(structured),
           let d = try? JSONSerialization.data(withJSONObject: structured, options: [.sortedKeys]),
           let s = String(data: d, encoding: .utf8) {
            parts.append(s)
        }
        var text = parts.joined(separator: "\n")
        if text.isEmpty { text = "(empty tool result)" }
        if (result["isError"] as? Bool) == true { text = "Tool error: " + text }
        return text
    }
}
