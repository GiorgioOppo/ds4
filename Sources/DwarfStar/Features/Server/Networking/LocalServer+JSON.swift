import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: JSON helpers

    /// Emit one Anthropic SSE event: `event: <name>\ndata: <json>\n\n`.
    func sse(_ conn: NWConnection, _ event: String, _ data: String) async throws {
        try await send(conn, Data("event: \(event)\ndata: \(data)\n\n".utf8))
    }

    func toolCallsJSON(_ calls: [ToolCall]) -> String {
        var parts: [String] = []
        for (i, c) in calls.enumerated() {
            parts.append("{\"index\":\(i),\"id\":\(jsonString(c.id)),\"type\":\"function\",\"function\":{\"name\":\(jsonString(c.name)),\"arguments\":\(jsonString(c.argumentsJSON))}}")
        }
        return "[" + parts.joined(separator: ",") + "]"
    }

    func modelsJSON() -> String {
        "{\"object\":\"list\",\"data\":[" + modelJSON(modelId) + "]}"
    }

    func modelJSON(_ id: String) -> String {
        "{\"id\":\(jsonString(id)),\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"dwarfstar\",\"name\":\(jsonString(modelName)),\"max_completion_tokens\":\(config.maxTokens)}"
    }

    /// Quote + escape an arbitrary string as a JSON string literal.
    func jsonString(_ s: String) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: [s]),
              let str = String(data: d, encoding: .utf8) else { return "\"\"" }
        return String(str.dropFirst().dropLast())   // strip the surrounding [ ]
    }
}

