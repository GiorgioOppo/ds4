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

    func modelsJSON() async -> String {
        let model = await modelJSON(modelId)
        return "{\"object\":\"list\",\"data\":[" + model + "]}"
    }

    func modelJSON(_ id: String) async -> String {
        // Advertise the loaded backend's actual context, not the server's
        // per-response token default or the GGUF's theoretical maximum.
        // Consumers such as Terminal-Bench Mini use this for their agent budget.
        let contextSize = await backend.modelInfo().contextSize
        let context = contextSize > 0 ? ",\"context_length\":\(contextSize)" : ""
        return "{\"id\":\(jsonString(id)),\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"dwarfstar\",\"name\":\(jsonString(modelName)),\"max_completion_tokens\":\(config.maxTokens)\(context)}"
    }

    /// Quote + escape an arbitrary string as a JSON string literal.
    func jsonString(_ s: String) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: [s]),
              let str = String(data: d, encoding: .utf8) else { return "\"\"" }
        return String(str.dropFirst().dropLast())   // strip the surrounding [ ]
    }
}
