import Foundation
import DS4Core
import DS4Engine

/// Maps an OpenAI `/v1/chat/completions` JSON body into the engine's types.
/// Tolerant: unknown fields are ignored, missing fields take sane defaults.
enum ChatRequestParser {
    struct Parsed {
        var turns: [ChatTurn]
        var tools: [ToolSpec]
        var sampling: SamplingParams
        var think: DS4ThinkMode
        var maxTokens: Int
        var stream: Bool
        var model: String?
    }

    static func parse(_ obj: [String: Any], defaultTokens: Int) -> Parsed {
        let turns = parseMessages(obj["messages"] as? [[String: Any]] ?? [])
        let tools = parseTools(obj["tools"] as? [[String: Any]] ?? [])

        var sampling = SamplingParams()
        if let t = number(obj["temperature"]) { sampling.temperature = Float(t) }
        if let p = number(obj["top_p"]) { sampling.topP = Float(p) }
        if let mp = number(obj["min_p"]) { sampling.minP = Float(mp) }
        if let k = obj["top_k"] as? Int { sampling.topK = k }
        if let s = obj["seed"] as? Int { sampling.seed = UInt64(bitPattern: Int64(s)) }

        // reasoning_effort: high/xhigh/medium enable thinking; low/minimal/none/absent disable.
        var think: DS4ThinkMode = .none
        if let effort = (obj["reasoning_effort"] as? String)?.lowercased(),
           ["high", "xhigh", "medium"].contains(effort) {
            think = .high
        }

        let maxTokens = (obj["max_tokens"] as? Int) ?? (obj["max_completion_tokens"] as? Int) ?? defaultTokens
        let stream = (obj["stream"] as? Bool) ?? false
        let model = obj["model"] as? String

        return Parsed(turns: turns, tools: tools, sampling: sampling, think: think,
                      maxTokens: max(1, maxTokens), stream: stream, model: model)
    }

    // MARK: Messages

    private static func parseMessages(_ msgs: [[String: Any]]) -> [ChatTurn] {
        var turns: [ChatTurn] = []
        for m in msgs {
            let role = m["role"] as? String ?? ""
            let content = contentString(m["content"])
            switch role {
            case "system", "developer":
                turns.append(.system(content))
            case "user":
                turns.append(.user(content))
            case "assistant":
                turns.append(.assistant(text: content, toolCalls: parseToolCalls(m["tool_calls"])))
            case "tool", "function":
                let id = m["tool_call_id"] as? String ?? ""
                let name = m["name"] as? String ?? ""
                turns.append(.toolResult(callId: id, name: name, content: content))
            default:
                if !content.isEmpty { turns.append(.user(content)) }
            }
        }
        return turns
    }

    /// `content` may be a string or an array of content parts ({type:"text",text:…}).
    private static func contentString(_ value: Any?) -> String {
        if let s = value as? String { return s }
        if let parts = value as? [[String: Any]] {
            var out = ""
            for p in parts {
                if (p["type"] as? String) == "text", let t = p["text"] as? String { out += t }
            }
            return out
        }
        return ""
    }

    private static func parseToolCalls(_ value: Any?) -> [ToolCall] {
        guard let arr = value as? [[String: Any]] else { return [] }
        var calls: [ToolCall] = []
        for (i, c) in arr.enumerated() {
            let fn = c["function"] as? [String: Any]
            let name = fn?["name"] as? String ?? ""
            let args = fn?["arguments"] as? String ?? "{}"
            let id = c["id"] as? String ?? "call_\(i)"
            if !name.isEmpty { calls.append(ToolCall(id: id, name: name, argumentsJSON: args)) }
        }
        return calls
    }

    // MARK: Tools

    private static func parseTools(_ tools: [[String: Any]]) -> [ToolSpec] {
        var specs: [ToolSpec] = []
        for t in tools {
            guard let fn = t["function"] as? [String: Any],
                  let name = fn["name"] as? String, !name.isEmpty else { continue }
            let desc = fn["description"] as? String ?? ""
            let paramsJSON: String
            if let params = fn["parameters"],
               let d = try? JSONSerialization.data(withJSONObject: params),
               let s = String(data: d, encoding: .utf8) {
                paramsJSON = s
            } else {
                paramsJSON = #"{"type":"object","properties":{}}"#
            }
            specs.append(ToolSpec(name: name, description: desc, parametersJSON: paramsJSON))
        }
        return specs
    }

    private static func number(_ value: Any?) -> Double? {
        if let d = value as? Double { return d }
        if let i = value as? Int { return Double(i) }
        if let n = value as? NSNumber { return n.doubleValue }
        return nil
    }
}
