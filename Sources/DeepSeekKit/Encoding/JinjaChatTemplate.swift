import Foundation

/// `ChatTemplate` implementation backed by `JinjaTemplate`. Maps the
/// host `Message`/`ToolCall` structs into the Jinja scope HuggingFace
/// templates expect (`messages`, `bos_token`, `eos_token`,
/// `add_generation_prompt`, `tools`).
public struct JinjaChatTemplate: ChatTemplate {
    private let template: JinjaTemplate
    /// Raw source kept for diagnostics.
    public let source: String

    public init(_ source: String) throws {
        self.source = source
        self.template = try JinjaTemplate(source)
    }

    public func render(messages: [Message], options: ChatTemplateOptions) throws -> String {
        var scope: [String: JinjaValue] = [
            "bos_token":            .string(options.bosToken),
            "eos_token":            .string(options.eosToken),
            "add_generation_prompt": .bool(options.addGenerationPrompt),
        ]
        scope["messages"] = .list(messages.map(messageToJinja))
        if let tools = options.tools {
            scope["tools"] = .list(tools.map(jsonToJinja))
        }
        return try template.render(context: scope)
    }

    private func messageToJinja(_ m: Message) -> JinjaValue {
        var dict: [String: JinjaValue] = [
            "role":    .string(m.role.rawValue),
            "content": .string(m.content),
        ]
        if let r = m.reasoningContent {
            dict["reasoning_content"] = .string(r)
        }
        if !m.toolCalls.isEmpty {
            dict["tool_calls"] = .list(m.toolCalls.map { tc in
                .dict([
                    "id":   .string(tc.id ?? ""),
                    "type": .string("function"),
                    "function": .dict([
                        "name":      .string(tc.name),
                        "arguments": .string(tc.args),
                    ]),
                ])
            })
        }
        if !m.toolOutputs.isEmpty {
            dict["tool_outputs"] = .list(m.toolOutputs.map { .string($0) })
        }
        return .dict(dict)
    }

    /// Converte un `JSONValue` ricorsivo in `JinjaValue`. Pattern
    /// uno-a-uno fra i case dei due tipi.
    private func jsonToJinja(_ v: JSONValue) -> JinjaValue {
        switch v {
        case .null:           return .null
        case .bool(let b):    return .bool(b)
        case .int(let i):     return .int(i)
        case .double(let d):  return .double(d)
        case .string(let s):  return .string(s)
        case .array(let a):   return .list(a.map(jsonToJinja))
        case .object(let o):
            var out: [String: JinjaValue] = [:]
            for (k, val) in o { out[k] = jsonToJinja(val) }
            return .dict(out)
        }
    }
}
