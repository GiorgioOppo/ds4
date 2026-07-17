import Foundation

public struct GLM52ToolParseResult: Sendable, Equatable {
    public let calls: [ToolCall]
    public let visibleText: String
    public let rawToolText: String?

    public init(calls: [ToolCall], visibleText: String, rawToolText: String?) {
        self.calls = calls
        self.visibleText = visibleText
        self.rawToolText = rawToolText
    }
}

public enum GLM52ToolCodecError: Error, Sendable, Equatable, CustomStringConvertible {
    case incomplete(String)
    case malformed(String)
    case invalidIdentifier(String)
    case toolCallInsideThinking

    public var description: String {
        switch self {
        case .incomplete(let expected): return "incomplete GLM tool call; expected \(expected)"
        case .malformed(let reason): return "malformed GLM tool call: \(reason)"
        case .invalidIdentifier(let value): return "invalid GLM tool identifier: \(value)"
        case .toolCallInsideThinking: return "GLM tool calls are not allowed inside <think>"
        }
    }
}

/// Renderer and strict parser for GLM's native flat XML tool grammar.
///
/// The grammar intentionally differs from DeepSeek DSML:
/// `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`.
public enum GLM52ToolCodec {
    private static let identifierCharacters = CharacterSet.alphanumerics
        .union(CharacterSet(charactersIn: "_-."))

    // MARK: Tool declarations

    /// Native system prompt used by GLM when tools are available.
    public static func toolsPrompt(_ tools: [ToolSpec]) throws -> String {
        guard !tools.isEmpty else { return "" }
        let schemas = try tools.map(functionSchemaJSON).joined(separator: "\n")
        // Keep this wording and whitespace aligned with the GGUF
        // `tokenizer.chat_template`; agent-specific guidance belongs in a
        // separate system message, not in the model-native protocol prefix.
        return """

        You may call one or more functions to assist with the user query.
        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        \(schemas)
        </tools>
        For each function call, output the function name and arguments within the following XML format:
        <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>
        """
    }

    /// The function object consumed by GLM's chat template, encoded
    /// deterministically for stable prompt/KV-cache prefixes.  The OpenAI
    /// `{"type":"function","function": ...}` transport wrapper is deliberately
    /// omitted because the model template unwraps it before rendering `<tools>`.
    public static func functionSchemaJSON(_ tool: ToolSpec) throws -> String {
        try requireIdentifier(tool.name)
        let parameters: Any
        if let data = tool.parametersJSON.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data),
           object is [String: Any] {
            parameters = neutralizeJSONStrings(object)
        } else {
            parameters = ["type": "object", "properties": [:]] as [String: Any]
        }
        let function: [String: Any] = [
            "description": GLM52ConversationProtocol.neutralizeControlTokens(in: tool.description),
            "name": tool.name,
            "parameters": parameters,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: function,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: Rendering

    /// Render structured historical calls using deterministic parameter order.
    /// A schema contributes its sorted property order; unknown keys follow in
    /// sorted order. Invalid JSON is preserved under an `arguments` key.
    public static func renderToolCalls(_ calls: [ToolCall], tools: [ToolSpec] = []) throws -> String {
        var orderByTool: [String: [String]] = [:]
        for tool in tools {
            orderByTool[tool.name] = parameterTypes(for: tool).keys.sorted()
        }
        var blocks: [String] = []
        blocks.reserveCapacity(calls.count)

        for call in calls {
            try requireIdentifier(call.name)
            var block = GLM52ConversationProtocol.toolCallOpen + call.name
            if let data = call.argumentsJSON.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let preferred = orderByTool[call.name] ?? []
                let order = preferred.filter { object[$0] != nil }
                    + object.keys.filter { !preferred.contains($0) }.sorted()
                for key in order {
                    try requireIdentifier(key)
                    block += GLM52ConversationProtocol.argumentKeyOpen
                    block += escapeTagBody(key, closingTag: GLM52ConversationProtocol.argumentKeyClose)
                    block += GLM52ConversationProtocol.argumentKeyClose
                    block += GLM52ConversationProtocol.argumentValueOpen
                    block += escapeTagBody(jsonFragment(object[key]!),
                                           closingTag: GLM52ConversationProtocol.argumentValueClose)
                    block += GLM52ConversationProtocol.argumentValueClose
                }
            } else {
                block += GLM52ConversationProtocol.argumentKeyOpen + "arguments"
                block += GLM52ConversationProtocol.argumentKeyClose
                block += GLM52ConversationProtocol.argumentValueOpen
                block += escapeTagBody(call.argumentsJSON,
                                       closingTag: GLM52ConversationProtocol.argumentValueClose)
                block += GLM52ConversationProtocol.argumentValueClose
            }
            block += GLM52ConversationProtocol.toolCallClose
            blocks.append(block)
        }
        return blocks.joined(separator: "\n\n")
    }

    public static func escapeToolResponse(_ content: String) -> String {
        escapeTagBody(content, closingTag: GLM52ConversationProtocol.toolResponseClose)
    }

    private static func escapeTagBody(_ text: String, closingTag: String) -> String {
        let escapedClose = "&lt;" + closingTag.dropFirst()
        let protected = text.replacingOccurrences(of: closingTag, with: escapedClose)
        return GLM52ConversationProtocol.neutralizeControlTokens(in: protected)
    }

    private static func jsonFragment(_ value: Any) -> String {
        if let string = value as? String { return string }
        guard JSONSerialization.isValidJSONObject([value]),
              let data = try? JSONSerialization.data(
                withJSONObject: value,
                options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]
              ) else { return "null" }
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: Strict parsing

    /// Parse zero or more complete calls. Once a call starts, all remaining
    /// non-whitespace text must be native GLM call syntax; malformed or partial
    /// blocks never escape as executable calls.
    public static func parseStrict(_ text: String, tools: [ToolSpec] = []) throws
        -> GLM52ToolParseResult {
        let open = GLM52ConversationProtocol.toolCallOpen
        guard let first = text.range(of: open)?.lowerBound else {
            return GLM52ToolParseResult(calls: [], visibleText: text, rawToolText: nil)
        }
        if toolCallIsInsideThinking(String(text[..<first])) {
            throw GLM52ToolCodecError.toolCallInsideThinking
        }

        var schemas: [String: ParameterSchema] = [:]
        for tool in tools { schemas[tool.name] = parameterSchema(for: tool) }
        var cursor = first
        var calls: [ToolCall] = []

        while cursor < text.endIndex {
            skipWhitespace(in: text, cursor: &cursor)
            guard cursor < text.endIndex else { break }
            guard text[cursor...].hasPrefix(open) else {
                throw GLM52ToolCodecError.malformed("non-whitespace text after a tool call")
            }
            cursor = text.index(cursor, offsetBy: open.count)

            guard let structural = text[cursor...].firstIndex(of: "<") else {
                throw GLM52ToolCodecError.incomplete("<arg_key> or </tool_call>")
            }
            let name = String(text[cursor..<structural])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            try requireIdentifier(name)
            if !tools.isEmpty, schemas[name] == nil {
                throw GLM52ToolCodecError.malformed("undeclared tool \(name)")
            }
            cursor = structural

            var arguments: [String: Any] = [:]
            while true {
                skipWhitespace(in: text, cursor: &cursor)
                if consume(GLM52ConversationProtocol.toolCallClose, in: text, cursor: &cursor) {
                    break
                }
                if isPartialPrefix(at: cursor,
                                   of: GLM52ConversationProtocol.toolCallClose,
                                   in: text) {
                    throw GLM52ToolCodecError.incomplete(GLM52ConversationProtocol.toolCallClose)
                }
                guard consume(GLM52ConversationProtocol.argumentKeyOpen,
                              in: text, cursor: &cursor) else {
                    if isPartialPrefix(at: cursor,
                                       of: GLM52ConversationProtocol.argumentKeyOpen,
                                       in: text) {
                        throw GLM52ToolCodecError.incomplete(GLM52ConversationProtocol.argumentKeyOpen)
                    }
                    throw GLM52ToolCodecError.malformed("expected <arg_key> or </tool_call>")
                }

                guard let keyEnd = text.range(
                    of: GLM52ConversationProtocol.argumentKeyClose,
                    range: cursor..<text.endIndex
                ) else {
                    if text.range(of: GLM52ConversationProtocol.toolCallClose,
                                  range: cursor..<text.endIndex) != nil {
                        throw GLM52ToolCodecError.malformed("unterminated <arg_key>")
                    }
                    throw GLM52ToolCodecError.incomplete(GLM52ConversationProtocol.argumentKeyClose)
                }
                if let callEnd = text.range(
                    of: GLM52ConversationProtocol.toolCallClose,
                    range: cursor..<text.endIndex
                ), callEnd.lowerBound < keyEnd.lowerBound {
                    throw GLM52ToolCodecError.malformed("unterminated <arg_key>")
                }
                // GLM's native template treats tag bodies as raw text; it does
                // not apply XML entity decoding.
                let key = String(text[cursor..<keyEnd.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                try requireIdentifier(key)
                if let schema = schemas[name], !schema.propertyNames.contains(key) {
                    throw GLM52ToolCodecError.malformed(
                        "undeclared argument \(key) for tool \(name)"
                    )
                }
                guard arguments[key] == nil else {
                    throw GLM52ToolCodecError.malformed("duplicate argument \(key)")
                }
                cursor = keyEnd.upperBound
                skipWhitespace(in: text, cursor: &cursor)

                guard consume(GLM52ConversationProtocol.argumentValueOpen,
                              in: text, cursor: &cursor) else {
                    if isPartialPrefix(at: cursor,
                                       of: GLM52ConversationProtocol.argumentValueOpen,
                                       in: text) {
                        throw GLM52ToolCodecError.incomplete(GLM52ConversationProtocol.argumentValueOpen)
                    }
                    throw GLM52ToolCodecError.malformed("expected <arg_value>")
                }
                guard let valueEnd = text.range(
                    of: GLM52ConversationProtocol.argumentValueClose,
                    range: cursor..<text.endIndex
                ) else {
                    if text.range(of: GLM52ConversationProtocol.toolCallClose,
                                  range: cursor..<text.endIndex) != nil {
                        throw GLM52ToolCodecError.malformed("unterminated <arg_value>")
                    }
                    throw GLM52ToolCodecError.incomplete(GLM52ConversationProtocol.argumentValueClose)
                }
                if let callEnd = text.range(
                    of: GLM52ConversationProtocol.toolCallClose,
                    range: cursor..<text.endIndex
                ), callEnd.lowerBound < valueEnd.lowerBound {
                    throw GLM52ToolCodecError.malformed("unterminated <arg_value>")
                }
                let rawValue = String(text[cursor..<valueEnd.lowerBound])
                arguments[key] = typedValue(rawValue, type: schemas[name]?.types[key])
                cursor = valueEnd.upperBound
            }

            if let schema = schemas[name] {
                let missing = schema.required.subtracting(arguments.keys).sorted()
                if !missing.isEmpty {
                    throw GLM52ToolCodecError.malformed(
                        "missing required argument(s) for tool \(name): \(missing.joined(separator: ", "))"
                    )
                }
            }

            let data = try JSONSerialization.data(
                withJSONObject: arguments,
                options: [.sortedKeys, .withoutEscapingSlashes]
            )
            calls.append(ToolCall(
                id: "call_\(calls.count)",
                name: name,
                argumentsJSON: String(decoding: data, as: UTF8.self)
            ))
        }

        guard !calls.isEmpty else {
            throw GLM52ToolCodecError.malformed("empty tool-call sequence")
        }
        let visible = String(text[..<first]).trimmingCharacters(in: .whitespacesAndNewlines)
        return GLM52ToolParseResult(
            calls: calls,
            visibleText: visible,
            rawToolText: String(text[first..<text.endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    /// Non-executing convenience parser. Malformed markup is returned as
    /// ordinary visible text instead of producing a partial invocation.
    public static func parse(_ text: String, tools: [ToolSpec] = []) -> GLM52ToolParseResult {
        (try? parseStrict(text, tools: tools))
            ?? GLM52ToolParseResult(calls: [], visibleText: text, rawToolText: nil)
    }

    private static func typedValue(_ raw: String, type: String?) -> Any {
        guard let type, type != "string",
              let data = raw.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
              valueMatches(type: type, value: value) else {
            return raw
        }
        return value
    }

    private static func valueMatches(type: String, value: Any) -> Bool {
        switch type {
        case "boolean":
            return (value as? NSNumber).map { String(cString: $0.objCType) == "c" } ?? false
        case "integer":
            guard let number = value as? NSNumber,
                  String(cString: number.objCType) != "c" else { return false }
            return number.doubleValue.rounded() == number.doubleValue
        case "number":
            guard let number = value as? NSNumber else { return false }
            return String(cString: number.objCType) != "c"
        case "array": return value is [Any]
        case "object": return value is [String: Any]
        case "null": return value is NSNull
        default: return true
        }
    }

    private struct ParameterSchema {
        let types: [String: String]
        let propertyNames: Set<String>
        let required: Set<String>
    }

    private static func parameterSchema(for tool: ToolSpec) -> ParameterSchema {
        guard let data = tool.parametersJSON.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let properties = root["properties"] as? [String: Any] else {
            return ParameterSchema(types: [:], propertyNames: [], required: [])
        }
        var result: [String: String] = [:]
        for (name, rawSchema) in properties {
            guard let schema = rawSchema as? [String: Any] else { continue }
            if let type = schema["type"] as? String {
                result[name] = type
            } else if let types = schema["type"] as? [String],
                      let nonNull = types.first(where: { $0 != "null" }) {
                result[name] = nonNull
            }
        }
        let required = Set((root["required"] as? [String]) ?? [])
        return ParameterSchema(
            types: result,
            propertyNames: Set(properties.keys),
            required: required
        )
    }

    private static func parameterTypes(for tool: ToolSpec) -> [String: String] {
        parameterSchema(for: tool).types
    }

    private static func neutralizeJSONStrings(_ value: Any) -> Any {
        if let string = value as? String {
            return GLM52ConversationProtocol.neutralizeControlTokens(in: string)
        }
        if let array = value as? [Any] { return array.map(neutralizeJSONStrings) }
        if let object = value as? [String: Any] {
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, item) in object {
                let safeKey = GLM52ConversationProtocol.neutralizeControlTokens(in: key)
                result[safeKey] = neutralizeJSONStrings(item)
            }
            return result
        }
        return value
    }

    private static func requireIdentifier(_ value: String) throws {
        guard !value.isEmpty,
              value.unicodeScalars.allSatisfy({ identifierCharacters.contains($0) }) else {
            throw GLM52ToolCodecError.invalidIdentifier(value)
        }
    }

    private static func skipWhitespace(in text: String, cursor: inout String.Index) {
        while cursor < text.endIndex, text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
    }

    private static func consume(_ literal: String, in text: String,
                                cursor: inout String.Index) -> Bool {
        guard text[cursor...].hasPrefix(literal) else { return false }
        cursor = text.index(cursor, offsetBy: literal.count)
        return true
    }

    private static func isPartialPrefix(at cursor: String.Index, of literal: String,
                                        in text: String) -> Bool {
        let suffix = String(text[cursor...])
        return !suffix.isEmpty && suffix.count < literal.count && literal.hasPrefix(suffix)
    }

    private static func toolCallIsInsideThinking(_ prefix: String) -> Bool {
        var cursor = prefix.startIndex
        var depth = 0
        while cursor < prefix.endIndex {
            let open = prefix.range(of: GLM52ConversationProtocol.thinkOpen,
                                    range: cursor..<prefix.endIndex)
            let close = prefix.range(of: GLM52ConversationProtocol.thinkClose,
                                     range: cursor..<prefix.endIndex)
            switch (open, close) {
            case (nil, nil): return depth > 0
            case (.some(let o), nil):
                depth += 1
                cursor = o.upperBound
            case (nil, .some(let c)):
                depth = max(0, depth - 1)
                cursor = c.upperBound
            case (.some(let o), .some(let c)) where o.lowerBound < c.lowerBound:
                depth += 1
                cursor = o.upperBound
            case (_, .some(let c)):
                depth = max(0, depth - 1)
                cursor = c.upperBound
            }
        }
        return depth > 0
    }

}

public enum GLM52ToolStreamState: Sendable, Equatable {
    case text
    case collectingToolCall
    case finished
    case failed(String)
}

/// Chunk-boundary-safe accumulator for streamed token text. It intentionally
/// executes nothing until `finish`: an incomplete later argument invalidates
/// the whole sequence rather than leaking an earlier call.
public final class GLM52IncrementalToolParser {
    public private(set) var state: GLM52ToolStreamState = .text
    public private(set) var bufferedText = ""

    public init() {}

    public func feed(_ chunk: String) {
        switch state {
        case .finished, .failed:
            return
        case .text, .collectingToolCall:
            bufferedText += chunk
            if bufferedText.contains(GLM52ConversationProtocol.toolCallOpen) {
                state = .collectingToolCall
            }
        }
    }

    public func finish(tools: [ToolSpec] = []) throws -> GLM52ToolParseResult {
        do {
            let result = try GLM52ToolCodec.parseStrict(bufferedText, tools: tools)
            state = .finished
            return result
        } catch {
            state = .failed(String(describing: error))
            throw error
        }
    }

    public func reset() {
        bufferedText = ""
        state = .text
    }
}
