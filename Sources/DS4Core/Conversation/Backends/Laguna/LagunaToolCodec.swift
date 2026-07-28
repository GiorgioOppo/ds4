import Foundation

public struct LagunaToolParseResult: Sendable, Equatable {
    public let calls: [ToolCall]
    public let visibleText: String
    public let rawToolText: String?

    public init(calls: [ToolCall], visibleText: String, rawToolText: String?) {
        self.calls = calls
        self.visibleText = visibleText
        self.rawToolText = rawToolText
    }
}

/// Result of the reference-server parse (`parse_glm_generated_message_ex`):
/// content and reasoning are split at the last thinking boundary, tool-call
/// arguments are the C-canonical all-string JSON object, and the raw tool
/// block is preserved for byte-stable re-rendering.
public struct LagunaServerParseResult: Sendable, Equatable {
    public let content: String
    public let reasoning: String?
    public let calls: [ToolCall]
    public let rawToolText: String?

    public init(content: String, reasoning: String?,
                calls: [ToolCall], rawToolText: String?) {
        self.content = content
        self.reasoning = reasoning
        self.calls = calls
        self.rawToolText = rawToolText
    }
}

public enum LagunaToolCodecError: Error, Sendable, Equatable, CustomStringConvertible {
    case incomplete(String)
    case malformed(String)
    case invalidIdentifier(String)
    case toolCallInsideThinking

    public var description: String {
        switch self {
        case .incomplete(let expected): return "incomplete Laguna tool call; expected \(expected)"
        case .malformed(let reason): return "malformed Laguna tool call: \(reason)"
        case .invalidIdentifier(let value): return "invalid Laguna tool identifier: \(value)"
        case .toolCallInsideThinking: return "Laguna tool calls are not allowed inside <think>"
        }
    }
}

/// Renderer and parsers for Laguna's native flat XML tool grammar.
///
/// Upstream `ds4` renders Laguna tool calls with the same argument helpers as
/// GLM (`append_glm_arguments_from_json`), so the grammar is identical:
/// `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`.
/// The differences live around the grammar, not inside it — Laguna declares
/// tools in an `<available_tools>` system section, `<tool_call>`/`</tool_call>`
/// are dedicated vocabulary tokens while the argument tags are plain text, and
/// consecutive calls are concatenated without separators.
///
/// Two parsers are provided.  `parseServer` is the exact port of the reference
/// server parser (lenient: any tool name, duplicate keys allowed, calls inside
/// unterminated thinking ignored, trailing text dropped, all-string C-spaced
/// arguments).  `parseStrict` keeps the port's agent-grade validations
/// (undeclared tools/arguments, duplicates, identifier charset, required
/// arguments, schema-typed values) on top of the same grammar and entity
/// decoding.
public enum LagunaToolCodec {
    private static let identifierCharacters = CharacterSet.alphanumerics
        .union(CharacterSet(charactersIn: "_-."))

    // MARK: Tool declarations

    /// Native system-section body used by Laguna when tools are available.
    /// The reference server appends this inside the `<system>` block, joined
    /// to the system prompt with one blank line.
    public static func toolsSection(_ tools: [ToolSpec],
                                    neutralize: Bool = true) throws -> String {
        guard !tools.isEmpty else { return "" }
        let schemas = try tools
            .map { try functionSchemaJSON($0, neutralize: neutralize) }
            .joined(separator: "\n")
        // Keep this wording and whitespace aligned with
        // `render_laguna_chat_prompt_text` in upstream `ds4_server.c`.
        return """
        ### Tools

        You may call functions to assist with the user query.
        All available function signatures are listed below:
        \(LagunaConversationProtocol.availableToolsOpen)
        \(schemas)
        \(LagunaConversationProtocol.availableToolsClose)
        """
    }

    /// The function object rendered inside `<available_tools>`, encoded
    /// deterministically for stable prompt/KV-cache prefixes.  The OpenAI
    /// `{"type":"function","function": ...}` transport wrapper is deliberately
    /// omitted, matching the GLM port convention.  (The reference server
    /// passes the client JSON through verbatim; this port re-encodes it so
    /// the same `ToolSpec` always yields the same prompt bytes.)
    public static func functionSchemaJSON(_ tool: ToolSpec,
                                          neutralize: Bool = true) throws -> String {
        try requireIdentifier(tool.name)
        let parameters: Any
        if let data = tool.parametersJSON.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data),
           object is [String: Any] {
            parameters = neutralize ? neutralizeJSONStrings(object) : object
        } else {
            parameters = ["type": "object", "properties": [:]] as [String: Any]
        }
        let function: [String: Any] = [
            "description": neutralize
                ? LagunaConversationProtocol.neutralizeControlTokens(in: tool.description)
                : tool.description,
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

    /// Render structured historical calls exactly like
    /// `append_laguna_tool_calls_text`: blocks concatenated without
    /// separators; arguments in the tool schema's property *declaration*
    /// order first, then the remaining arguments in their original JSON
    /// order; string values as their decoded text, other values as minified
    /// raw JSON; invalid argument JSON preserved under an `arguments` key.
    public static func renderToolCalls(_ calls: [ToolCall],
                                       tools: [ToolSpec] = [],
                                       neutralize: Bool = true) throws -> String {
        var orderByTool: [String: [String]] = [:]
        for tool in tools {
            orderByTool[tool.name] =
                orderedPropertyNames(parametersJSON: tool.parametersJSON) ?? []
        }
        var blocks: [String] = []
        blocks.reserveCapacity(calls.count)

        for call in calls {
            try requireIdentifier(call.name)
            var block = LagunaConversationProtocol.toolCallOpen + call.name
            if let arguments = scanOrderedArguments(call.argumentsJSON) {
                var used = [Bool](repeating: false, count: arguments.count)
                var sequence: [Int] = []
                for property in orderByTool[call.name] ?? [] {
                    guard let index = arguments.indices.first(where: {
                        !used[$0] && arguments[$0].key == property
                    }) else { continue }
                    used[index] = true
                    sequence.append(index)
                }
                sequence.append(contentsOf: arguments.indices.filter { !used[$0] })
                for index in sequence {
                    let argument = arguments[index]
                    try requireIdentifier(argument.key)
                    block += LagunaConversationProtocol.argumentKeyOpen
                    block += escapeTagBody(argument.key,
                                           closingTag: LagunaConversationProtocol.argumentKeyClose,
                                           neutralize: neutralize)
                    block += LagunaConversationProtocol.argumentKeyClose
                    block += LagunaConversationProtocol.argumentValueOpen
                    block += escapeTagBody(argument.value,
                                           closingTag: LagunaConversationProtocol.argumentValueClose,
                                           neutralize: neutralize)
                    block += LagunaConversationProtocol.argumentValueClose
                }
            } else {
                block += LagunaConversationProtocol.argumentKeyOpen + "arguments"
                block += LagunaConversationProtocol.argumentKeyClose
                block += LagunaConversationProtocol.argumentValueOpen
                block += escapeTagBody(call.argumentsJSON,
                                       closingTag: LagunaConversationProtocol.argumentValueClose,
                                       neutralize: neutralize)
                block += LagunaConversationProtocol.argumentValueClose
            }
            block += LagunaConversationProtocol.toolCallClose
            blocks.append(block)
        }
        return blocks.joined()
    }

    /// Tool output is plain data inside the `<tool_response>` wrapper.
    /// Preserve literal `<`, `>` and `&` so shell output and file snippets stay
    /// intact, but escape the exact closing sentinel so a malicious or
    /// accidental payload cannot terminate the wrapper early
    /// (`laguna_chat_append_tool_response` in upstream `ds4.c`).
    public static func escapeToolResponse(_ content: String) -> String {
        escapeTagBody(content,
                      closingTag: LagunaConversationProtocol.toolResponseClose,
                      neutralize: true)
    }

    private static func escapeTagBody(_ text: String, closingTag: String,
                                      neutralize: Bool) -> String {
        let escapedClose = "&lt;" + closingTag.dropFirst()
        let protected = text.replacingOccurrences(of: closingTag, with: escapedClose)
        return neutralize
            ? LagunaConversationProtocol.neutralizeControlTokens(in: protected)
            : protected
    }

    // MARK: Reference-server parsing

    /// Exact port of `parse_glm_generated_message_ex` for the Laguna syntax.
    ///
    /// With `requireThinkingClosed` the search starts after the *last*
    /// `</think>`; when thinking never closes, any tool markup inside the
    /// reasoning is ignored and the whole text is returned as content and
    /// reasoning.  Any non-tool text after the final `</tool_call>` is
    /// dropped, duplicate keys are kept, the tool name only needs to be
    /// non-empty, and arguments become the C-canonical
    /// `{"key": "value", …}` all-string object in order of appearance.
    /// Returns `nil` where the reference returns false (malformed markup);
    /// the caller decides between recovery and plain-text fallback.
    public static func parseServer(_ text: String,
                                   requireThinkingClosed: Bool = false)
        -> LagunaServerParseResult? {
        let p = LagunaConversationProtocol.self
        let bytes = Array(text.utf8)
        let open = Array(p.toolCallOpen.utf8)
        let close = Array(p.toolCallClose.utf8)
        let keyOpen = Array(p.argumentKeyOpen.utf8)
        let keyClose = Array(p.argumentKeyClose.utf8)
        let valueOpen = Array(p.argumentValueOpen.utf8)
        let valueClose = Array(p.argumentValueClose.utf8)
        let thinkClose = Array(p.thinkClose.utf8)

        var searchStart = 0
        if requireThinkingClosed {
            guard let lastThink = lastIndex(of: thinkClose, in: bytes) else {
                let (content, reasoning) = splitReasoningContent(bytes, length: bytes.count)
                return LagunaServerParseResult(content: content,
                                               reasoning: reasoning,
                                               calls: [], rawToolText: nil)
            }
            searchStart = lastThink + thinkClose.count
        }

        guard let start = firstIndex(of: open, in: bytes, from: searchStart) else {
            let (content, reasoning) = splitReasoningContent(bytes, length: bytes.count)
            return LagunaServerParseResult(content: content,
                                           reasoning: reasoning,
                                           calls: [], rawToolText: nil)
        }

        var rawBlockStart = start
        if start >= 2, bytes[start - 2] == 0x0a, bytes[start - 1] == 0x0a {
            rawBlockStart = start - 2
        }
        var contentLength = rawBlockStart
        while contentLength > 0,
              LagunaConversationProtocol.isASCIIWhitespace(
                Unicode.Scalar(bytes[contentLength - 1])) {
            contentLength -= 1
        }

        var calls: [ToolCall] = []
        var position = start
        while true {
            position = skipASCIIWhitespace(bytes, position)
            guard matches(bytes, at: position, open) else { break }
            position += open.count

            guard let closeIndex = firstIndex(of: close, in: bytes, from: position) else {
                return nil
            }
            var argIndex = firstIndex(of: keyOpen, in: bytes, from: position)
            if let a = argIndex, a > closeIndex { argIndex = nil }

            let nameEndLimit = argIndex ?? closeIndex
            let name = trimmedASCIIString(bytes, position, nameEndLimit)
            guard !name.isEmpty else { return nil }
            position = nameEndLimit

            var argumentsJSON = "{"
            var first = true
            while true {
                position = skipASCIIWhitespace(bytes, position)
                if matches(bytes, at: position, close) {
                    position += close.count
                    break
                }
                guard matches(bytes, at: position, keyOpen) else { return nil }
                position += keyOpen.count
                guard let keyEnd = firstIndex(of: keyClose, in: bytes, from: position),
                      keyEnd <= closeIndex else { return nil }
                let key = dsmlUnescape(trimmedASCIIString(bytes, position, keyEnd))
                position = keyEnd + keyClose.count
                position = skipASCIIWhitespace(bytes, position)
                guard matches(bytes, at: position, valueOpen) else { return nil }
                position += valueOpen.count
                guard let valueEnd = firstIndex(of: valueClose, in: bytes, from: position),
                      valueEnd <= closeIndex else { return nil }
                let value = dsmlUnescape(
                    String(decoding: bytes[position..<valueEnd], as: UTF8.self))
                if !first { argumentsJSON += ", " }
                first = false
                argumentsJSON += jsonEscape(key) + ": " + jsonEscape(value)
                position = valueEnd + valueClose.count
            }
            argumentsJSON += "}"
            calls.append(ToolCall(id: "call_\(calls.count)",
                                  name: name,
                                  argumentsJSON: argumentsJSON))

            let next = skipASCIIWhitespace(bytes, position)
            position = next
            if !matches(bytes, at: next, open) { break }
        }

        guard !calls.isEmpty else { return nil }
        let rawToolText = String(decoding: bytes[rawBlockStart..<position], as: UTF8.self)
        let (content, reasoning) = splitReasoningContent(bytes, length: contentLength)
        return LagunaServerParseResult(content: content,
                                       reasoning: reasoning,
                                       calls: calls,
                                       rawToolText: rawToolText)
    }

    // MARK: Strict parsing

    /// Parse zero or more complete calls. Once a call starts, all remaining
    /// non-whitespace text must be native Laguna call syntax; malformed or
    /// partial blocks never escape as executable calls.  Stricter than the
    /// reference server on purpose (agent-grade): undeclared tools and
    /// arguments, duplicate keys, non-identifier names and trailing text are
    /// errors here where the server tolerates or drops them — use
    /// `parseServer` for the reference behavior.
    public static func parseStrict(_ text: String, tools: [ToolSpec] = []) throws
        -> LagunaToolParseResult {
        let open = LagunaConversationProtocol.toolCallOpen
        guard let first = text.range(of: open)?.lowerBound else {
            return LagunaToolParseResult(calls: [], visibleText: text, rawToolText: nil)
        }
        if toolCallIsInsideThinking(String(text[..<first])) {
            throw LagunaToolCodecError.toolCallInsideThinking
        }

        var schemas: [String: ParameterSchema] = [:]
        for tool in tools { schemas[tool.name] = parameterSchema(for: tool) }
        var cursor = first
        var calls: [ToolCall] = []

        while cursor < text.endIndex {
            skipWhitespace(in: text, cursor: &cursor)
            guard cursor < text.endIndex else { break }
            guard text[cursor...].hasPrefix(open) else {
                throw LagunaToolCodecError.malformed("non-whitespace text after a tool call")
            }
            cursor = text.index(cursor, offsetBy: open.count)

            guard let structural = text[cursor...].firstIndex(of: "<") else {
                throw LagunaToolCodecError.incomplete("<arg_key> or </tool_call>")
            }
            let name = String(LagunaConversationProtocol.trimASCIIWhitespace(
                text[cursor..<structural]))
            try requireIdentifier(name)
            if !tools.isEmpty, schemas[name] == nil {
                throw LagunaToolCodecError.malformed("undeclared tool \(name)")
            }
            cursor = structural

            var arguments: [String: Any] = [:]
            while true {
                skipWhitespace(in: text, cursor: &cursor)
                if consume(LagunaConversationProtocol.toolCallClose, in: text, cursor: &cursor) {
                    break
                }
                if isPartialPrefix(at: cursor,
                                   of: LagunaConversationProtocol.toolCallClose,
                                   in: text) {
                    throw LagunaToolCodecError.incomplete(LagunaConversationProtocol.toolCallClose)
                }
                guard consume(LagunaConversationProtocol.argumentKeyOpen,
                              in: text, cursor: &cursor) else {
                    if isPartialPrefix(at: cursor,
                                       of: LagunaConversationProtocol.argumentKeyOpen,
                                       in: text) {
                        throw LagunaToolCodecError.incomplete(LagunaConversationProtocol.argumentKeyOpen)
                    }
                    throw LagunaToolCodecError.malformed("expected <arg_key> or </tool_call>")
                }

                guard let keyEnd = text.range(
                    of: LagunaConversationProtocol.argumentKeyClose,
                    range: cursor..<text.endIndex
                ) else {
                    if text.range(of: LagunaConversationProtocol.toolCallClose,
                                  range: cursor..<text.endIndex) != nil {
                        throw LagunaToolCodecError.malformed("unterminated <arg_key>")
                    }
                    throw LagunaToolCodecError.incomplete(LagunaConversationProtocol.argumentKeyClose)
                }
                if let callEnd = text.range(
                    of: LagunaConversationProtocol.toolCallClose,
                    range: cursor..<text.endIndex
                ), callEnd.lowerBound < keyEnd.lowerBound {
                    throw LagunaToolCodecError.malformed("unterminated <arg_key>")
                }
                // Tag bodies carry the renderer's XML-entity escapes
                // (`dsml_unescape_text` upstream): decode them so a
                // render→parse round trip is stable.
                let key = dsmlUnescape(String(
                    LagunaConversationProtocol.trimASCIIWhitespace(
                        text[cursor..<keyEnd.lowerBound])))
                try requireIdentifier(key)
                if let schema = schemas[name], !schema.propertyNames.contains(key) {
                    throw LagunaToolCodecError.malformed(
                        "undeclared argument \(key) for tool \(name)"
                    )
                }
                guard arguments[key] == nil else {
                    throw LagunaToolCodecError.malformed("duplicate argument \(key)")
                }
                cursor = keyEnd.upperBound
                skipWhitespace(in: text, cursor: &cursor)

                guard consume(LagunaConversationProtocol.argumentValueOpen,
                              in: text, cursor: &cursor) else {
                    if isPartialPrefix(at: cursor,
                                       of: LagunaConversationProtocol.argumentValueOpen,
                                       in: text) {
                        throw LagunaToolCodecError.incomplete(LagunaConversationProtocol.argumentValueOpen)
                    }
                    throw LagunaToolCodecError.malformed("expected <arg_value>")
                }
                guard let valueEnd = text.range(
                    of: LagunaConversationProtocol.argumentValueClose,
                    range: cursor..<text.endIndex
                ) else {
                    if text.range(of: LagunaConversationProtocol.toolCallClose,
                                  range: cursor..<text.endIndex) != nil {
                        throw LagunaToolCodecError.malformed("unterminated <arg_value>")
                    }
                    throw LagunaToolCodecError.incomplete(LagunaConversationProtocol.argumentValueClose)
                }
                if let callEnd = text.range(
                    of: LagunaConversationProtocol.toolCallClose,
                    range: cursor..<text.endIndex
                ), callEnd.lowerBound < valueEnd.lowerBound {
                    throw LagunaToolCodecError.malformed("unterminated <arg_value>")
                }
                let rawValue = dsmlUnescape(String(text[cursor..<valueEnd.lowerBound]))
                arguments[key] = typedValue(rawValue, type: schemas[name]?.types[key])
                cursor = valueEnd.upperBound
            }

            if let schema = schemas[name] {
                let missing = schema.required.subtracting(arguments.keys).sorted()
                if !missing.isEmpty {
                    throw LagunaToolCodecError.malformed(
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
            throw LagunaToolCodecError.malformed("empty tool-call sequence")
        }
        let visible = String(text[..<first]).trimmingCharacters(in: .whitespacesAndNewlines)
        return LagunaToolParseResult(
            calls: calls,
            visibleText: visible,
            rawToolText: String(text[first..<text.endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    /// Non-executing convenience parser. Malformed markup is returned as
    /// ordinary visible text instead of producing a partial invocation.
    public static func parse(_ text: String, tools: [ToolSpec] = []) -> LagunaToolParseResult {
        (try? parseStrict(text, tools: tools))
            ?? LagunaToolParseResult(calls: [], visibleText: text, rawToolText: nil)
    }

    // MARK: Entity decoding (`dsml_unescape_text`)

    /// Decode the renderer's XML entities (`&amp; &lt; &gt; &quot; &apos;`);
    /// any other `&` stays literal, exactly like the reference.
    static func dsmlUnescape(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var output = ""
        output.reserveCapacity(text.count)
        var view = Substring(text)
        while let amp = view.firstIndex(of: "&") {
            output += view[..<amp]
            view = view[amp...]
            var replaced = false
            for (entity, decoded) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                                      ("&quot;", "\""), ("&apos;", "'")] {
                if view.hasPrefix(entity) {
                    output += decoded
                    view = view.dropFirst(entity.count)
                    replaced = true
                    break
                }
            }
            if !replaced {
                output += "&"
                view = view.dropFirst()
            }
        }
        output += view
        return output
    }

    // MARK: Ordered JSON scanning (`json_args_parse` / `parse_schema_properties`)

    /// Scan a JSON object preserving declaration order: string values are
    /// decoded to their text, every other value keeps its minified raw JSON.
    /// Returns nil when the text is not a JSON object, mirroring
    /// `json_args_parse` so the caller falls back to the `arguments` key.
    static func scanOrderedArguments(_ json: String)
        -> [(key: String, value: String, isString: Bool)]? {
        var scanner = RawJSONScanner(Array(json.utf8))
        scanner.skipWhitespace()
        guard scanner.consume(0x7b) else { return nil } // {
        scanner.skipWhitespace()
        var result: [(key: String, value: String, isString: Bool)] = []
        while let byte = scanner.peek(), byte != 0x7d { // }
            guard let key = scanner.parseString() else { return nil }
            scanner.skipWhitespace()
            guard scanner.consume(0x3a) else { return nil } // :
            scanner.skipWhitespace()
            if scanner.peek() == 0x22 { // "
                guard let value = scanner.parseString() else { return nil }
                result.append((key, value, true))
            } else {
                guard let raw = scanner.rawValue() else { return nil }
                result.append((key, minifyRawJSON(raw), false))
            }
            scanner.skipWhitespace()
            _ = scanner.consume(0x2c) // ,
            scanner.skipWhitespace()
        }
        guard scanner.consume(0x7d) else { return nil }
        return result
    }

    /// Top-level `properties` names of a JSON-Schema object in declaration
    /// order (`parse_schema_properties`); nil when the schema does not parse,
    /// in which case arguments keep their original order.
    static func orderedPropertyNames(parametersJSON: String) -> [String]? {
        var scanner = RawJSONScanner(Array(parametersJSON.utf8))
        scanner.skipWhitespace()
        guard scanner.consume(0x7b) else { return nil }
        scanner.skipWhitespace()
        var names: [String] = []
        while let byte = scanner.peek(), byte != 0x7d {
            guard let key = scanner.parseString() else { return nil }
            scanner.skipWhitespace()
            guard scanner.consume(0x3a) else { return nil }
            if key == "properties" {
                scanner.skipWhitespace()
                guard scanner.consume(0x7b) else { return nil }
                scanner.skipWhitespace()
                while let inner = scanner.peek(), inner != 0x7d {
                    guard let property = scanner.parseString() else { return nil }
                    scanner.skipWhitespace()
                    guard scanner.consume(0x3a) else { return nil }
                    names.append(property)
                    guard scanner.skipValue() else { return nil }
                    scanner.skipWhitespace()
                    _ = scanner.consume(0x2c)
                    scanner.skipWhitespace()
                }
                guard scanner.consume(0x7d) else { return nil }
            } else {
                guard scanner.skipValue() else { return nil }
            }
            scanner.skipWhitespace()
            _ = scanner.consume(0x2c)
            scanner.skipWhitespace()
        }
        guard scanner.consume(0x7d) else { return nil }
        return names
    }

    /// Strip inter-token whitespace from raw JSON while preserving string
    /// bytes exactly (`json_minify_raw_value`).
    static func minifyRawJSON(_ raw: String) -> String {
        var output = [UInt8]()
        output.reserveCapacity(raw.utf8.count)
        var inString = false
        var escaped = false
        for byte in raw.utf8 {
            if inString {
                output.append(byte)
                if escaped { escaped = false }
                else if byte == 0x5c { escaped = true } // backslash
                else if byte == 0x22 { inString = false }
            } else if byte == 0x22 {
                inString = true
                output.append(byte)
            } else if !LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(byte)) {
                output.append(byte)
            }
        }
        return String(decoding: output, as: UTF8.self)
    }

    /// Minimal JSON string escaping used by the reference (`json_escape`):
    /// quote and backslash, named \n \r \t, other C0 controls as \u00xx.
    static func jsonEscape(_ text: String) -> String {
        var output = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            default:
                if scalar.value < 0x20 {
                    output += String(format: "\\u%04x", scalar.value)
                } else {
                    output.unicodeScalars.append(scalar)
                }
            }
        }
        return output + "\""
    }

    private struct RawJSONScanner {
        let bytes: [UInt8]
        var position = 0

        init(_ bytes: [UInt8]) { self.bytes = bytes }

        func peek() -> UInt8? {
            position < bytes.count ? bytes[position] : nil
        }

        mutating func consume(_ byte: UInt8) -> Bool {
            guard peek() == byte else { return false }
            position += 1
            return true
        }

        mutating func skipWhitespace() {
            while let byte = peek(),
                  LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(byte)) {
                position += 1
            }
        }

        /// JSON string with standard escapes decoded (upstream `json_string`),
        /// including surrogate pairs; a lone surrogate becomes U+FFFD.
        mutating func parseString() -> String? {
            skipWhitespace()
            guard consume(0x22) else { return nil }
            var scalars: [Unicode.Scalar] = []
            var utf8Buffer: [UInt8] = []
            func flushUTF8() {
                guard !utf8Buffer.isEmpty else { return }
                scalars.append(contentsOf:
                    String(decoding: utf8Buffer, as: UTF8.self).unicodeScalars)
                utf8Buffer.removeAll(keepingCapacity: true)
            }
            while let byte = peek(), byte != 0x22 {
                position += 1
                if byte != 0x5c {
                    utf8Buffer.append(byte)
                    continue
                }
                flushUTF8()
                guard let escapeByte = peek() else { return nil }
                position += 1
                switch escapeByte {
                case 0x22: scalars.append("\"")
                case 0x5c: scalars.append("\\")
                case 0x2f: scalars.append("/")
                case 0x62: scalars.append("\u{08}")
                case 0x66: scalars.append("\u{0C}")
                case 0x6e: scalars.append("\n")
                case 0x72: scalars.append("\r")
                case 0x74: scalars.append("\t")
                case 0x75: // \uXXXX
                    guard var code = parseHex4() else { return nil }
                    if (0xd800...0xdbff).contains(code) {
                        let saved = position
                        if peek() == 0x5c, bytes.count > position + 1,
                           bytes[position + 1] == 0x75 {
                            position += 2
                            if let low = parseHex4(), (0xdc00...0xdfff).contains(low) {
                                code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
                            } else {
                                position = saved
                                code = 0xfffd
                            }
                        } else {
                            code = 0xfffd
                        }
                    } else if (0xdc00...0xdfff).contains(code) {
                        code = 0xfffd
                    }
                    scalars.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                default:
                    return nil
                }
            }
            flushUTF8()
            guard consume(0x22) else { return nil }
            var output = ""
            output.unicodeScalars.append(contentsOf: scalars)
            return output
        }

        private mutating func parseHex4() -> UInt32? {
            var value: UInt32 = 0
            for _ in 0..<4 {
                guard let byte = peek() else { return nil }
                position += 1
                let digit: UInt32
                switch byte {
                case 0x30...0x39: digit = UInt32(byte - 0x30)
                case 0x41...0x46: digit = UInt32(byte - 0x41 + 10)
                case 0x61...0x66: digit = UInt32(byte - 0x61 + 10)
                default: return nil
                }
                value = value << 4 | digit
            }
            return value
        }

        /// Skip one complete JSON value; false when the input ends inside it.
        mutating func skipValue() -> Bool {
            skipWhitespace()
            guard let byte = peek() else { return false }
            switch byte {
            case 0x22:
                return skipString()
            case 0x7b, 0x5b: // { [
                let closer: UInt8 = byte == 0x7b ? 0x7d : 0x5d
                position += 1
                skipWhitespace()
                if consume(closer) { return true }
                while true {
                    guard skipValue() else { return false }
                    skipWhitespace()
                    if byte == 0x7b {
                        // Object entries: the value above was the key.
                        guard consume(0x3a) else { return false }
                        guard skipValue() else { return false }
                        skipWhitespace()
                    }
                    if consume(closer) { return true }
                    guard consume(0x2c) else { return false }
                    skipWhitespace()
                }
            default:
                // Literal (number/true/false/null): consume until delimiter.
                let start = position
                while let b = peek(),
                      b != 0x2c, b != 0x7d, b != 0x5d,
                      !LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(b)) {
                    position += 1
                }
                return position > start
            }
        }

        private mutating func skipString() -> Bool {
            guard consume(0x22) else { return false }
            var escaped = false
            while let byte = peek() {
                position += 1
                if escaped { escaped = false }
                else if byte == 0x5c { escaped = true }
                else if byte == 0x22 { return true }
            }
            return false
        }

        /// The raw byte span of one complete JSON value.
        mutating func rawValue() -> String? {
            skipWhitespace()
            let start = position
            guard skipValue() else { return nil }
            return String(decoding: bytes[start..<position], as: UTF8.self)
        }
    }

    // MARK: Byte-level helpers for the server parser

    private static func matches(_ bytes: [UInt8], at index: Int, _ needle: [UInt8]) -> Bool {
        guard index >= 0, index + needle.count <= bytes.count else { return false }
        for offset in 0..<needle.count where bytes[index + offset] != needle[offset] {
            return false
        }
        return true
    }

    private static func firstIndex(of needle: [UInt8], in bytes: [UInt8],
                                   from start: Int = 0) -> Int? {
        guard !needle.isEmpty, start <= bytes.count else { return nil }
        var index = max(0, start)
        while index + needle.count <= bytes.count {
            if matches(bytes, at: index, needle) { return index }
            index += 1
        }
        return nil
    }

    private static func lastIndex(of needle: [UInt8], in bytes: [UInt8]) -> Int? {
        var found: Int?
        var from = 0
        while let index = firstIndex(of: needle, in: bytes, from: from) {
            found = index
            from = index + 1
        }
        return found
    }

    private static func skipASCIIWhitespace(_ bytes: [UInt8], _ index: Int) -> Int {
        var index = index
        while index < bytes.count,
              LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(bytes[index])) {
            index += 1
        }
        return index
    }

    private static func trimmedASCIIString(_ bytes: [UInt8], _ start: Int, _ end: Int) -> String {
        var start = start
        var end = end
        while start < end,
              LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(bytes[start])) {
            start += 1
        }
        while end > start,
              LagunaConversationProtocol.isASCIIWhitespace(Unicode.Scalar(bytes[end - 1])) {
            end -= 1
        }
        return String(decoding: bytes[start..<end], as: UTF8.self)
    }

    /// `split_reasoning_content`: strip one leading `<think>`, split at the
    /// first `</think>`; with no close tag the text stays whole and there is
    /// no reasoning.
    private static func splitReasoningContent(_ bytes: [UInt8], length: Int)
        -> (content: String, reasoning: String?) {
        let p = LagunaConversationProtocol.self
        let slice = Array(bytes[0..<length])
        var body = 0
        let thinkOpen = Array(p.thinkOpen.utf8)
        if matches(slice, at: 0, thinkOpen) { body = thinkOpen.count }
        let thinkClose = Array(p.thinkClose.utf8)
        guard let close = firstIndex(of: thinkClose, in: slice, from: body) else {
            return (String(decoding: slice, as: UTF8.self), nil)
        }
        let reasoning = String(decoding: slice[body..<close], as: UTF8.self)
        let content = String(decoding: slice[(close + thinkClose.count)...], as: UTF8.self)
        return (content, reasoning)
    }

    // MARK: Schema helpers

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

    private static func neutralizeJSONStrings(_ value: Any) -> Any {
        if let string = value as? String {
            return LagunaConversationProtocol.neutralizeControlTokens(in: string)
        }
        if let array = value as? [Any] { return array.map(neutralizeJSONStrings) }
        if let object = value as? [String: Any] {
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, item) in object {
                let safeKey = LagunaConversationProtocol.neutralizeControlTokens(in: key)
                result[safeKey] = neutralizeJSONStrings(item)
            }
            return result
        }
        return value
    }

    private static func requireIdentifier(_ value: String) throws {
        guard !value.isEmpty,
              value.unicodeScalars.allSatisfy({ identifierCharacters.contains($0) }) else {
            throw LagunaToolCodecError.invalidIdentifier(value)
        }
    }

    private static func skipWhitespace(in text: String, cursor: inout String.Index) {
        while cursor < text.endIndex,
              let scalar = text[cursor].unicodeScalars.first,
              text[cursor].unicodeScalars.count == 1,
              LagunaConversationProtocol.isASCIIWhitespace(scalar) {
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
        // An empty suffix (chunk boundary exactly on a tag edge) counts as a
        // partial prefix: the reference agent parser waits for more bytes
        // whenever the buffer ends before the next structural tag.
        let suffix = String(text[cursor...])
        return suffix.count < literal.count && literal.hasPrefix(suffix)
    }

    private static func toolCallIsInsideThinking(_ prefix: String) -> Bool {
        var cursor = prefix.startIndex
        var depth = 0
        while cursor < prefix.endIndex {
            let open = prefix.range(of: LagunaConversationProtocol.thinkOpen,
                                    range: cursor..<prefix.endIndex)
            let close = prefix.range(of: LagunaConversationProtocol.thinkClose,
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

public enum LagunaToolStreamState: Sendable, Equatable {
    case text
    case collectingToolCall
    case finished
    case failed(String)
}

/// Chunk-boundary-safe incremental parser for streamed token text, following
/// the reference agent's tagged-syntax stream parser: incomplete markup waits
/// for more bytes, completed malformation fails immediately (so the caller
/// can surface a retryable tool error without waiting for end of stream), and
/// calls become visible as soon as their `</tool_call>` arrives.  It still
/// executes nothing by itself: `finish` performs the final validated parse.
public final class LagunaIncrementalToolParser {
    public private(set) var state: LagunaToolStreamState = .text
    public private(set) var bufferedText = ""
    /// Calls whose markup is already complete in the buffered stream.  Grows
    /// call by call; never contains partially streamed markup.
    public private(set) var completedCalls: [ToolCall] = []

    public init() {}

    public func feed(_ chunk: String) {
        switch state {
        case .finished, .failed:
            return
        case .text, .collectingToolCall:
            bufferedText += chunk
            guard bufferedText.contains(LagunaConversationProtocol.toolCallOpen) else {
                return
            }
            state = .collectingToolCall
            do {
                let result = try LagunaToolCodec.parseStrict(bufferedText)
                completedCalls = result.calls
            } catch LagunaToolCodecError.incomplete {
                // Chunk boundary inside a tag or value: wait for more bytes.
            } catch {
                state = .failed(String(describing: error))
            }
        }
    }

    public func finish(tools: [ToolSpec] = []) throws -> LagunaToolParseResult {
        do {
            let result = try LagunaToolCodec.parseStrict(bufferedText, tools: tools)
            state = .finished
            completedCalls = result.calls
            return result
        } catch {
            state = .failed(String(describing: error))
            throw error
        }
    }

    public func reset() {
        bufferedText = ""
        state = .text
        completedCalls = []
    }
}
