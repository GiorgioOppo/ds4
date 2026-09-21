import Foundation

/// ChatML contract shared by Bonsai2 and Qwen3.8, with distinct effort prompts.
public enum QwenChatRenderer {
    public static let highEffort = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."

    public static func safe(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "<|", with: "<\u{200B}|")
            .replacingOccurrences(of: "<think>", with: "<\u{200B}think>")
            .replacingOccurrences(of: "</think>", with: "<\u{200B}/think>")
        // These delimiters are optional special tokens in Qwen vocabularies.
        // Payload text must not become protocol controls in the rendered scan.
        for marker in ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"] {
            result = result.replacingOccurrences(of: marker, with: "<\u{200B}" + marker.dropFirst())
        }
        return result
    }

    public static func render(turns: [ChatTurn], tools: [ToolSpec] = [],
                              architecture: ModelArchitectureID,
                              reasoning: ThinkMode = .none) throws -> String {
        guard architecture == .bonsai2 || architecture == .qwen38FlashNext else {
            throw ModelArchitectureError.unsupportedArchitecture(architecture)
        }
        try ToolHistoryValidator.validate(turns)
        var system: [String] = []
        if reasoning.enabled && architecture == .qwen38FlashNext { system.append(highEffort) }
        if !tools.isEmpty { system.append(try QwenToolCodec.toolsPrompt(tools)) }
        for case .system(let content) in turns {
            let value = safe(content.trimmingCharacters(in: .whitespacesAndNewlines))
            if !value.isEmpty { system.append(value) }
        }
        var out = system.isEmpty ? "" : "<|im_start|>system\n" + system.joined(separator: "\n\n") + "<|im_end|>\n"
        // Qwen's tool-response wrappers contain neither names nor call IDs.
        // Restore parallel results to invocation order before removing IDs.
        var ordered = turns, callOrder: [String: Int] = [:], index = 0
        while index < ordered.count {
            if case .assistant(_, let calls) = ordered[index] {
                callOrder = [:]
                for (offset, call) in calls.enumerated() where !call.id.isEmpty { callOrder[call.id] = offset }
            }
            if case .toolResult = ordered[index] {
                var end = index + 1
                while end < ordered.count {
                    guard case .toolResult = ordered[end] else { break }; end += 1
                }
                let sorted = ordered[index..<end].enumerated().sorted { lhs, rhs in
                    func rank(_ turn: ChatTurn, fallback: Int) -> Int {
                        guard case .toolResult(let id, _, _) = turn else { return fallback }
                        return callOrder[id] ?? (callOrder.count + fallback)
                    }
                    return rank(lhs.element, fallback: lhs.offset) < rank(rhs.element, fallback: rhs.offset)
                }.map(\.element)
                ordered.replaceSubrange(index..<end, with: sorted); index = end
            } else { index += 1 }
        }
        var toolOpen = false, pending = false
        for turn in ordered {
            if case .system = turn { continue }
            if case .toolResult(_, _, let content) = turn {
                if !toolOpen { out += "<|im_start|>user" }
                out += "\n<tool_response>\n" + safe(content).replacingOccurrences(of: "</tool_response>", with: "&lt;/tool_response>") + "\n</tool_response>"
                toolOpen = true; pending = true
                continue
            }
            if toolOpen { out += "<|im_end|>\n"; toolOpen = false }
            switch turn {
            case .user(let content):
                out += "<|im_start|>user\n" + safe(content.trimmingCharacters(in: .whitespacesAndNewlines)) + "<|im_end|>\n"
                pending = true
            case .assistant(let content, let calls):
                // Persisted history contains visible answer text, not a replay
                // of private reasoning. This matches an empty reasoning block.
                out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + safe(content)
                if !calls.isEmpty {
                    if !content.isEmpty { out += "\n\n" }
                    out += try QwenToolCodec.render(calls)
                }
                out += "<|im_end|>\n"; pending = false
            case .system, .toolResult: break
            }
        }
        if toolOpen { out += "<|im_end|>\n" }
        if pending {
            out += reasoning.enabled ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }
        return out
    }
}

/// Native Qwen function/parameter XML, as used by the pinned C server.
public enum QwenToolCodec {
    private static func identifier(_ text: String) throws {
        guard !text.isEmpty, text.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-." )).contains($0) }) else {
            throw GLM52ToolCodecError.invalidIdentifier(text)
        }
    }
    public static func toolsPrompt(_ tools: [ToolSpec]) throws -> String {
        let schemas = try tools.map { "{\"type\": \"function\", \"function\": " + (try GLM52ToolCodec.functionSchemaJSON($0)) + "}" }.joined(separator: "\n")
        return "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + QwenChatRenderer.safe(schemas) + "\n" + """
        </tools>

        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT>
        """
    }
    public static func render(_ calls: [ToolCall]) throws -> String {
        try calls.map { call in
            try identifier(call.name)
            guard let object = try JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8)) as? [String: Any] else {
                throw GLM52ToolCodecError.malformed("tool arguments must be an object")
            }
            var out = "<tool_call>\n<function=\(call.name)>\n"
            for key in object.keys.sorted() {
                try identifier(key)
                let value: String
                if let string = object[key] as? String { value = string }
                else { value = String(decoding: try JSONSerialization.data(withJSONObject: object[key]!, options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]), as: UTF8.self) }
                out += "<parameter=\(key)>\n" + QwenChatRenderer.safe(value).replacingOccurrences(of: "</parameter>", with: "&lt;/parameter>") + "\n</parameter>\n"
            }
            return out + "</function>\n</tool_call>"
        }.joined(separator: "\n")
    }

    public static func parseStrict(_ text: String, tools: [ToolSpec]) throws -> GLM52ToolParseResult {
        guard let first = text.range(of: "<tool_call>") else {
            return GLM52ToolParseResult(calls: [], visibleText: text, rawToolText: nil)
        }
        var cursor = first.lowerBound
        var normalized = String(text[..<cursor])
        func skip() { while cursor < text.endIndex && text[cursor].isWhitespace { cursor = text.index(after: cursor) } }
        func take(_ literal: String) throws {
            guard text[cursor...].hasPrefix(literal) else { throw GLM52ToolCodecError.incomplete(literal) }
            cursor = text.index(cursor, offsetBy: literal.count)
        }
        func until(_ literal: String) throws -> String {
            guard let range = text.range(of: literal, range: cursor..<text.endIndex) else { throw GLM52ToolCodecError.incomplete(literal) }
            let value = String(text[cursor..<range.lowerBound]); cursor = range.upperBound; return value
        }
        while cursor < text.endIndex {
            skip(); if cursor == text.endIndex { break }
            try take("<tool_call>"); skip(); try take("<function=")
            let name = try until(">"); try identifier(name); skip()
            normalized += "<tool_call>" + name
            while !text[cursor...].hasPrefix("</function>") {
                try take("<parameter=")
                let key = try until(">"); try identifier(key)
                var value = try until("</parameter>")
                // Only the framing newlines are removed; indentation and
                // leading/trailing spaces in string parameters are data.
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }
                guard !value.contains("</arg_value>"), !value.contains("</tool_call>") else {
                    throw GLM52ToolCodecError.malformed("nested tool delimiter in parameter")
                }
                normalized += "<arg_key>" + key + "</arg_key><arg_value>" + value + "</arg_value>"
                skip()
            }
            try take("</function>"); skip(); try take("</tool_call>")
            normalized += "</tool_call>"; skip()
        }
        let parsed = try GLM52ToolCodec.parseStrict(normalized, tools: tools)
        return GLM52ToolParseResult(calls: parsed.calls, visibleText: parsed.visibleText, rawToolText: String(text[first.lowerBound...]))
    }
}
