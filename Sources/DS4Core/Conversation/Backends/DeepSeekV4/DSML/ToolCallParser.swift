import Foundation

// MARK: - Parsing tool calls from generated text

public enum ToolCallParser {
    /// Error returned by the execution parser.  The associated reason is intended
    /// for diagnostics only: callers must treat every error as "do not execute".
    public enum StrictError: Error, Sendable, Equatable, LocalizedError {
        case malformed(String)

        public var errorDescription: String? {
            switch self {
            case .malformed(let reason):
                return "Malformed DSML tool-call block: \(reason)"
            }
        }
    }

    /// Recovery parser for display and diagnostics.  It deliberately accepts a
    /// block truncated at the end of the message, so it MUST NOT be used to decide
    /// whether a model-emitted call is safe to execute.  Use `parseStrict` there.
    public static func parse(_ text: String, markup m: ToolMarkup) -> (calls: [ToolCall], visibleText: String) {
        guard let start = text.range(of: m.callsOpen) else { return ([], text) }
        let visible = String(text[text.startIndex..<start.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let afterOpen = start.upperBound
        let end = text.range(of: m.callsClose, range: afterOpen..<text.endIndex)?.lowerBound ?? text.endIndex
        let block = String(text[afterOpen..<end])

        var calls: [ToolCall] = []
        var idx = 0
        var search = block.startIndex
        let invokePrefix = "<\(m.dsml)invoke"
        while let io = block.range(of: invokePrefix, range: search..<block.endIndex) {
            let close = block.range(of: m.invokeClose, range: io.upperBound..<block.endIndex)?.lowerBound ?? block.endIndex
            let body = String(block[io.lowerBound..<close])
            if let call = parseInvoke(body, markup: m, index: idx) { calls.append(call); idx += 1 }
            search = close < block.endIndex ? block.index(after: close) : block.endIndex
            if search >= block.endIndex { break }
        }
        return (calls, visible)
    }

    /// Parse a completed assistant message for execution.
    ///
    /// Unlike the recovery parser, this is all-or-nothing: the outer block, every
    /// invocation, and every parameter must be closed and structurally valid.  A
    /// malformed message throws and never exposes a partial list of executable
    /// calls.  Messages without DSML markup remain valid and return no calls.
    public static func parseStrict(_ text: String, markup m: ToolMarkup) throws
        -> (calls: [ToolCall], visibleText: String) {
        guard let blockOpen = text.range(of: m.callsOpen) else {
            // A closing/nested DSML tag without the outer opener is malformed,
            // not a plain answer.  This also prevents callers from accidentally
            // executing a fragment recovered from elsewhere in the message.
            guard !text.contains(m.dsml) else {
                throw StrictError.malformed("DSML markup without a tool_calls opener")
            }
            return ([], text)
        }

        let prefix = text[..<blockOpen.lowerBound]
        guard !prefix.contains(m.dsml) else {
            throw StrictError.malformed("unexpected DSML markup before tool_calls")
        }
        guard let blockClose = text.range(
            of: m.callsClose,
            range: blockOpen.upperBound..<text.endIndex
        ) else {
            throw StrictError.malformed("unclosed tool_calls block")
        }

        let suffix = text[blockClose.upperBound...]
        guard suffix.allSatisfy(\.isWhitespace) else {
            throw StrictError.malformed("non-whitespace content after tool_calls")
        }

        let limit = blockClose.lowerBound
        var cursor = blockOpen.upperBound
        var calls: [ToolCall] = []
        skipWhitespace(in: text, cursor: &cursor, limit: limit)
        while cursor < limit {
            let invokePrefix = "<\(m.dsml)invoke"
            guard text[cursor..<limit].hasPrefix(invokePrefix) else {
                throw StrictError.malformed("unexpected content in tool_calls")
            }
            calls.append(try parseInvokeStrict(
                in: text,
                cursor: &cursor,
                limit: limit,
                markup: m,
                index: calls.count
            ))
            skipWhitespace(in: text, cursor: &cursor, limit: limit)
        }

        guard !calls.isEmpty else {
            throw StrictError.malformed("tool_calls contains no invocation")
        }
        let visible = String(prefix).trimmingCharacters(in: .whitespacesAndNewlines)
        return (calls, visible)
    }

    /// Conservative recovery for a common low-bit model defect: every invoke
    /// and parameter is complete, but generation ends immediately after the
    /// final `</invoke>` and omits only the outer `</tool_calls>` envelope.
    ///
    /// The normal strict parser remains unchanged. Callers must supply an
    /// explicit read-only allow-list; a write/edit/delete invocation is never
    /// made executable by this repair path.
    public static func parseRepairingReadOnlyEnvelope(
        _ text: String,
        markup m: ToolMarkup,
        allowedToolNames: Set<String>
    ) throws -> (calls: [ToolCall], visibleText: String) {
        do { return try parseStrict(text, markup: m) }
        catch let originalError {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.contains(m.callsOpen),
                  !trimmed.contains(m.callsClose),
                  trimmed.hasSuffix(m.invokeClose) else {
                throw originalError
            }

            let repaired = trimmed + m.callsClose
            let parsed = try parseStrict(repaired, markup: m)
            guard !parsed.calls.isEmpty,
                  parsed.calls.allSatisfy({ allowedToolNames.contains($0.name) }) else {
                throw originalError
            }
            return parsed
        }
    }

    /// Defensive display cleanup: remove (possibly malformed) tool-call markup the
    /// model spelled out as plain text — a leaked ｜DSML｜ fragment or a degraded
    /// "<tool_c:…>" that did NOT parse into a real call. Cuts from the first such
    /// marker to the end of the message: once the model starts emitting broken
    /// markup the tail is junk, and the actual tool result is shown in its own
    /// bubble. The markers are tool-specific (fullwidth bars never occur in prose
    /// or code), so ordinary text — including math like "5 < 3" — is untouched.
    public static func stripLeakedMarkup(_ text: String, markup m: ToolMarkup) -> String {
        let markers = [m.dsml, "<tool_call", "</tool_call", "<tool_c:", "</tool_c"]
        var cut = text.endIndex
        for marker in markers {
            if let r = text.range(of: marker), r.lowerBound < cut { cut = r.lowerBound }
        }
        guard cut < text.endIndex else { return text }
        return String(text[text.startIndex..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Parse one `<DSML|invoke name="…"> …params…` body into a ToolCall.
    static func parseInvoke(_ body: String, markup m: ToolMarkup, index: Int) -> ToolCall? {
        guard let name = attributeValue("name", in: body) else { return nil }
        var params: [(name: String, isString: Bool, value: String)] = []
        var search = body.startIndex
        let paramPrefix = "<\(m.dsml)parameter"
        while let po = body.range(of: paramPrefix, range: search..<body.endIndex) {
            guard let tagEnd = body.range(of: ">", range: po.upperBound..<body.endIndex) else { break }
            let tag = String(body[po.lowerBound..<tagEnd.upperBound])
            let pname = attributeValue("name", in: tag) ?? ""
            let isString = (attributeValue("string", in: tag) ?? "true") == "true"
            let valStart = tagEnd.upperBound
            let pclose = body.range(of: m.paramClose, range: valStart..<body.endIndex)?.lowerBound ?? body.endIndex
            let value = String(body[valStart..<pclose]).trimmingCharacters(in: .whitespacesAndNewlines)
            params.append((pname, isString, value))
            search = pclose < body.endIndex ? body.index(after: pclose) : body.endIndex
            if search >= body.endIndex { break }
        }
        return ToolCall(id: "call_\(index)", name: name, argumentsJSON: paramsToJSON(params))
    }

    /// Strict counterpart of `parseInvoke`.  `cursor` starts at the invocation's
    /// opening tag and is advanced past its closing tag only on success.
    private static func parseInvokeStrict(
        in text: String,
        cursor: inout String.Index,
        limit: String.Index,
        markup m: ToolMarkup,
        index: Int
    ) throws -> ToolCall {
        let tagEnd = try openingTagEnd(in: text, from: cursor, limit: limit)
        let tag = String(text[cursor...tagEnd])
        let attributes = try strictAttributes(in: tag, element: "invoke", markup: m)
        guard attributes.keys.count == 1, let name = attributes["name"] else {
            throw StrictError.malformed("invoke must contain exactly one name attribute")
        }
        guard isValidIdentifier(name) else {
            throw StrictError.malformed("invoke has an invalid or empty name")
        }
        cursor = text.index(after: tagEnd)

        var params: [(name: String, isString: Bool, value: String)] = []
        var parameterNames = Set<String>()
        while true {
            skipWhitespace(in: text, cursor: &cursor, limit: limit)
            guard cursor < limit else {
                throw StrictError.malformed("unclosed invoke \(name)")
            }
            if text[cursor..<limit].hasPrefix(m.invokeClose) {
                cursor = text.index(cursor, offsetBy: m.invokeClose.count)
                return ToolCall(id: "call_\(index)", name: name, argumentsJSON: paramsToJSON(params))
            }

            let parameterPrefix = "<\(m.dsml)parameter"
            guard text[cursor..<limit].hasPrefix(parameterPrefix) else {
                throw StrictError.malformed("unexpected content inside invoke \(name)")
            }
            let parameter = try parseParameterStrict(
                in: text,
                cursor: &cursor,
                limit: limit,
                markup: m
            )
            guard parameterNames.insert(parameter.name).inserted else {
                throw StrictError.malformed("duplicate parameter \(parameter.name)")
            }
            params.append(parameter)
        }
    }

    /// Parse one complete strict parameter and advance `cursor` past its close.
    private static func parseParameterStrict(
        in text: String,
        cursor: inout String.Index,
        limit: String.Index,
        markup m: ToolMarkup
    ) throws -> (name: String, isString: Bool, value: String) {
        let tagEnd = try openingTagEnd(in: text, from: cursor, limit: limit)
        let tag = String(text[cursor...tagEnd])
        let attributes = try strictAttributes(in: tag, element: "parameter", markup: m)
        let names = Set(attributes.keys)
        let standardShape = names == Set(["name", "string"])
        // DeepSeek V4 occasionally substitutes the attribute name `array` for
        // `string` while keeping the trained `false` value around a JSON array.
        // It carries exactly the same unquoted-JSON meaning and is safe to
        // normalize only in this two-attribute, false-valued shape.
        let arrayAliasShape = names == Set(["name", "array"])
            && attributes["array"] == "false"
        // Another compact form emitted by the model puts a JSON number directly
        // in a `number` attribute and leaves the parameter body empty:
        //   <parameter name="from_line" number="120"></parameter>
        // Accept only that exact, unambiguous shape; it is canonicalized to the
        // same non-string JSON value used by `string="false">120`.
        let numberValueShape = names == Set(["name", "number"])
        guard (standardShape || arrayAliasShape || numberValueShape),
              let name = attributes["name"] else {
            throw StrictError.malformed(
                "parameter requires name plus string, array=false, or a numeric number attribute"
            )
        }
        let stringFlag = standardShape ? (attributes["string"] ?? "") : "false"
        guard isValidIdentifier(name) else {
            throw StrictError.malformed("parameter has an invalid or empty name")
        }
        guard stringFlag == "true" || stringFlag == "false" else {
            throw StrictError.malformed("parameter string attribute must be true or false")
        }

        let valueStart = text.index(after: tagEnd)
        guard let close = text.range(of: m.paramClose, range: valueStart..<limit) else {
            throw StrictError.malformed("unclosed parameter \(name)")
        }
        let rawValue = text[valueStart..<close.lowerBound]
        guard !rawValue.contains(m.dsml) else {
            throw StrictError.malformed("nested DSML markup in parameter \(name)")
        }
        let bodyValue = String(rawValue).trimmingCharacters(in: .whitespacesAndNewlines)
        let value: String
        if numberValueShape {
            guard bodyValue.isEmpty,
                  let number = attributes["number"],
                  isJSONNumber(number) else {
                throw StrictError.malformed(
                    "number-valued parameter \(name) must have an empty body and a valid JSON number"
                )
            }
            value = number
        } else {
            value = bodyValue
        }
        let isString = stringFlag == "true"
        if !isString {
            guard let data = value.data(using: .utf8),
                  (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) != nil else {
                throw StrictError.malformed("non-string parameter \(name) is not valid JSON")
            }
        }
        cursor = close.upperBound
        return (name, isString, value)
    }

    /// Locate the `>` which ends an opening tag, ignoring escaped quotes and
    /// rejecting an unterminated quoted attribute.
    private static func openingTagEnd(
        in text: String,
        from start: String.Index,
        limit: String.Index
    ) throws -> String.Index {
        var cursor = start
        var inQuote = false
        var escaped = false
        while cursor < limit {
            let character = text[cursor]
            if escaped {
                escaped = false
            } else if character == "\\", inQuote {
                escaped = true
            } else if character == "\"" {
                inQuote.toggle()
            } else if character == ">", !inQuote {
                return cursor
            }
            cursor = text.index(after: cursor)
        }
        throw StrictError.malformed(inQuote ? "unterminated attribute" : "unterminated opening tag")
    }

    /// Parse an XML-ish opening tag while rejecting missing, duplicate, and
    /// unknown syntax.  Attribute order and whitespace around `=` are accepted.
    private static func strictAttributes(
        in tag: String,
        element: String,
        markup m: ToolMarkup
    ) throws -> [String: String] {
        let prefix = "<\(m.dsml)\(element)"
        guard tag.hasPrefix(prefix), tag.hasSuffix(">") else {
            throw StrictError.malformed("invalid \(element) opening tag")
        }

        var cursor = tag.index(tag.startIndex, offsetBy: prefix.count)
        let end = tag.index(before: tag.endIndex)
        if cursor < end, !tag[cursor].isWhitespace {
            throw StrictError.malformed("invalid \(element) element name")
        }

        var result: [String: String] = [:]
        while true {
            while cursor < end, tag[cursor].isWhitespace { cursor = tag.index(after: cursor) }
            if cursor == end { return result }

            let nameStart = cursor
            while cursor < end, isAttributeNameCharacter(tag[cursor]) {
                cursor = tag.index(after: cursor)
            }
            guard nameStart < cursor else {
                throw StrictError.malformed("invalid attribute in \(element)")
            }
            let name = String(tag[nameStart..<cursor])
            while cursor < end, tag[cursor].isWhitespace { cursor = tag.index(after: cursor) }
            guard cursor < end, tag[cursor] == "=" else {
                throw StrictError.malformed("attribute \(name) has no equals sign")
            }
            cursor = tag.index(after: cursor)
            while cursor < end, tag[cursor].isWhitespace { cursor = tag.index(after: cursor) }
            guard cursor < end, tag[cursor] == "\"" else {
                throw StrictError.malformed("attribute \(name) is not quoted")
            }
            cursor = tag.index(after: cursor)

            var value = ""
            var closed = false
            while cursor < end {
                let character = tag[cursor]
                if character == "\\" {
                    let next = tag.index(after: cursor)
                    guard next < end else {
                        throw StrictError.malformed("unterminated escape in attribute \(name)")
                    }
                    if tag[next] == "\"" || tag[next] == "\\" {
                        value.append(tag[next])
                        cursor = tag.index(after: next)
                    } else {
                        throw StrictError.malformed("invalid escape in attribute \(name)")
                    }
                } else if character == "\"" {
                    cursor = tag.index(after: cursor)
                    closed = true
                    break
                } else {
                    value.append(character)
                    cursor = tag.index(after: cursor)
                }
            }
            guard closed else {
                throw StrictError.malformed("unterminated attribute \(name)")
            }
            guard result.updateValue(value, forKey: name) == nil else {
                throw StrictError.malformed("duplicate attribute \(name)")
            }
        }
    }

    private static func skipWhitespace(
        in text: String,
        cursor: inout String.Index,
        limit: String.Index
    ) {
        while cursor < limit, text[cursor].isWhitespace { cursor = text.index(after: cursor) }
    }

    private static func isAttributeNameCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "_" || character == "-"
    }

    /// Tool and parameter identifiers are eventually used as registry/JSON keys;
    /// keep control characters, whitespace, quotes, and markup out of them.
    private static func isValidIdentifier(_ value: String) -> Bool {
        !value.isEmpty && value.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" || $0 == "."
        }
    }

    /// Strict JSON number grammar. This deliberately excludes booleans, quoted
    /// strings, NaN/Infinity, leading zeroes, and any trailing content.
    private static func isJSONNumber(_ value: String) -> Bool {
        value.range(
            of: #"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#,
            options: .regularExpression
        ) != nil
    }

    /// Build a JSON arguments object from parsed DSML parameters.
    static func paramsToJSON(_ params: [(name: String, isString: Bool, value: String)]) -> String {
        var parts: [String] = []
        for p in params {
            let key = jsonString(p.name)
            let val: String
            if p.isString {
                val = jsonString(p.value)
            } else if let d = p.value.data(using: .utf8),
                      (try? JSONSerialization.jsonObject(with: d, options: [.fragmentsAllowed])) != nil {
                val = p.value
            } else {
                val = jsonString(p.value)
            }
            parts.append("\(key):\(val)")
        }
        return "{" + parts.joined(separator: ",") + "}"
    }

    /// Extract `attr="value"` from an XML-ish tag/body. Backslash-escaped quotes
    /// inside the value (\" ) are honored and unescaped, so a name like
    /// `say_\"hi\"` doesn't truncate the value at the first quote.
    static func attributeValue(_ attr: String, in s: String) -> String? {
        guard let r = s.range(of: "\(attr)=\"") else { return nil }
        var value = ""
        var i = r.upperBound
        while i < s.endIndex {
            let c = s[i]
            if c == "\\", s.index(after: i) < s.endIndex, s[s.index(after: i)] == "\"" {
                value.append("\"")
                i = s.index(i, offsetBy: 2)
            } else if c == "\"" {
                return value
            } else {
                value.append(c)
                i = s.index(after: i)
            }
        }
        return nil   // unterminated attribute
    }
}
