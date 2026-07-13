import Foundation

// MARK: - Parsing tool calls from generated text

public enum ToolCallParser {
    /// Extract tool calls from a completed assistant message and return the visible
    /// text (the DSML tool-call block stripped).
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

