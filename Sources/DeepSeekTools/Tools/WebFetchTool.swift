import Foundation

/// Fetch a URL and return its body. Text content is returned verbatim;
/// HTML pages are stripped to plain text (a very small HTML→text pass,
/// not a full DOM parse — when the model needs structure it should
/// ask for raw HTML via `raw=true`). 1 MB body cap.
public struct WebFetchTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "webfetch",
            description:
                "Effettua una GET di una URL e restituisce il body della risposta. L'HTML viene ridotto a " +
                "testo semplice, a meno che raw=true. Limitato a 1 MB.",
            category: .network,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "url": SchemaBuilder.string(description: "URL assoluta (solo http/https)."),
                    "raw": SchemaBuilder.boolean(description: "Restituisce il body così com'è. Default false."),
                ],
                required: ["url"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "webfetch \(input["url"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let urlString = try input.string("url")
        let raw = input.optionalBool("raw") ?? false
        guard let url = URL(string: urlString),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw ToolError.invalidInput("URL must be http(s)")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("DeepSeek-V4-Pro-MacOS/1.0", forHTTPHeaderField: "User-Agent")
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw ToolError.external(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw ToolError.external("non-HTTP response")
        }
        if !(200..<300).contains(http.statusCode) {
            throw ToolError.external("HTTP \(http.statusCode)")
        }
        let cap = 1_048_576
        let body = data.count > cap ? data.prefix(cap) : data
        guard let text = String(data: Data(body), encoding: .utf8) else {
            throw ToolError.invalidInput("body is not valid UTF-8 (\(data.count) bytes)")
        }
        let isHTML = http.value(forHTTPHeaderField: "Content-Type")?
            .lowercased().contains("text/html") ?? false
        let output = (raw || !isHTML) ? text : stripHTML(text)
        return ToolOutput(
            output: output,
            metadata: [
                "status": "\(http.statusCode)",
                "bytes": "\(data.count)",
                "content-type": http.value(forHTTPHeaderField: "Content-Type") ?? "",
            ]
        )
    }

    /// Very basic HTML→text reducer: strip script/style blocks, remove
    /// other tags, collapse whitespace, decode a handful of common
    /// entities. For anything serious the model should ask for raw.
    private func stripHTML(_ html: String) -> String {
        let scriptRE = try? NSRegularExpression(
            pattern: "<(script|style)[^>]*>[\\s\\S]*?</\\1>",
            options: .caseInsensitive)
        var s = html as NSString
        if let scriptRE {
            let range = NSRange(location: 0, length: s.length)
            s = scriptRE.stringByReplacingMatches(
                in: s as String, range: range, withTemplate: "") as NSString
        }
        let tagRE = try? NSRegularExpression(pattern: "<[^>]+>")
        if let tagRE {
            let range = NSRange(location: 0, length: s.length)
            s = tagRE.stringByReplacingMatches(
                in: s as String, range: range, withTemplate: "") as NSString
        }
        var result = s as String
        for (entity, char) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                                ("&quot;", "\""), ("&#39;", "'"), ("&nbsp;", " ")] {
            result = result.replacingOccurrences(of: entity, with: char)
        }
        let collapsed = try? NSRegularExpression(pattern: "\n\\s*\n\\s*\n+")
        if let collapsed {
            let nsResult = result as NSString
            let range = NSRange(location: 0, length: nsResult.length)
            result = collapsed.stringByReplacingMatches(
                in: result, range: range, withTemplate: "\n\n")
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
