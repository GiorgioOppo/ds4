import Foundation

/// Web search. Backends are pluggable through `WebSearchProvider`;
/// the bundled `DuckDuckGoLite` provider scrapes the no-JS endpoint
/// `https://html.duckduckgo.com/html/` — works without API keys but
/// is rate-limited and may break when the page changes. For
/// production usage configure a real provider (Brave, Tavily, Serper)
/// via the host's settings.
public struct WebSearchTool: Tool {
    public let provider: WebSearchProvider

    public init(provider: WebSearchProvider = DuckDuckGoLiteProvider()) {
        self.provider = provider
    }

    public var schema: ToolSchema {
        ToolSchema(
            name: "websearch",
            description:
                "Search the web and return a short ranked list of {title, " +
                "url, snippet}. Use 'webfetch' to read a result page in full.",
            category: .network,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "query": SchemaBuilder.string(description: "Search query."),
                    "limit": SchemaBuilder.integer(description: "Max results. Default 5.", minimum: 1),
                ],
                required: ["query"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "websearch '\(input["query"] as? String ?? "?")'"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let query = try input.string("query")
        let limit = input.optionalInteger("limit") ?? 5
        let results = try await provider.search(query: query, limit: limit)
        if results.isEmpty {
            return ToolOutput(output: "no results", metadata: ["count": "0"])
        }
        let formatted = results.enumerated().map { i, r in
            "[\(i + 1)] \(r.title)\n    \(r.url)\n    \(r.snippet)"
        }.joined(separator: "\n\n")
        return ToolOutput(output: formatted, metadata: ["count": "\(results.count)"])
    }
}

public struct WebSearchResult: Sendable {
    public let title: String
    public let url: String
    public let snippet: String

    public init(title: String, url: String, snippet: String) {
        self.title = title
        self.url = url
        self.snippet = snippet
    }
}

public protocol WebSearchProvider: Sendable {
    func search(query: String, limit: Int) async throws -> [WebSearchResult]
}

/// Default scraping-based provider. Fragile by design — included so
/// `websearch` works out-of-the-box for casual use; replace with a
/// real API for anything that matters.
public struct DuckDuckGoLiteProvider: WebSearchProvider {
    public init() {}

    public func search(query: String, limit: Int) async throws -> [WebSearchResult] {
        guard let escaped = query.addingPercentEncoding(
                withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://html.duckduckgo.com/html/?q=\(escaped)") else {
            throw ToolError.invalidInput("could not build query URL")
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 15
        req.setValue("Mozilla/5.0 DeepSeek-V4-Pro-MacOS", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse,
              200..<300 ~= http.statusCode else {
            throw ToolError.external("websearch backend returned non-2xx")
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw ToolError.external("non-UTF8 body")
        }
        return parseResults(html: html, limit: limit)
    }

    /// Very brittle HTML parsing — looks for `<a class="result__a"
    /// href="…">title</a>` followed by `<a class="result__snippet">
    /// snippet</a>`. Will silently return empty if DDG changes
    /// templates. Documented limitation.
    private func parseResults(html: String, limit: Int) -> [WebSearchResult] {
        let pattern = #"<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>[\s\S]*?<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)</a>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let nsHtml = html as NSString
        let matches = regex.matches(in: html, range: NSRange(location: 0, length: nsHtml.length))
        var out: [WebSearchResult] = []
        for m in matches.prefix(limit) {
            guard m.numberOfRanges >= 4 else { continue }
            let urlEnc = nsHtml.substring(with: m.range(at: 1))
            let title = decode(nsHtml.substring(with: m.range(at: 2)))
            let snippet = decode(stripTags(nsHtml.substring(with: m.range(at: 3))))
            let cleanURL = unwrapDDGRedirect(urlEnc)
            out.append(WebSearchResult(title: title, url: cleanURL, snippet: snippet))
        }
        return out
    }

    private func unwrapDDGRedirect(_ raw: String) -> String {
        // DDG wraps as //duckduckgo.com/l/?uddg=<url-encoded>&…
        guard let range = raw.range(of: "uddg=") else { return raw }
        let tail = String(raw[range.upperBound...])
        let stop = tail.firstIndex(of: "&") ?? tail.endIndex
        let encoded = String(tail[..<stop])
        return encoded.removingPercentEncoding ?? raw
    }

    private func stripTags(_ s: String) -> String {
        guard let re = try? NSRegularExpression(pattern: "<[^>]+>") else { return s }
        let ns = s as NSString
        return re.stringByReplacingMatches(
            in: s, range: NSRange(location: 0, length: ns.length), withTemplate: "")
    }

    private func decode(_ s: String) -> String {
        var t = s
        for (e, c) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                        ("&quot;", "\""), ("&#x27;", "'"), ("&#39;", "'")] {
            t = t.replacingOccurrences(of: e, with: c)
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
