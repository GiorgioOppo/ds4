import Foundation
import DS4Core

extension ToolRegistry {
    /// Web search via DuckDuckGo's keyless HTML endpoint. Returns the top
    /// results as JSON (title, url, snippet) for the model to then read with
    /// web_fetch. No API key; SSRF-guarded through WebClient. The endpoint is
    /// overridable with DS4_SEARCH_URL (must contain "%@" for the query) for
    /// users who prefer a keyed search API behind a compatible HTML shape.
    static let webSearch = BuiltinTool(
        spec: ToolSpec(name: "web_search",
                       description: "Search the web and return the top results (title, url, snippet). Follow up with web_fetch to read a result's page.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"},"count":{"type":"integer","description":"max results, default 5"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query"), !q.trimmingCharacters(in: .whitespaces).isEmpty else {
                return #"{"error":"missing 'query' argument"}"#
            }
            let count = max(1, min(intArg(argsJSON, "count") ?? 5, 10))
            let encoded = q.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? q
            let template = ProcessInfo.processInfo.environment["DS4_SEARCH_URL"]
                ?? "https://html.duckduckgo.com/html/?q=%@"
            let urlString = template.replacingOccurrences(of: "%@", with: encoded)
            do {
                let r = try WebClient.get(urlString, accept: "text/html,application/xhtml+xml")
                guard (200..<300).contains(r.status) else {
                    return #"{"error":"search HTTP \#(r.status)"}"#
                }
                let html = String(data: r.body, encoding: .utf8)
                    ?? String(data: r.body, encoding: .isoLatin1) ?? ""
                let results = parseDuckDuckGo(html, limit: count)
                if results.isEmpty {
                    return #"{"query":"\#(WebClient.jsonEscape(q))","results":[],"note":"nessun risultato estratto (formato del motore cambiato? imposta DS4_SEARCH_URL)"}"#
                }
                let items = results.map {
                    #"{"title":"\#(WebClient.jsonEscape($0.title))","url":"\#(WebClient.jsonEscape($0.url))","snippet":"\#(WebClient.jsonEscape($0.snippet))"}"#
                }.joined(separator: ",")
                return #"{"query":"\#(WebClient.jsonEscape(q))","results":[\#(items)]}"#
            } catch {
                return #"{"error":"\#(WebClient.jsonEscape("\(error)"))"}"#
            }
        })

    /// Parse the DuckDuckGo HTML results page. Each result is an
    /// `<a class="result__a" href="…">title</a>` plus a `result__snippet`.
    /// DDG wraps external links as /l/?uddg=<encoded real url> — unwrapped here.
    private struct WebResult { let title: String; let url: String; let snippet: String }
    private static func parseDuckDuckGo(_ html: String, limit: Int) -> [WebResult] {
        var out: [WebResult] = []
        let anchorRE = try? NSRegularExpression(
            pattern: #"<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#,
            options: [.dotMatchesLineSeparators, .caseInsensitive])
        let snippetRE = try? NSRegularExpression(
            pattern: #"class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>"#,
            options: [.dotMatchesLineSeparators, .caseInsensitive])
        let ns = html as NSString
        let anchors = anchorRE?.matches(in: html, range: NSRange(location: 0, length: ns.length)) ?? []
        let snippets = snippetRE?.matches(in: html, range: NSRange(location: 0, length: ns.length)) ?? []
        func strip(_ s: String) -> String { WebClient.htmlToText(s, limit: 400) }
        for (i, m) in anchors.enumerated() where out.count < limit {
            let href = ns.substring(with: m.range(at: 1))
            let title = strip(ns.substring(with: m.range(at: 2)))
            let snippet = i < snippets.count ? strip(ns.substring(with: snippets[i].range(at: 1))) : ""
            out.append(WebResult(title: title, url: unwrapDDG(href), snippet: snippet))
        }
        return out
    }

    /// DDG redirect links look like //duckduckgo.com/l/?uddg=<pct-encoded url>&…
    private static func unwrapDDG(_ href: String) -> String {
        guard href.contains("uddg=") else {
            return href.hasPrefix("//") ? "https:" + href : href
        }
        if let range = href.range(of: "uddg=") {
            let tail = String(href[range.upperBound...])
            let enc = tail.prefix { $0 != "&" }
            if let dec = String(enc).removingPercentEncoding { return dec }
        }
        return href.hasPrefix("//") ? "https:" + href : href
    }
}
