import Foundation
import DS4Core

extension ToolRegistry {
    /// Fetch metadata for a page before deciding whether it is a useful source.
    static let webPageInfo = BuiltinTool(
        spec: ToolSpec(name: "web_page_info",
                       description: "Fetch lightweight page metadata for an http(s) URL: final URL, MIME type, title, description, canonical URL, and published/modified date hints when present. Blocks local/private addresses.",
                       parametersJSON: #"{"type":"object","properties":{"url":{"type":"string","description":"absolute http(s) URL"}},"required":["url"]}"#),
        run: { argsJSON in
            guard let url = stringArg(argsJSON, "url"), !url.isEmpty else {
                return #"{"error":"missing 'url' argument"}"#
            }
            do {
                let r = try WebClient.get(url, accept: "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5")
                let raw = String(data: r.body, encoding: .utf8)
                    ?? String(data: r.body, encoding: .isoLatin1) ?? ""
                let isHTML = r.mime.contains("html") || raw.range(of: "(?i)<html|<!doctype html", options: .regularExpression) != nil
                guard isHTML else {
                    return #"{"url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))","status":\#(r.status),"mime":"\#(WebClient.jsonEscape(r.mime))","bytes":\#(r.body.count)}"#
                }
                let info = pageMetadata(raw)
                return #"{"url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))","status":\#(r.status),"mime":"\#(WebClient.jsonEscape(r.mime))","title":"\#(WebClient.jsonEscape(info.title))","description":"\#(WebClient.jsonEscape(info.description))","canonical":"\#(WebClient.jsonEscape(info.canonical))","published":"\#(WebClient.jsonEscape(info.published))","modified":"\#(WebClient.jsonEscape(info.modified))"}"#
            } catch {
                return #"{"error":"\#(WebClient.jsonEscape("\(error)"))"}"#
            }
        })

    /// Fetch several pages in one tool call for cross-checking sources.
    static let webFetchMany = BuiltinTool(
        spec: ToolSpec(name: "web_fetch_many",
                       description: "Fetch up to 4 http(s) URLs and return readable text for each page. Use for cross-checking multiple search results. Blocks local/private addresses.",
                       parametersJSON: #"{"type":"object","properties":{"urls":{"type":"array","items":{"type":"string"},"description":"absolute http(s) URLs, max 4"},"max_chars_per_page":{"type":"integer","description":"text cap per page, default 4000, max 8000"}},"required":["urls"]}"#),
        run: { argsJSON in
            guard let urls = stringArrayArg(argsJSON, "urls"), !urls.isEmpty else {
                return #"{"error":"missing 'urls' argument"}"#
            }
            let maxChars = max(500, min(intArg(argsJSON, "max_chars_per_page") ?? 4000, 8000))
            let items = urls.prefix(4).map { url -> String in
                do {
                    let r = try WebClient.get(url, accept: "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5")
                    guard (200..<300).contains(r.status) else {
                        return #"{"url":"\#(WebClient.jsonEscape(url))","error":"HTTP \#(r.status)"}"#
                    }
                    let raw = String(data: r.body, encoding: .utf8)
                        ?? String(data: r.body, encoding: .isoLatin1) ?? ""
                    let isHTML = r.mime.contains("html") || raw.range(of: "(?i)<html|<!doctype html", options: .regularExpression) != nil
                    let text = isHTML ? WebClient.htmlToText(raw, limit: maxChars)
                                      : (raw.count > maxChars ? String(raw.prefix(maxChars)) + "\n...[truncated]" : raw)
                    let title = isHTML ? pageMetadata(raw).title : ""
                    return #"{"url":"\#(WebClient.jsonEscape(r.finalURL.absoluteString))","status":\#(r.status),"mime":"\#(WebClient.jsonEscape(r.mime))","title":"\#(WebClient.jsonEscape(title))","content":"\#(WebClient.jsonEscape(text))"}"#
                } catch {
                    return #"{"url":"\#(WebClient.jsonEscape(url))","error":"\#(WebClient.jsonEscape("\(error)"))"}"#
                }
            }.joined(separator: ",")
            return #"{"results":[\#(items)]}"#
        })

    private static func stringArrayArg(_ json: String, _ key: String) -> [String]? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let arr = obj[key] as? [String] { return arr.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } }
        if let s = obj[key] as? String, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [s] }
        return nil
    }

    private struct PageMetadata {
        var title = ""
        var description = ""
        var canonical = ""
        var published = ""
        var modified = ""
    }

    private static func pageMetadata(_ html: String) -> PageMetadata {
        var info = PageMetadata()
        info.title = firstMatch(html, pattern: #"(?is)<title[^>]*>(.*?)</title>"#)
            .map { WebClient.htmlToText($0, limit: 300) } ?? ""
        info.description = metaContent(html, names: ["description", "og:description", "twitter:description"])
        info.canonical = linkHref(html, rel: "canonical")
        if info.canonical.isEmpty { info.canonical = metaContent(html, names: ["og:url"]) }
        info.published = metaContent(html, names: ["article:published_time", "date", "pubdate", "publishdate", "dc.date"])
        info.modified = metaContent(html, names: ["article:modified_time", "last-modified", "modified", "dc.modified"])
        return info
    }

    private static func metaContent(_ html: String, names: [String]) -> String {
        for name in names {
            let escaped = NSRegularExpression.escapedPattern(for: name)
            let patterns = [
                #"(?is)<meta[^>]*(?:name|property)=["']\#(escaped)["'][^>]*content=["']([^"']+)["'][^>]*>"#,
                #"(?is)<meta[^>]*content=["']([^"']+)["'][^>]*(?:name|property)=["']\#(escaped)["'][^>]*>"#,
            ]
            for pattern in patterns {
                if let match = firstMatch(html, pattern: pattern) { return WebClient.htmlToText(match, limit: 500) }
            }
        }
        return ""
    }

    private static func linkHref(_ html: String, rel: String) -> String {
        let escaped = NSRegularExpression.escapedPattern(for: rel)
        let patterns = [
            #"(?is)<link[^>]*rel=["'][^"']*\#(escaped)[^"']*["'][^>]*href=["']([^"']+)["'][^>]*>"#,
            #"(?is)<link[^>]*href=["']([^"']+)["'][^>]*rel=["'][^"']*\#(escaped)[^"']*["'][^>]*>"#,
        ]
        for pattern in patterns {
            if let match = firstMatch(html, pattern: pattern) { return WebClient.htmlToText(match, limit: 500) }
        }
        return ""
    }

    private static func firstMatch(_ text: String, pattern: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
        let ns = text as NSString
        guard let m = re.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              m.numberOfRanges > 1 else { return nil }
        return ns.substring(with: m.range(at: 1))
    }
}
